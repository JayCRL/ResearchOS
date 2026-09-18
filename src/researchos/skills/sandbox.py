"""The skill sandbox: static and behavioural analysis over *text*, never execution of skill code.

Why this module exists, and what it deliberately is not
------------------------------------------------------
The fastest way to evaluate an external skill is to run it. ResearchOS refuses, because running
downloaded code inside the research project is precisely how raw evidence gets overwritten and
provenance gets destroyed. So the sandbox is honest about its own limits:

* ``SandboxReport.static_analysis_only`` is always ``True``. Nothing here imports, executes or
  subprocesses a skill — the report says so, in the artifact itself, so nobody can mistake a static
  verdict for a runtime guarantee.
* Findings are *heuristics with named codes*, reviewed by a human, not a proof of safety. The
  severity split is what makes them usable: a delete of ``evidence/raw`` is a ``BLOCKER``; an
  unknown licence is ``LOW``. Only ``HIGH``/``BLOCKER`` findings (or any touch of a protected path)
  fail a skill, so an advisory-only skill with a sloppy README still passes.
* The declared capability ceiling is the second gate. An external skill may only *propose*; a
  capability that declares a write, network use or code execution exceeds that ceiling and is not
  granted. The violation is reported rather than raised, because a sandbox verdict must be a
  document a human can read and disagree with.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import EXTERNAL_SKILL_CAPABILITIES, Cap, Principal
from ..models import SandboxFinding, SandboxReport, SkillCapability, SkillCard
from ..models.common import Severity, dedupe_preserve_order
from ..models.timeline import TimelineEventKind

#: Paths that hold the research record. Any *mutation* of one of these is a data-integrity event,
#: not a style issue: raw evidence, research state, provenance, claims, audits and the event log.
#: Patterns are matched against path-normalised text (backslashes become ``/``).
PROTECTED_PATH_PATTERNS: tuple[str, ...] = (
    r"\.researchos/state",
    r"evidence/raw",
    r"provenance",
    r"claims/",
    r"audits?/",
    r"state/events\.jsonl",
    r"state/research_state\.yaml",
)

#: Path class -> (finding code, severity) for a mutation detected in the neighbourhood of that path.
#: Order matters: the most specific class wins.
_PROTECTED_PATH_CODES: tuple[tuple[str, str, Severity], ...] = (
    (r"evidence/raw", "DELETES_RAW_EVIDENCE", Severity.BLOCKER),
    (r"state/events\.jsonl", "HIDES_AUDIT", Severity.BLOCKER),
    (r"audits?/", "HIDES_AUDIT", Severity.HIGH),
    (r"provenance", "MODIFIES_PROVENANCE", Severity.HIGH),
    (r"claims/", "REWRITES_CORE_CLAIM", Severity.HIGH),
    (r"experiments/", "MODIFIES_RESEARCH_STATE", Severity.BLOCKER),
    (r"\.researchos/state", "MODIFIES_RESEARCH_STATE", Severity.BLOCKER),
    (r"state/research_state\.yaml", "MODIFIES_RESEARCH_STATE", Severity.BLOCKER),
)

#: The declared denylist: ``(regex, finding code, severity)`` scanned over the skill's text.
#:
#: The first group are behaviour patterns. The second group re-states the protected-path codes in
#: their inline form (``shutil.rmtree("evidence/raw")``); the neighbourhood scan below catches the
#: split form (path bound to a variable one line earlier). Belt and braces is intentional: missing a
#: delete of raw evidence is the one failure this module must not have. The third group are
#: declaration patterns for problems normally read off the ``SkillCard`` — they exist so a manifest
#: shipped *inside* a skill package is flagged before installation too.
DEFAULT_DENYLIST: tuple[tuple[str, str, Severity], ...] = (
    # --- behaviour
    (r"\b(?:subprocess|os\.system|os\.popen|pty\.spawn|commands\.getoutput)\b", "SHELL_EXECUTION", Severity.HIGH),
    (r"\bshell\s*=\s*True\b", "SHELL_EXECUTION", Severity.HIGH),
    (r"\b(?:eval|exec)\s*\(", "DYNAMIC_CODE_EXECUTION", Severity.HIGH),
    (r"\b(?:__import__|compile|pickle\.loads|marshal\.loads|types\.FunctionType)\s*\(", "DYNAMIC_CODE_EXECUTION", Severity.HIGH),
    (r"\b(?:requests\.|httpx\.|urllib\.request|urlopen|socket\.socket|aiohttp\.|http\.client)\b", "NETWORK_ACCESS", Severity.MEDIUM),
    (r"\b(?:curl|wget)\s+[-a-z]", "NETWORK_ACCESS", Severity.MEDIUM),
    # --- protected record, inline form
    (r"(?:rmtree|unlink|remove|removedirs|rmdir|truncate|move|replace)\s*\([^)\n]*evidence[/\\]raw", "DELETES_RAW_EVIDENCE", Severity.BLOCKER),
    (r"(?:write_text|write_bytes|writelines|safe_dump|dump|open)\s*\([^)\n]*\.researchos[/\\]state", "MODIFIES_RESEARCH_STATE", Severity.BLOCKER),
    (r"(?:rmtree|unlink|remove|removedirs|rmdir)\s*\([^)\n]*state[/\\]events\.jsonl", "HIDES_AUDIT", Severity.BLOCKER),
    (r"(?:write_text|write_bytes|writelines|safe_dump|dump|open)\s*\([^)\n]*provenance", "MODIFIES_PROVENANCE", Severity.HIGH),
    (r"(?:rmtree|unlink|remove|removedirs|rmdir|truncate|open)\s*\([^)\n]*audits?/", "HIDES_AUDIT", Severity.HIGH),
    (r"(?:rmtree|unlink|remove|removedirs|rmdir|truncate|open)\s*\([^)\n]*claims/", "REWRITES_CORE_CLAIM", Severity.HIGH),
    (r"(?:rmtree|unlink|remove|removedirs|rmdir)\s*\([^)\n]*(?:fail|FAIL)", "REMOVES_FAILED_EXPERIMENT", Severity.BLOCKER),
    # --- declaration surface (normally read off the card; catches an embedded manifest)
    (r"(?im)^\s*(?:install_requires|dependencies|requires)\s*[:=]\s*\[[^\]\n]*\b(?:latest|\*|any)\b", "UNPINNED_DEPENDENCY", Severity.MEDIUM),
    (r"(?im)^\s*(?:licen[cs]e)\s*[:=]\s*[\"']?\s*(?:unknown|none|n/?a|tbd)?\s*[\"']?\s*$", "LICENSE_UNKNOWN", Severity.LOW),
    (r"(?im)^\s*(?:tests?|test_command)\s*[:=]\s*\[\s*\]\s*$", "NO_TESTS", Severity.LOW),
    (r"(?im)^\s*(?:requires_network|network)\s*[:=]\s*(?:true|yes)\s*$", "NETWORK_ACCESS", Severity.MEDIUM),
    (r"(?im)^\s*(?:requires_execution|exec)\s*[:=]\s*(?:true|yes)\s*$", "OVERBROAD_CAPABILITY", Severity.HIGH),
    (r"(?im)^\s*writes?\s*[:=]\s*\[?[^\]\n]*(?:evidence/raw|\.researchos/state|claims/|audits?/|provenance)", "UNDECLARED_WRITE", Severity.HIGH),
)

#: Codes that describe an *operation on the research record*. These are the codes
#: :meth:`SkillSandbox.check_operation` refuses at runtime, not just at review time.
PROTECTED_OPERATION_CODES: frozenset[str] = frozenset(
    {
        "DELETES_RAW_EVIDENCE",
        "MODIFIES_RESEARCH_STATE",
        "MODIFIES_PROVENANCE",
        "HIDES_AUDIT",
        "REMOVES_FAILED_EXPERIMENT",
        "REWRITES_CORE_CLAIM",
    }
)

_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.BLOCKER,
)

#: A call that changes something. Requiring the ``(`` keeps prose ("removes the failed run") out.
_MUTATION_CALL = re.compile(
    r"\b(?:rmtree|unlink|remove|removedirs|rmdir|truncate|rename|replace|move|copy|copy2|copyfile|"
    r"write_text|write_bytes|writelines|safe_dump|dump_all|dump|save|to_yaml|makedirs?)\s*\(",
    re.IGNORECASE,
)
_DESTRUCTIVE_CALL = re.compile(
    r"\b(?:rmtree|unlink|remove|removedirs|rmdir|truncate|rename|move|os\.replace)\s*\(",
    re.IGNORECASE,
)
_OPEN_WRITE = re.compile(r"\bopen\s*\([^)\n]*[\"'][rwax]?[wax+]")
_WRITE_TARGET = re.compile(
    r"\b(?:write_text|write_bytes|writelines|safe_dump|dump_all|to_yaml|save)\s*\(\s*[\"']([^\"'\n]{1,200})[\"']"
)
_OPEN_WRITE_TARGET = re.compile(r"\bopen\s*\(\s*[\"']([^\"'\n]{1,200})[\"']\s*,\s*[\"'][rwax]?[wax+]")

#: Directories an external skill may write to without exceeding the proposal ceiling. Everything
#: else is either the research record (never writable) or someone else's state.
_SCRATCH_PREFIXES: tuple[str, ...] = ("scratch/", "cache/", "tmp/", "sandbox/")

_UNKNOWN_LICENSES: frozenset[str] = frozenset({"", "unknown", "none", "n/a", "na", "tbd", "unlicensed"})
_UNPINNED_VERSIONS: frozenset[str] = frozenset({"", "*", "latest", "any", "unpinned"})


def _severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER.index(severity)


def _normalise_text(text: str) -> str:
    return text.replace("\\", "/")


def _path_class(text: str) -> tuple[str, str, Severity] | None:
    """First protected path class present in ``text``, or ``None``."""
    for pattern, code, severity in _PROTECTED_PATH_CODES:
        if re.search(pattern, text):
            if pattern.startswith(r"experiments/"):
                code = "REMOVES_FAILED_EXPERIMENT" if re.search(r"fail", text, re.IGNORECASE) else code
            return pattern, code, severity
    return None


class SkillSandbox:
    """Static/behavioural sandbox and declared-capability ceiling for external skills."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ================================================================== analysis

    def inspect(
        self,
        principal: Principal,
        card: SkillCard,
        *,
        content: str | None = None,
        path: str | Path | None = None,
    ) -> SandboxReport:
        """Produce and persist a sandbox verdict for ``card``.

        ``content`` and ``path`` are alternative ways to hand over the skill's *text*. With neither,
        only the card's own declarations are checked — a legitimate, weaker inspection, recorded as
        such in ``notes``. Nothing is executed under any circumstance.
        """
        principal.require(Cap.SKILL_EVALUATE, f"skill.sandbox:{card.skill_id}")

        text: str | None = content
        source = "card declarations only"
        if text is None and path is not None:
            target = Path(path)
            if not target.is_file():
                raise SkillError(
                    f"sandbox cannot inspect {target}: not a file", resource=str(target)
                )
            text = target.read_text(encoding="utf-8", errors="replace")
            source = f"text of {target}"
        elif text is not None:
            source = "supplied content"

        findings: list[SandboxFinding] = []
        protected_paths: list[str] = []
        files_written: list[str] = []
        denied_operations: list[str] = []
        network_used = False

        # --- card-level declarations
        findings.extend(self._card_findings(card))
        for capability in card.capabilities:
            for target in capability.writes:
                files_written.append(target)
            if capability.requires_network:
                network_used = True

        granted, violations = self._ceiling_verdict(card)
        for code, target, severity, message in violations:
            findings.append(
                SandboxFinding(
                    code=code,
                    severity=severity,
                    message=message,
                    location=f"capability:{target}",
                )
            )
            if code == "UNDECLARED_WRITE":
                denied_operations.append(f"write:{target}")
            elif code == "OVERBROAD_CAPABILITY":
                denied_operations.append(f"capability:{target}")

        # --- text-level behaviour
        if text is not None:
            normalised = _normalise_text(text)
            lines = normalised.splitlines()
            for index, line in enumerate(lines):
                window = "\n".join(lines[max(0, index - 1) : index + 2])
                if _MUTATION_CALL.search(window) or _OPEN_WRITE.search(window):
                    found = _path_class(window)
                    if found is not None:
                        _, code, severity = found
                        protected = next(
                            (
                                pattern
                                for pattern in PROTECTED_PATH_PATTERNS
                                if re.search(pattern, window)
                            ),
                            None,
                        )
                        if protected:
                            protected_paths.append(protected)
                        findings.append(
                            SandboxFinding(
                                code=code,
                                severity=severity,
                                message=(
                                    "a mutating call operates on a protected research path "
                                    f"({protected or 'protected path'}); this is data loss, not a "
                                    "style issue"
                                ),
                                location=f"line {index + 1}",
                                snippet=line.strip()[:200],
                            )
                        )
                for pattern, code, severity in DEFAULT_DENYLIST:
                    if re.search(pattern, line):
                        findings.append(
                            SandboxFinding(
                                code=code,
                                severity=severity,
                                message=self._message_for(code),
                                location=f"line {index + 1}",
                                snippet=line.strip()[:200],
                            )
                        )
                        if code == "NETWORK_ACCESS":
                            network_used = True
                        if code in PROTECTED_OPERATION_CODES:
                            found = _path_class(window)
                            if found is not None:
                                protected_paths.append(found[0])
            findings.extend(self._undeclared_writes(card, normalised, files_written))

        findings = self._dedupe(findings)
        passed = not any(f.severity in (Severity.HIGH, Severity.BLOCKER) for f in findings)
        if protected_paths:
            passed = False

        report = SandboxReport(
            skill_id=card.skill_id,
            passed=passed,
            findings=findings,
            capability_ceiling=granted,
            denied_operations=dedupe_preserve_order(denied_operations),
            network_used=network_used,
            files_written=dedupe_preserve_order(files_written),
            protected_paths_touched=dedupe_preserve_order(
                list(protected_paths) + self._declared_protected_writes(card)
            ),
            static_analysis_only=True,
            created_by=principal.name,
            notes=[
                "static and behavioural analysis over text only: no skill code was imported, "
                "executed or subprocessed",
                f"analysed {source}",
                "findings are heuristics for human review, not a proof of safety",
            ],
        )
        self.kernel.sandbox_reports.save(report)
        self.kernel.events.append(
            "skill.sandboxed",
            actor=principal.name,
            payload={
                "skill_id": card.skill_id,
                "sandbox_report_id": report.sandbox_report_id,
                "passed": report.passed,
                "findings": len(report.findings),
                "blocking": len(
                    [
                        f
                        for f in report.findings
                        if f.severity in (Severity.HIGH, Severity.BLOCKER)
                    ]
                ),
                "protected_paths_touched": report.protected_paths_touched,
                "static_analysis_only": True,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill sandboxed: {card.name}",
            detail=(
                f"{card.skill_id}: {'passed' if report.passed else 'FAILED'} "
                f"({len(report.findings)} finding(s), static analysis only)"
            ),
            actor=principal.name,
            refs=[card.skill_id, report.sandbox_report_id],
            state_revision=self.kernel.state.revision(),
        )
        return report

    def declared_ceiling(self, card: SkillCard) -> list[str]:
        """Capabilities actually granted to the skill, validated against the external whitelist.

        Entries are values from :data:`~researchos.kernel.permissions.EXTERNAL_SKILL_CAPABILITIES`
        (``propose.*``): an external skill proposes. A declared capability that wants to write,
        execute or reach the network is *not* granted — see :meth:`ceiling_violations` for why, and
        :meth:`inspect` for the findings that record it.
        """
        granted, _ = self._ceiling_verdict(card)
        return granted

    def ceiling_violations(self, card: SkillCard) -> list[str]:
        """Human-readable reasons a declared capability exceeds the external ceiling."""
        _, violations = self._ceiling_verdict(card)
        return [f"{code}: {target} — {message}" for code, target, _, message in violations]

    def check_operation(self, operation: str) -> SandboxFinding | None:
        """Judge a *runtime* operation string; return the worst finding, or ``None`` if permitted.

        The sandbox cannot run a skill, but the kernel can refuse what a skill asks for. This is the
        function that turns the static denylist into an enforced boundary: an operation naming a
        protected path, or a shell/exec/network primitive, never returns ``None``.
        """
        normalised = _normalise_text(operation)
        candidates: list[SandboxFinding] = []
        found = _path_class(normalised)
        if found is not None and (
            _MUTATION_CALL.search(normalised) or _OPEN_WRITE.search(normalised)
        ):
            _, code, severity = found
            candidates.append(
                SandboxFinding(
                    code=code,
                    severity=severity,
                    message=self._message_for(code),
                    location="operation",
                    snippet=operation.strip()[:200],
                )
            )
        for pattern, code, severity in DEFAULT_DENYLIST:
            if re.search(pattern, normalised):
                candidates.append(
                    SandboxFinding(
                        code=code,
                        severity=severity,
                        message=self._message_for(code),
                        location="operation",
                        snippet=operation.strip()[:200],
                    )
                )
        if not candidates:
            return None
        return max(candidates, key=lambda f: (_severity_rank(f.severity), f.code))

    # ================================================================== internals

    def _card_findings(self, card: SkillCard) -> list[SandboxFinding]:
        findings: list[SandboxFinding] = []
        if (card.license or "").strip().lower() in _UNKNOWN_LICENSES:
            findings.append(
                SandboxFinding(
                    code="LICENSE_UNKNOWN",
                    severity=Severity.LOW,
                    message=(
                        "the licence is not declared; an artifact without a licence cannot be "
                        "installed (advisory for sandboxing, blocking for installation)"
                    ),
                    location="card.license",
                )
            )
        if not card.tests:
            findings.append(
                SandboxFinding(
                    code="NO_TESTS",
                    severity=Severity.LOW,
                    message=(
                        "the skill ships no tests of its own; the ResearchOS benchmark will have to "
                        "carry the whole evaluation burden"
                    ),
                    location="card.tests",
                )
            )
        for dependency in card.dependencies:
            version = (dependency.version or "").strip().lower()
            if dependency.optional:
                continue
            if version in _UNPINNED_VERSIONS:
                findings.append(
                    SandboxFinding(
                        code="UNPINNED_DEPENDENCY",
                        severity=Severity.MEDIUM,
                        message=(
                            f"dependency {dependency.name!r} is unpinned, so a benchmark result "
                            "cannot be reproduced later"
                        ),
                        location=f"dependency:{dependency.name}",
                    )
                )
        return findings

    def _ceiling_verdict(
        self, card: SkillCard
    ) -> tuple[list[str], list[tuple[str, str, Severity, str]]]:
        """Return ``(granted capabilities, violations)`` for the card's declaration.

        This is the ceiling check. A protected-path write is rejected *and* reported as
        ``UNDECLARED_WRITE`` at ``HIGH`` severity, so it also fails the report — the rejection is
        not merely a missing grant.
        """
        granted: list[str] = []
        violations: list[tuple[str, str, Severity, str]] = []
        for capability in card.capabilities:
            proposed = self._propose_capability(capability)
            over: list[tuple[str, str, Severity, str]] = []
            for target in capability.writes:
                normalised = _normalise_text(target).lstrip("./")
                if _path_class(normalised) is not None:
                    over.append(
                        (
                            "UNDECLARED_WRITE",
                            target,
                            Severity.HIGH,
                            f"declares a write to protected research path {target!r}; raw evidence, "
                            "research state, provenance, claims and audits are never writable by an "
                            "external skill (invariant 7)",
                        )
                    )
                elif not normalised.startswith(_SCRATCH_PREFIXES):
                    over.append(
                        (
                            "UNDECLARED_WRITE",
                            target,
                            Severity.MEDIUM,
                            f"declares a write to {target!r}, outside the sandbox scratch area; an "
                            "external skill may only propose",
                        )
                    )
            if capability.requires_execution:
                over.append(
                    (
                        "OVERBROAD_CAPABILITY",
                        capability.name,
                        Severity.HIGH,
                        f"capability {capability.name!r} requires code execution, which the "
                        "proposal-only ceiling does not include",
                    )
                )
            if capability.requires_network:
                over.append(
                    (
                        "OVERBROAD_CAPABILITY",
                        capability.name,
                        Severity.MEDIUM,
                        f"capability {capability.name!r} declares network access; recorded so a "
                        "reviewer can see it, and not part of the granted ceiling",
                    )
                )
            if not capability.is_advisory_only and not capability.writes:
                over.append(
                    (
                        "OVERBROAD_CAPABILITY",
                        capability.name,
                        Severity.MEDIUM,
                        f"capability {capability.name!r} is declared as mutating but names no write "
                        "target; the ceiling cannot bound it",
                    )
                )
            if over:
                violations.extend(over)
                continue
            if proposed in EXTERNAL_SKILL_CAPABILITIES:
                granted.append(proposed.value)
        return dedupe_preserve_order(granted), violations

    @staticmethod
    def _propose_capability(capability: SkillCapability) -> Cap:
        """Map a free-form capability name onto the external proposal whitelist.

        Capability names in a :class:`SkillCard` are free text ("literature review", "peer review"),
        while the permission system is a closed enum. The mapping is deliberately conservative: an
        unknown capability is granted only as a *recommendation*, and anything that actually asks
        for writes, execution or network is rejected earlier and never reaches this mapping.
        """
        name = capability.name.lower()
        if any(word in name for word in ("draft", "writ", "prose", "style", "edit", "language")):
            return Cap.PROPOSE_DRAFT
        if any(word in name for word in ("analysis", "statistic", "metric", "compute", "audit")):
            return Cap.PROPOSE_ANALYSIS
        if any(word in name for word in ("procedure", "method", "design", "protocol", "search", "review")):
            return Cap.PROPOSE_PROCEDURE
        return Cap.PROPOSE_RECOMMENDATION

    def _undeclared_writes(
        self, card: SkillCard, text: str, files_written: list[str]
    ) -> list[SandboxFinding]:
        """Writes visible in the text that the card never declared."""
        declared = [
            _normalise_text(target).lstrip("./")
            for capability in card.capabilities
            for target in capability.writes
        ]
        targets = [m.group(1) for m in _WRITE_TARGET.finditer(text)]
        targets += [m.group(1) for m in _OPEN_WRITE_TARGET.finditer(text)]
        findings: list[SandboxFinding] = []
        for target in dedupe_preserve_order(targets):
            normalised = _normalise_text(target).lstrip("./")
            if _path_class(normalised) is not None:
                continue  # already reported with a protected-path code
            if normalised.startswith(_SCRATCH_PREFIXES):
                files_written.append(target)
                continue
            if any(
                normalised == write or normalised.startswith(write.rstrip("/") + "/")
                for write in declared
            ):
                files_written.append(target)
                continue
            findings.append(
                SandboxFinding(
                    code="UNDECLARED_WRITE",
                    severity=Severity.MEDIUM,
                    message=(
                        f"the text writes to {target!r}, which no declared capability covers; "
                        "activity the ceiling cannot bound is activity the sandbox cannot bound"
                    ),
                    location=target,
                )
            )
        return findings

    @staticmethod
    def _declared_protected_writes(card: SkillCard) -> list[str]:
        touched: list[str] = []
        for capability in card.capabilities:
            for target in capability.writes:
                if _path_class(_normalise_text(target)) is not None:
                    touched.append(target)
        return touched

    @staticmethod
    def _dedupe(findings: Sequence[SandboxFinding]) -> list[SandboxFinding]:
        seen: set[tuple[str, str | None, str]] = set()
        out: list[SandboxFinding] = []
        for finding in findings:
            key = (finding.code, finding.location, finding.snippet or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(finding)
        return sorted(
            out,
            key=lambda f: (-_severity_rank(f.severity), f.code, f.location or "", f.snippet or ""),
        )

    @staticmethod
    def _message_for(code: str) -> str:
        return {
            "SHELL_EXECUTION": "spawns a shell/process; a sandboxed skill may not execute anything",
            "DYNAMIC_CODE_EXECUTION": "evaluates code at runtime, so static analysis cannot bound it",
            "NETWORK_ACCESS": "reaches the network; recorded so a reviewer can decide, not granted",
            "DELETES_RAW_EVIDENCE": "deletes or rewrites raw evidence, which is the ground truth of the project",
            "MODIFIES_RESEARCH_STATE": "writes research state directly instead of proposing a transition",
            "MODIFIES_PROVENANCE": "rewrites provenance; provenance written at record time is immutable",
            "HIDES_AUDIT": "mutates an audit or event record; the audit trail is append-only",
            "REMOVES_FAILED_EXPERIMENT": "removes a failed experiment; failed experiments are results",
            "REWRITES_CORE_CLAIM": "rewrites a claim record instead of filing a transition",
            "UNPINNED_DEPENDENCY": "depends on an unpinned version, so results cannot be reproduced",
            "UNDECLARED_WRITE": "writes outside its declared capability set",
            "OVERBROAD_CAPABILITY": "declares capability beyond the proposal-only ceiling",
            "LICENSE_UNKNOWN": "licence not declared",
            "NO_TESTS": "ships no tests",
        }.get(code, "sandbox finding")


__all__ = [
    "DEFAULT_DENYLIST",
    "PROTECTED_OPERATION_CODES",
    "PROTECTED_PATH_PATTERNS",
    "SkillSandbox",
]

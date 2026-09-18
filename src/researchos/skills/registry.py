"""The Skill Registry: the single door into the skill plane, and the router that picks a skill.

Why this module exists
----------------------
External skills are untrusted input. The registry is where that assumption is encoded:

* **A new registration is a claim, not a promotion.** Whatever the incoming ``SkillCard`` asserts,
  a *new* skill enters at ``DISCOVERED`` with ``UNTRUSTED`` trust. Status and trust are earned
  afterwards by the sandbox and the benchmark harness, never self-certified by the manifest.
* **Registration is a proposal.** It requires ``skill.propose``; nothing an agent does here makes a
  skill usable.
* **Routing is provenance-gated.** :meth:`SkillRegistry.route` only ever returns an ``ACTIVE`` skill
  whose ``is_usable()`` is true, and it breaks ties by ``skill_id`` so two runs on the same registry
  choose the same skill. A router that returned a different skill per invocation would make every
  benchmark result unattributable.
* **Installation is a human act, and it is not activation.** :meth:`SkillRegistry.install` records
  that a human accepted a pinned, licensed, sandbox-passing artifact. It raises trust to
  ``SANDBOXED`` — the value the sandbox report already earned — and deliberately does *not* move the
  lifecycle status: status transitions belong to :mod:`researchos.skills.lifecycle`.
"""

from __future__ import annotations

from typing import Sequence, cast

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models import (
    Audit,
    AuditKind,
    AuditVerdict,
    BenchmarkRun,
    Finding,
    SandboxReport,
    SkillCard,
    SkillSource,
    SkillTrust,
    SkillVersion,
)
from ..models.common import Severity, SkillStatus, dedupe_preserve_order, sha256_text
from ..models.timeline import TimelineEventKind

#: Statuses whose ACTIVE/VERIFIED members the router may consider.
_USABLE_STATUSES = (SkillStatus.VERIFIED, SkillStatus.ACTIVE)

#: Trust levels that mean "something was actually run and checked".
_EARNED_TRUST = (SkillTrust.SANDBOXED, SkillTrust.BENCHMARKED, SkillTrust.TRUSTED)


def _normalise(value: str) -> str:
    """Lowercase, separator-free form used for capability matching."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in value.lower())
    return " ".join(cleaned.split())


def _capability_matches(query: str, candidate: str) -> bool:
    """True when ``query`` names ``candidate`` (exact, or one is a whole-word subsequence).

    Word-boundary padding keeps ``analysis`` from matching ``meta-analysis-ish`` while still letting
    ``statistical analysis`` satisfy a request for ``analysis``.
    """
    if not query or not candidate:
        return False
    if query == candidate:
        return True
    return f" {query} " in f" {candidate} " or f" {candidate} " in f" {query} "


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator <= 0 else round(numerator / denominator, 4)


class SkillRegistry:
    """Read-through view over ``kernel.skills`` / ``kernel.skill_versions`` plus the router."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ================================================================== reads

    def get(self, skill_id: str) -> SkillCard | None:
        return self.kernel.skills.get(skill_id)

    def require(self, skill_id: str) -> SkillCard:
        """Like :meth:`get`, but raises ``SkillError`` instead of returning ``None``."""
        card = self.get(skill_id)
        if card is None:
            raise SkillError(f"skill {skill_id!r} is not registered", resource=skill_id)
        return card

    def list(
        self,
        *,
        status: SkillStatus | None = None,
        source: SkillSource | None = None,
    ) -> list[SkillCard]:
        wanted_status = SkillStatus(status) if isinstance(status, str) else status
        wanted_source = SkillSource(source) if isinstance(source, str) else source
        cards = self.kernel.skills.all()
        if wanted_status is not None:
            cards = [c for c in cards if c.status is wanted_status]
        if wanted_source is not None:
            cards = [c for c in cards if c.source is wanted_source]
        return sorted(cards, key=lambda c: (c.name, c.skill_id))

    def active(self) -> list[SkillCard]:
        """ACTIVE skills, best quality first — the routing shortlist."""
        return sorted(
            self.list(status=SkillStatus.ACTIVE),
            key=lambda c: (-c.quality_metrics.score(), c.skill_id),
        )

    def by_capability(self, capability: str) -> list[SkillCard]:
        """The Skill Router's candidate set: every skill declaring ``capability``, best first.

        Deliberately *not* filtered by status — an auditor needs to see that a capability is only
        served by a DEPRECATED skill, which is exactly one of the registry-audit findings.
        """
        query = _normalise(capability)
        found: list[SkillCard] = []
        for card in self.kernel.skills.all():
            names = [_normalise(c.name) for c in card.capabilities]
            names.extend(_normalise(tag) for tag in card.tags)
            if any(_capability_matches(query, name) for name in names):
                found.append(card)
        return sorted(found, key=lambda c: (-c.quality_metrics.score(), c.skill_id))

    def route(self, capability: str) -> SkillCard | None:
        """Best ACTIVE, usable skill for ``capability``, or ``None``.

        Ordering is ``quality_metrics.score()`` then ``skill_id``: deterministic, so a benchmark
        recorded against "the skill the router picks" stays attributable. A skill that is only
        VERIFIED (sandboxed but not yet activated) is never returned: routing is for activated
        skills only.
        """
        candidates = [
            card
            for card in self.by_capability(capability)
            if card.status is SkillStatus.ACTIVE and card.is_usable()
        ]
        return candidates[0] if candidates else None

    def versions(self, skill_id: str) -> list[SkillVersion]:
        found = [v for v in self.kernel.skill_versions.all() if v.skill_id == skill_id]
        return sorted(found, key=lambda v: (v.created_at, v.skill_version_id))

    def latest_version(self, skill_id: str) -> SkillVersion | None:
        found = self.versions(skill_id)
        return found[-1] if found else None

    def health(self) -> dict[str, float | int]:
        """Counts per status plus one 0..1 score, built from components with stated denominators.

        Empty denominators count as 1.0 ("nothing broken"), so an empty registry is not reported as
        unhealthy. Every component is reported separately in spirit: the score is a summary, and the
        counts in the same dict are what an operator should look at first.
        """
        cards = self.list()
        counts: dict[str, float | int] = {status.value: 0 for status in SkillStatus}
        for card in cards:
            counts[card.status.value] = int(counts[card.status.value]) + 1

        gaps = self.kernel.skill_gaps.all()
        open_gaps = [g for g in gaps if g.status == "OPEN"]
        addressed = [g for g in gaps if g.status == "ADDRESSED"]

        pinned = [c for c in cards if self._is_pinned(c)]
        usable = [c for c in cards if c.status in _USABLE_STATUSES]
        benchmarked = [c for c in usable if self._passing_run(c.skill_id) is not None]
        actives = [c for c in cards if c.status is SkillStatus.ACTIVE]
        trusted = [c for c in actives if c.trust in _EARNED_TRUST]

        components = (
            _ratio(len(pinned), len(cards)),
            _ratio(len(benchmarked), len(usable)),
            _ratio(len(trusted), len(actives)),
            _ratio(len(addressed), len(addressed) + len(open_gaps)),
        )
        counts["total"] = len(cards)
        counts["gaps_open"] = len(open_gaps)
        counts["gaps_addressed"] = len(addressed)
        counts["score"] = round(sum(components) / len(components), 4)
        return counts

    # ================================================================== writes

    def register(
        self,
        principal: Principal,
        card: SkillCard,
        *,
        version: str | None = None,
        change_summary: str = "",
    ) -> SkillCard:
        """Register a new skill: never a promotion, always a claim under evaluation.

        A *new* registration is forced to ``DISCOVERED`` + ``UNTRUSTED``. A manifest that arrives
        claiming ``ACTIVE``/``TRUSTED`` is downgraded (and the downgrade is recorded in the event
        payload), because trust is what the sandbox and the benchmark harness produce, not what a
        downloaded file asserts.
        """
        principal.require(Cap.SKILL_PROPOSE, f"skill.register:{card.skill_id}")
        if self.kernel.skills.get(card.skill_id) is not None:
            raise SkillError(
                f"skill {card.skill_id} is already registered; register a new card for a new skill "
                "or pin the new revision with pin_version()",
                principal=principal.name,
                resource=card.skill_id,
            )

        downgraded: list[str] = []
        updates: dict[str, object] = {}
        if card.status is not SkillStatus.DISCOVERED:
            downgraded.append(f"status {card.status.value}->DISCOVERED")
            updates["status"] = SkillStatus.DISCOVERED
        if card.trust is not SkillTrust.UNTRUSTED:
            downgraded.append(f"trust {card.trust.value}->UNTRUSTED")
            updates["trust"] = SkillTrust.UNTRUSTED
        saved = cast(SkillCard, card.with_updates(**updates)) if updates else card

        self.kernel.skills.save(saved)
        skill_version = SkillVersion(
            skill_id=saved.skill_id,
            version=version or saved.version,
            commit=saved.commit,
            content_hash=saved.content_hash,
            card_hash=saved.card_hash(),
            change_summary=change_summary or "initial registration",
        )
        self.kernel.skill_versions.save(skill_version)

        self.kernel.events.append(
            "skill.registered",
            actor=principal.name,
            payload={
                "skill_id": saved.skill_id,
                "name": saved.name,
                "source": saved.source.value,
                "status": saved.status.value,
                "trust": saved.trust.value,
                "skill_version_id": skill_version.skill_version_id,
                "downgraded": downgraded,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill registered: {saved.name}",
            detail=(
                f"{saved.skill_id} from {saved.source.value}; entering at DISCOVERED/UNTRUSTED"
                + ("; declared " + ", ".join(downgraded) + " refused" if downgraded else "")
            ),
            actor=principal.name,
            refs=[saved.skill_id, skill_version.skill_version_id],
            state_revision=self.kernel.state.revision(),
        )
        return saved

    def pin_version(
        self,
        principal: Principal,
        skill_id: str,
        *,
        commit: str,
        content_hash: str,
    ) -> SkillVersion:
        """Pin the skill to an exact commit + content digest.

        Benchmark results are only attributable if the artifact they measured can be named again.
        An unpinned skill can be sandboxed but can never be trusted, and the registry audit reports
        it — so pinning is a first-class operation rather than a manifest field.
        """
        principal.require(Cap.SKILL_PROPOSE, f"skill.pin:{skill_id}")
        card = self.require(skill_id)
        if not commit.strip():
            raise SkillError(f"a pin for {skill_id} needs a commit", resource=skill_id)
        digest = content_hash.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise SkillError(
                f"content_hash for {skill_id} must be a 64-character hex SHA-256, got "
                f"{content_hash!r}",
                resource=skill_id,
            )

        previous = self.latest_version(skill_id)
        skill_version = SkillVersion(
            skill_id=skill_id,
            version=card.version,
            commit=commit,
            content_hash=digest,
            card_hash=card.card_hash(),
            parent_skill_version_ids=[previous.skill_version_id] if previous else [],
            change_summary=f"pinned to {commit[:12]}",
        )
        self.kernel.skill_versions.save(skill_version)

        updated = cast(
            SkillCard, card.with_updates(commit=commit, content_hash=digest)
        )
        self.kernel.skills.save(updated)

        self.kernel.events.append(
            "skill.pinned",
            actor=principal.name,
            payload={
                "skill_id": skill_id,
                "commit": commit,
                "content_hash": digest,
                "skill_version_id": skill_version.skill_version_id,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill pinned: {card.name}",
            detail=f"{skill_id} pinned to {commit[:12]} ({digest[:12]})",
            actor=principal.name,
            refs=[skill_id, skill_version.skill_version_id],
            state_revision=self.kernel.state.revision(),
        )
        return skill_version

    def record_synthesis(
        self,
        principal: Principal,
        skill_id: str,
        *,
        rule: str,
        synthesis_inputs: Sequence[str],
    ) -> SkillVersion:
        """Mark the latest version as synthesized from parents, with the injected research rule.

        ``SkillCard`` cannot carry "this was generated", but ``SkillVersion`` can, and the claim
        "C = A + B + rule" has to be inspectable later: a synthesized skill is *not* a trusted skill,
        and the rule is the only part a human can review.
        """
        principal.require(Cap.SKILL_PROPOSE, f"skill.synthesis:{skill_id}")
        self.require(skill_id)
        if not rule.strip():
            raise SkillError(
                f"synthesis of {skill_id} requires the research-specific rule that was injected",
                resource=skill_id,
            )
        latest = self.latest_version(skill_id)
        if latest is None:
            raise SkillError(f"skill {skill_id} has no SkillVersion to mark", resource=skill_id)
        parent_versions: list[str] = []
        for parent_id in synthesis_inputs:
            parent_version = self.latest_version(parent_id)
            if parent_version is not None:
                parent_versions.append(parent_version.skill_version_id)
        updated = cast(
            SkillVersion,
            latest.with_updates(
                synthesized=True,
                synthesis_rule=rule,
                synthesis_inputs=list(synthesis_inputs),
                parent_skill_version_ids=parent_versions,
            ),
        )
        self.kernel.skill_versions.save(updated)
        self.kernel.events.append(
            "skill.synthesized",
            actor=principal.name,
            payload={
                "skill_id": skill_id,
                "skill_version_id": updated.skill_version_id,
                "synthesis_inputs": list(synthesis_inputs),
                "rule": rule[:300],
            },
        )
        return updated

    def install(self, principal: Principal, skill_id: str) -> SkillCard:
        """Human acceptance of an external artifact — explicitly *not* activation.

        Gates: human principal, ``skill.install``, a pinned commit + content digest, a known
        licence, and a passing sandbox report. The only state it changes is trust
        (``UNTRUSTED`` -> ``SANDBOXED``, which the sandbox report already justified) plus an
        installation record. Status stays where the lifecycle put it, so installation can never
        route a skill into research work.
        """
        principal.require(Cap.SKILL_INSTALL, f"skill.install:{skill_id}")
        if not principal.is_human:
            raise SkillError(
                f"only a human may install a skill; {principal.name!r} is {principal.kind.value}",
                principal=principal.name,
                resource=skill_id,
            )
        card = self.require(skill_id)
        if card.status is SkillStatus.DEPRECATED:
            raise SkillError(
                f"skill {skill_id} is DEPRECATED and cannot be installed", resource=skill_id
            )

        latest = self.latest_version(skill_id)
        commit = card.commit or (latest.commit if latest else None)
        digest = card.content_hash or (latest.content_hash if latest else None)
        if not commit or not digest:
            raise SkillError(
                f"cannot install {skill_id}: it is unpinned. Pin the artifact first with "
                "pin_version(commit=..., content_hash=...) so the benchmark measures a named revision",
                resource=skill_id,
            )
        if not card.license or card.license.strip().lower() in {"unknown", "none", "n/a", "unlicensed"}:
            raise SkillError(
                f"cannot install {skill_id}: its licence is unknown "
                f"({card.license!r}); an unlicensed artifact cannot enter a research project",
                resource=skill_id,
            )
        report = self._passing_sandbox_report(skill_id)
        if report is None:
            raise SkillError(
                f"cannot install {skill_id}: no passing sandbox report "
                "(run SkillSandbox.inspect first; static analysis is the gate, not a formality)",
                resource=skill_id,
            )

        installation = {
            "installed_by": principal.name,
            "commit": commit,
            "content_hash": digest,
            "sandbox_report_id": report.sandbox_report_id,
        }
        updated = cast(
            SkillCard,
            card.with_updates(
                trust=SkillTrust.SANDBOXED,
                benchmark={**card.benchmark, "installation": installation},
            ),
        )
        self.kernel.skills.save(updated)
        self.kernel.events.append(
            "skill.installed",
            actor=principal.name,
            payload={
                "skill_id": skill_id,
                "status": updated.status.value,
                "trust": updated.trust.value,
                "commit": commit,
                "sandbox_report_id": report.sandbox_report_id,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill installed: {card.name}",
            detail=(
                f"human accepted {skill_id} at {commit[:12]} (licence {card.license}); "
                "installation is not activation"
            ),
            actor=principal.name,
            refs=[skill_id, report.sandbox_report_id],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    def audit_registry(self, principal: Principal, *, task_id: str | None = None) -> Audit:
        """Audit the skill plane's provenance and gates; return the persisted :class:`Audit`.

        ``AuditKind`` has no SKILL member, and ``STYLE`` would be plainly wrong. ``PROVENANCE`` is
        the defensible choice: every finding here is about whether a claim of skill status can be
        traced to the commit, sandbox report and benchmark run that justified it. (See REQUESTED
        CHANGES: an ``AuditKind.SKILL`` member would be more precise.)
        """
        principal.require(Cap.AUDIT_CREATE, "skill.registry.audit")
        findings: list[Finding] = []

        for card in self.list():
            if card.status is SkillStatus.ACTIVE:
                run = self._run_for(card)
                if run is None:
                    findings.append(
                        Finding(
                            code="ACTIVE_WITHOUT_BENCHMARK_RUN",
                            severity=Severity.BLOCKER,
                            message=(
                                f"skill {card.name!r} is ACTIVE but its benchmark run "
                                f"{card.last_benchmark_run_id!r} cannot be found; activation is "
                                "unattributable"
                            ),
                            target_ref=card.skill_id,
                        )
                    )
                else:
                    ok, reasons = run.activatable()
                    if not ok:
                        findings.append(
                            Finding(
                                code="ACTIVE_BENCHMARK_NOT_ACTIVATABLE",
                                severity=Severity.HIGH,
                                message=(
                                    f"skill {card.name!r} is ACTIVE on a run that no longer clears "
                                    "the activation gate: " + "; ".join(reasons)
                                ),
                                target_ref=card.skill_id,
                            )
                        )
                if card.trust is SkillTrust.UNTRUSTED:
                    findings.append(
                        Finding(
                            code="ACTIVE_UNTRUSTED",
                            severity=Severity.BLOCKER,
                            message=(
                                f"skill {card.name!r} is ACTIVE while UNTRUSTED — it never passed "
                                "the sandbox"
                            ),
                            target_ref=card.skill_id,
                        )
                    )
            if not self._is_pinned(card):
                findings.append(
                    Finding(
                        code="UNPINNED_SKILL_PROVENANCE",
                        severity=Severity.HIGH,
                        message=(
                            f"skill {card.name!r} has no commit/content_hash pin; nothing it produced "
                            "can be attributed to a revision"
                        ),
                        target_ref=card.skill_id,
                    )
                )

        for card in self.list(status=SkillStatus.DEPRECATED):
            if not card.superseded_by:
                findings.append(
                    Finding(
                        code="DEPRECATED_WITHOUT_SUPERSEDOR",
                        severity=Severity.MEDIUM,
                        message=(
                            f"DEPRECATED skill {card.name!r} names no superseding skill; the "
                            "replacement is undocumented"
                        ),
                        target_ref=card.skill_id,
                    )
                )
            for capability in card.capabilities:
                replacements = [
                    other
                    for other in self.by_capability(capability.name)
                    if other.skill_id != card.skill_id and other.is_usable()
                ]
                if not replacements:
                    findings.append(
                        Finding(
                            code="DEPRECATED_STILL_ROUTABLE",
                            severity=Severity.HIGH,
                            message=(
                                f"capability {capability.name!r} is provided only by DEPRECATED skill "
                                f"{card.name!r}; the router has no replacement to fall back to"
                            ),
                            target_ref=card.skill_id,
                        )
                    )

        for gap in self.kernel.skill_gaps.all():
            if not gap.addressed_by_skill_id:
                findings.append(
                    Finding(
                        code="GAP_UNADDRESSED",
                        severity=Severity.HIGH if gap.occurrences >= 3 else Severity.MEDIUM,
                        message=(
                            f"skill gap seen {gap.occurrences}x is unaddressed: "
                            f"{gap.missing_capability} — {gap.description}"
                        ),
                        target_ref=gap.gap_id,
                    )
                )
            elif self.get(gap.addressed_by_skill_id) is None:
                findings.append(
                    Finding(
                        code="GAP_ADDRESSED_BY_MISSING_SKILL",
                        severity=Severity.HIGH,
                        message=(
                            f"gap {gap.gap_id} claims to be addressed by "
                            f"{gap.addressed_by_skill_id!r}, which is not registered"
                        ),
                        target_ref=gap.gap_id,
                    )
                )

        findings.sort(key=lambda f: (f.code, f.target_ref or ""))
        blocking = any(f.severity in (Severity.HIGH, Severity.BLOCKER) for f in findings)
        verdict = AuditVerdict.FAIL if blocking else (AuditVerdict.WARN if findings else AuditVerdict.PASS)
        audit = Audit(
            kind=AuditKind.PROVENANCE,
            title="Skill registry provenance audit",
            subjects=dedupe_preserve_order(
                [str(f.target_ref) for f in findings if f.target_ref]
            ),
            findings=findings,
            summary=(
                f"{len(findings)} finding(s) across {self.kernel.skills.count()} skill(s); "
                f"verdict {verdict.value}"
            ),
            verdict=verdict,
            created_by=principal.name,
            tool="researchos.skills.registry",
            task_id=task_id,
        )
        return self.kernel.save_audit(principal, audit)

    # ================================================================== helpers

    def _is_pinned(self, card: SkillCard) -> bool:
        if card.commit and card.content_hash:
            return True
        return any(v.commit and v.content_hash for v in self.versions(card.skill_id))

    def _run_for(self, card: SkillCard) -> BenchmarkRun | None:
        if card.last_benchmark_run_id:
            found = self.kernel.benchmark_runs.get(card.last_benchmark_run_id)
            if found is not None:
                return found
        return None

    def _latest_run(self, skill_id: str) -> BenchmarkRun | None:
        runs = [r for r in self.kernel.benchmark_runs.all() if r.skill_id == skill_id]
        if not runs:
            return None
        return max(runs, key=lambda r: (r.created_at, r.benchmark_run_id))

    def _passing_run(self, skill_id: str) -> BenchmarkRun | None:
        run = self._latest_run(skill_id)
        if run is None:
            return None
        ok, _ = run.activatable()
        return run if ok else None

    def _passing_sandbox_report(self, skill_id: str) -> SandboxReport | None:
        reports = [
            r
            for r in self.kernel.sandbox_reports.all()
            if r.skill_id == skill_id and r.passed
        ]
        if not reports:
            return None
        return max(reports, key=lambda r: (r.created_at, r.sandbox_report_id))

    @staticmethod
    def content_hash_of(text: str) -> str:
        """Convenience for callers that need a valid ``content_hash`` input."""
        return sha256_text(text)


__all__ = ["SkillRegistry"]

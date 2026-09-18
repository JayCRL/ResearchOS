"""Skill discovery: turning "we could not do X" into a candidate skill, honestly labelled.

Why this module exists
----------------------
A skill meta-system is only as good as the gaps it notices. Discovery is therefore *evidence-driven*
before it is search-driven:

* :meth:`SkillDiscoveryAgent.detect_gaps` reads the project's own records — audit findings, task
  findings, failed/aborted experiments, blocking conflicts — and reports a :class:`SkillGap` only
  when the **same gap signal repeats**. One occurrence is an incident; a repeat is a missing
  capability. Nothing here asks a model what the project "feels" it lacks.
* Queries are **capability-oriented** (:data:`CAPABILITY_QUERY_TEMPLATES`). "research agent" style
  queries return frameworks that do everything and are verifiable in nothing; "citation
  verification" returns a component whose behaviour can be benchmarked.
* A discovered skill is a **claim from a keyword search**, so :meth:`search` can only ever produce
  ``DISCOVERED``/``UNTRUSTED`` cards, and persists them exclusively through
  :meth:`SkillRegistry.register`. Discovery has no code path that writes ``ACTIVE``, and no code path
  that skips the registry — an unpinned repository hit is untrusted input, exactly like any other.
* :meth:`synthesise` implements ``A + B + research rule -> C``. The product is explicitly **not
  trusted**: it is filed ``CANDIDATE``/``UNTRUSTED``/``SYNTHESIZED`` with its parents and the rule
  recorded on its :class:`SkillVersion`, and it must still earn the sandbox and the benchmark.

Popularity is recorded, never scored: a repository's star count lands in ``card.benchmark
["source_signals"]`` for a human to look at. There is deliberately no quality metric for it —
``SkillQualityMetrics`` has no field for popularity, and this module does not smuggle one in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models import (
    ConflictKind,
    ExperimentStatus,
    FindingKind,
    SkillCapability,
    SkillCard,
    SkillGap,
    SkillSource,
    SkillTrust,
)
from ..models.common import Severity, SkillStatus, dedupe_preserve_order, utcnow
from ..models.timeline import TimelineEventKind
from .lifecycle import SkillLifecycle
from .registry import SkillRegistry

#: The capability vocabulary discovery searches in. Deliberately *component* capabilities rather
#: than roles: "literature review", not "research agent". A component can be sandboxed and
#: benchmarked; a do-everything agent cannot, so it cannot be trusted either.
CAPABILITY_AREAS: tuple[str, ...] = (
    "citation verification",
    "novelty search",
    "systematic review",
    "literature review",
    "experiment design",
    "statistical analysis",
    "causal inference",
    "mechanistic interpretability",
    "reproducibility",
    "paper reproduction",
    "figure audit",
    "latex",
    "benchmarking",
    "peer review",
    "scientific writing",
    "data analysis",
    "research engineering",
)

#: Query templates, rendered with ``{capability}``. Every one names a *capability* because that is
#: what gets benchmarked; none of them asks for a general-purpose research agent.
CAPABILITY_QUERY_TEMPLATES: tuple[str, ...] = (
    "{capability} open source tool",
    "{capability} python library",
    "{capability} github implementation",
    "{capability} cli component",
    "{capability} benchmark suite",
    "{capability} deterministic implementation",
)

#: Extra phrasings that identify each area in free text (audit messages, task finding statements).
#: Scanned in :data:`CAPABILITY_AREAS` order, so the more specific areas are listed first there.
_AREA_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "citation verification": ("citation", "doi", "arxiv id", "reference check"),
    "novelty search": ("novelty", "prior art", "prior-art", "closest prior work"),
    "systematic review": ("systematic review", "screening", "inclusion criteria"),
    "literature review": ("literature", "related work", "survey", "search coverage"),
    "experiment design": ("experiment design", "confound", "baseline", "ablation", "design defect"),
    "statistical analysis": ("statistic", "p-value", "significance", "multiple comparison", "seed count"),
    "causal inference": ("causal", "causation", "necessity", "sufficiency", "intervention"),
    "mechanistic interpretability": ("mechanism", "interpretab", "circuit", "probe", "activation"),
    "reproducibility": ("reproduc", "seed", "environment", "container", "determinism"),
    "paper reproduction": ("reproduce the paper", "paper reproduction", "replication of the paper"),
    "figure audit": ("figure", "plot", "chart", "axis"),
    "latex": ("latex", "bibtex", "tex file", "typeset"),
    "benchmarking": ("benchmark", "leaderboard", "evaluation harness"),
    "peer review": ("peer review", "reviewer", "referee"),
    "scientific writing": ("writing", "prose", "language", "tone", "drafting"),
    "data analysis": ("data analysis", "dataframe", "aggregation", "dataset statistics"),
    "research engineering": ("tooling", "infrastructure", "scripting", "pipeline plumbing"),
}

#: Which capability the default benchmark vocabulary's finding codes evidence. Codes absent here
#: fall back to a capability phrase derived from the code itself, so an unknown auditor code still
#: produces a readable gap instead of being dropped.
FINDING_CODE_CAPABILITIES: Mapping[str, str] = {
    "CITATION_UNRESOLVED": "citation verification",
    "COVERAGE_INSUFFICIENT": "literature review",
    "MISSED_TERMINOLOGY": "novelty search",
    "PRIOR_ART_MATRIX_INCOMPLETE": "novelty search",
    "NO_ANALYSIS_ARTIFACT": "reproducibility",
    "INSUFFICIENT_N": "statistical analysis",
    "MULTIPLE_COMPARISONS_UNCORRECTED": "statistical analysis",
    "BASELINE_NOT_MATCHED": "experiment design",
    "SEED_INCONSISTENCY": "reproducibility",
    "EXCLUSION_UNRECORDED": "reproducibility",
    "CAUSAL_LANGUAGE_WITHOUT_INTERVENTION": "causal inference",
    "NECESSITY_CLAIM_WITHOUT_REMOVAL": "causal inference",
    "SINGLE_SETTING_GENERALIZATION": "causal inference",
    "ALTERNATIVE_EXPLANATION_UNELIMINATED": "mechanistic interpretability",
    "UNGROUNDED_NUMBER": "scientific writing",
    "LANGUAGE_EXCEEDS_EVIDENCE": "scientific writing",
    "NOVELTY_UNSUPPORTED": "novelty search",
    "TEMPLATE_PHRASE": "scientific writing",
}

#: A failed run is a design capability gap; an aborted run a reproducibility one.
_FAILED_EXPERIMENT_CAPABILITY: Mapping[ExperimentStatus, tuple[str, Severity]] = {
    ExperimentStatus.FAILED: ("experiment design", Severity.HIGH),
    ExperimentStatus.ABORTED: ("reproducibility", Severity.MEDIUM),
}

#: A blocking conflict is a capability the project lacked when it produced the disagreement.
_CONFLICT_CAPABILITIES: Mapping[ConflictKind, str] = {
    ConflictKind.VALUE: "data analysis",
    ConflictKind.CLAIM: "causal inference",
    ConflictKind.METRIC: "benchmarking",
    ConflictKind.DESIGN: "experiment design",
    ConflictKind.LITERATURE: "citation verification",
    ConflictKind.PROVENANCE: "reproducibility",
}

#: Gap statuses that a re-run may refresh in place. A terminal gap (ADDRESSED / WONT_FIX) is never
#: reopened silently: a recurrence files a *new* gap, so history stays readable.
_LIVE_GAP_STATUSES: frozenset[str] = frozenset({"OPEN", "SEARCHING"})

_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.BLOCKER,
)

#: Result keys a provider may use, in the order this module prefers them. Providers are network
#: shims; naming their keys here keeps parsing a data question instead of a code path per provider.
_NAME_KEYS: tuple[str, ...] = ("name", "title", "full_name", "repository", "repo", "url")
_REPO_KEYS: tuple[str, ...] = ("repository", "repo", "html_url", "url", "clone_url", "full_name")
_COMMIT_KEYS: tuple[str, ...] = ("commit", "sha", "pinned_commit", "revision", "head_sha")
_LICENSE_KEYS: tuple[str, ...] = ("license", "spdx_id", "license_spdx", "license_name")
_STARS_KEYS: tuple[str, ...] = ("stars", "stargazers_count", "stars_count", "star_count")
_DESCRIPTION_KEYS: tuple[str, ...] = ("description", "summary", "readme_summary")
_ENTRYPOINT_KEYS: tuple[str, ...] = ("entrypoint", "entry_point", "skill_file", "path")
_DOCS_KEYS: tuple[str, ...] = ("documentation_url", "homepage", "docs_url", "docs")


def _severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER.index(severity)


def _first_str(result: Mapping[str, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("spdx_id") or value.get("name") or value.get("key")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _canonical_capability(text: str) -> str:
    """Map free text onto the capability vocabulary; unrecognised text is kept as its own phrase."""
    cleaned = " ".join(text.lower().replace("_", " ").split())
    for area in CAPABILITY_AREAS:
        if area in cleaned:
            return area
    for area in CAPABILITY_AREAS:
        if any(keyword in cleaned for keyword in _AREA_KEYWORDS.get(area, ())):
            return area
    return cleaned


def _capability_for_text(text: str) -> str | None:
    """The capability a free-text statement is about, or ``None`` when it names none.

    Returning ``None`` matters: ordinary observations ("the metric looked noisy") must not be
    promoted into capability gaps just because they exist.
    """
    cleaned = " ".join(text.lower().replace("_", " ").split())
    for area in CAPABILITY_AREAS:
        if area in cleaned or any(
            keyword in cleaned for keyword in _AREA_KEYWORDS.get(area, ())
        ):
            return area
    return None


def _capability_from_code(code: str) -> str:
    mapped = FINDING_CODE_CAPABILITIES.get(code)
    if mapped:
        return mapped
    return " ".join(code.lower().split("_")).strip()


def _max_severity(severities: Sequence[Severity]) -> Severity:
    if not severities:
        return Severity.MEDIUM
    return max(severities, key=_severity_rank)


@dataclass(frozen=True)
class _Observation:
    """One recorded fact that evidences a missing capability. Occurrences are counted in these."""

    key: str
    capability: str
    code: str
    origin: str
    evidence_id: str
    at: datetime
    severity: Severity


@dataclass
class _Group:
    """Observations sharing one gap key, accumulated for a single :class:`SkillGap`."""

    key: str
    observations: list[_Observation] = field(default_factory=list)

    @property
    def capability(self) -> str:
        return self.observations[0].capability

    @property
    def origins(self) -> list[str]:
        return dedupe_preserve_order(sorted({o.origin for o in self.observations}))

    @property
    def evidence(self) -> list[str]:
        return dedupe_preserve_order(sorted({o.evidence_id for o in self.observations}))

    @property
    def codes(self) -> list[str]:
        return dedupe_preserve_order(sorted({o.code for o in self.observations}))


class SkillDiscoveryAgent:
    """Finds capability gaps in the project's own records and sources candidate skills."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel
        self.registry = SkillRegistry(kernel)
        self.lifecycle = SkillLifecycle(kernel)

    # ================================================================== gap detection

    def detect_gaps(
        self, principal: Principal, *, min_occurrences: int = 2, task_id: str | None = None
    ) -> list[SkillGap]:
        """Deterministically report the capability gaps the project's records repeat.

        Sources: audit findings (``kernel.audits``), task findings of kind ``TOOLING``
        (``kernel.tasks_store``), failed/aborted experiments and blocking conflicts
        (``kernel.ledger.blocking()``). A gap key is the finding code (or the derived capability for
        task findings and experiments), so the *same* signal seen ``min_occurrences`` times — and
        not once — is what becomes a gap. ``occurrences`` is the number of contributing records, and
        re-running merges into the OPEN gap it already filed rather than filing a duplicate.

        A capability already served by an ``ACTIVE``, usable skill is not a gap.
        """
        principal.require(Cap.SKILL_DISCOVER, "skill.discover")
        if min_occurrences < 1:
            raise SkillError(f"min_occurrences must be >= 1, got {min_occurrences!r}")

        groups: dict[str, _Group] = {}
        for observation in self._observations(task_id=task_id):
            groups.setdefault(observation.key, _Group(key=observation.key)).observations.append(
                observation
            )

        gaps: list[SkillGap] = []
        for key in sorted(groups):
            group = groups[key]
            if len(group.observations) < min_occurrences:
                continue
            if self.registry.route(group.capability) is not None:
                continue
            gaps.append(self._file_gap(principal, group, task_id=task_id))
        return gaps

    def capability_queries(self, missing_capability: str) -> list[str]:
        """Capability-oriented queries for one missing capability; deterministic, order-preserving."""
        capability = _canonical_capability(missing_capability)
        if not capability:
            raise SkillError("a capability query set needs a capability to search for")
        queries = [capability]
        queries.extend(
            template.format(capability=capability) for template in CAPABILITY_QUERY_TEMPLATES
        )
        return dedupe_preserve_order(queries)

    # ================================================================== discovery search

    def search(
        self,
        principal: Principal,
        gap: SkillGap,
        *,
        provider: Callable[[str], Sequence[Mapping[str, object]]],
        max_results_per_query: int = 5,
    ) -> list[SkillCard]:
        """Search for candidates for ``gap`` and register each hit as ``DISCOVERED``/``UNTRUSTED``.

        ``provider`` is injected and may be anything network-shaped; its results are treated as
        untrusted text. Registration goes through :meth:`SkillRegistry.register` — the only writer —
        so every hit enters at ``DISCOVERED`` with ``UNTRUSTED`` trust and cannot be routed into
        research work. An already-registered repository is not registered twice.
        """
        principal.require(Cap.SKILL_DISCOVER, "skill.discover")
        if max_results_per_query < 1:
            raise SkillError(f"max_results_per_query must be >= 1, got {max_results_per_query!r}")
        if self.kernel.skill_gaps.get(gap.gap_id) is None:
            raise SkillError(
                f"gap {gap.gap_id!r} is not recorded; file it with detect_gaps before searching",
                resource=gap.gap_id,
            )

        queries = list(gap.search_queries) or self.capability_queries(gap.missing_capability)
        cards: list[SkillCard] = []
        seen_repositories: set[str] = set()
        for query in queries:
            results = provider(query)
            accepted = 0
            for result in results:
                if accepted >= max_results_per_query:
                    break
                if not isinstance(result, Mapping):
                    raise SkillError(
                        f"provider returned {type(result).__name__} for query {query!r}; expected a "
                        "mapping of repository metadata",
                        resource=gap.gap_id,
                    )
                card = self._card_from_result(principal, result, query=query, gap=gap)
                fingerprint = (card.repository or card.name).lower()
                if fingerprint in seen_repositories or self._already_registered(card):
                    continue
                seen_repositories.add(fingerprint)
                cards.append(
                    self.registry.register(
                        principal,
                        card,
                        change_summary=f"discovered for gap {gap.gap_id} via query {query!r}",
                    )
                )
                accepted += 1

        self._mark_searching(principal, gap, queries=queries, found=len(cards))
        return cards

    # ================================================================== synthesis

    def synthesise(
        self,
        principal: Principal,
        *,
        skill_ids: Sequence[str],
        rule: str,
        name: str,
        description: str,
        version: str = "0.1.0",
    ) -> SkillCard:
        """Combine parent skills under an injected research rule: ``A + B + rule -> C``.

        The product is a **new claim**, so it is filed at ``CANDIDATE`` with ``UNTRUSTED`` trust and
        ``SYNTHESIZED`` source, carrying its parents and the rule on the card *and* on its
        :class:`SkillVersion`. It inherits its parents' declared capabilities (a declaration, not a
        measurement) and nothing else: it is not installable, not routable, and not trusted until the
        sandbox and the benchmark say so.
        """
        principal.require(Cap.SKILL_PROPOSE, "skill.synthesis")
        if not rule.strip():
            raise SkillError(
                "synthesis requires the research-specific rule that was injected; without it the "
                "product is an undocumented merge of two unaudited skills"
            )
        if not name.strip():
            raise SkillError("a synthesized skill needs a name")
        parents = [
            self.registry.require(skill_id)
            for skill_id in dedupe_preserve_order(list(skill_ids))
        ]
        if not parents:
            raise SkillError("synthesis needs at least one parent skill id")
        parent_ids = [parent.skill_id for parent in parents]

        card = SkillCard(
            name=name,
            description=description,
            version=version,
            source=SkillSource.SYNTHESIZED,
            status=SkillStatus.DISCOVERED,
            trust=SkillTrust.UNTRUSTED,
            discovered_by=principal.name,
            discovery_rationale=(
                f"synthesized from {', '.join(parent_ids)} under the rule: {rule[:300]}"
            ),
            parent_skill_ids=parent_ids,
            capabilities=self._merged_capabilities(parents),
            tags=dedupe_preserve_order(
                [tag for parent in parents for tag in parent.tags] + ["synthesized"]
            ),
            limitations=dedupe_preserve_order(
                [
                    "synthesized from other skills: no sandbox verdict and no benchmark run exists "
                    "for this combination",
                    "inherits its parents' declared capabilities, which are declarations rather than "
                    "measurements",
                    *[f"parent: {parent.skill_id} ({parent.name})" for parent in parents],
                ]
            ),
            benchmark={
                "synthesis": {
                    "rule": rule,
                    "parent_skill_ids": parent_ids,
                }
            },
        )
        registered = self.registry.register(
            principal,
            card,
            change_summary=f"synthesized from {', '.join(parent_ids)}",
        )
        self.registry.record_synthesis(
            principal, registered.skill_id, rule=rule, synthesis_inputs=parent_ids
        )
        return self.lifecycle.transition(
            principal,
            registered.skill_id,
            SkillStatus.CANDIDATE,
            reason=f"synthesized from {', '.join(parent_ids)}; not trusted, awaiting sandbox",
        )

    # ================================================================== outcomes

    def record_outcome(
        self,
        principal: Principal,
        gap_id: str,
        *,
        addressed_by_skill_id: str | None,
        note: str = "",
    ) -> SkillGap:
        """Close a gap: by a registered skill (``ADDRESSED``) or by a recorded decision.

        ``addressed_by_skill_id`` must name a *registered* skill — otherwise the registry audit's
        ``GAP_ADDRESSED_BY_MISSING_SKILL`` finding would be true the moment this returns. Recording
        that no skill will address the gap requires a note: an unexplained WONT_FIX is
        indistinguishable from an abandoned one.
        """
        principal.require(Cap.SKILL_DISCOVER, "skill.gap.outcome")
        gap = self.kernel.skill_gaps.get(gap_id)
        if gap is None:
            raise SkillError(f"skill gap {gap_id!r} is not recorded", resource=gap_id)
        if addressed_by_skill_id is not None:
            if self.registry.get(addressed_by_skill_id) is None:
                raise SkillError(
                    f"cannot mark gap {gap_id} addressed by {addressed_by_skill_id!r}: it is not "
                    "registered, so the claim would be untraceable",
                    resource=gap_id,
                )
            status = "ADDRESSED"
        else:
            if not note.strip():
                raise SkillError(
                    f"recording gap {gap_id} as unaddressable by a skill requires a note saying why",
                    resource=gap_id,
                )
            status = "WONT_FIX"

        updated = gap.with_updates(
            status=status,
            addressed_by_skill_id=addressed_by_skill_id,
            last_seen=utcnow(),
        )
        self.kernel.skill_gaps.save(updated)
        self.kernel.events.append(
            "skill.gap.outcome",
            actor=principal.name,
            payload={
                "gap_id": gap_id,
                "status": status,
                "missing_capability": gap.missing_capability,
                "addressed_by_skill_id": addressed_by_skill_id,
                "note": note[:300],
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill gap {status}: {gap.missing_capability}",
            detail=note or f"gap {gap_id} addressed by {addressed_by_skill_id}",
            actor=principal.name,
            refs=[ref for ref in (gap_id, addressed_by_skill_id) if ref],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    # ================================================================== internals

    def _observations(self, *, task_id: str | None) -> list[_Observation]:
        observations: list[_Observation] = []
        observations.extend(self._audit_observations(task_id=task_id))
        observations.extend(self._task_observations(task_id=task_id))
        if task_id is None:
            # Experiments and conflicts carry no task linkage, so a task-scoped scan cannot
            # attribute them and does not pretend to.
            observations.extend(self._experiment_observations())
            observations.extend(self._conflict_observations())
        return observations

    def _audit_observations(self, *, task_id: str | None) -> list[_Observation]:
        out: list[_Observation] = []
        for audit in sorted(self.kernel.audits.all(), key=lambda a: a.audit_id):
            if task_id is not None and audit.task_id != task_id:
                continue
            for finding in audit.findings:
                code = finding.code.strip().upper()
                if not code:
                    continue
                out.append(
                    _Observation(
                        key=code,
                        capability=_capability_from_code(code),
                        code=code,
                        origin="audit",
                        evidence_id=audit.audit_id,
                        at=audit.created_at,
                        severity=finding.severity,
                    )
                )
        return out

    def _task_observations(self, *, task_id: str | None) -> list[_Observation]:
        out: list[_Observation] = []
        for task in sorted(self.kernel.tasks_store.all(), key=lambda t: t.task_id):
            if task_id is not None and task.task_id != task_id:
                continue
            for finding in task.findings:
                if finding.kind is not FindingKind.TOOLING:
                    continue  # a research observation is not a missing capability
                capability = _capability_for_text(finding.statement)
                if capability is None:
                    continue  # unrecognised statement: no capability vocabulary match, no gap
                out.append(
                    _Observation(
                        key=f"TASK_TOOLING:{capability}",
                        capability=capability,
                        code="TASK_TOOLING",
                        origin="task",
                        evidence_id=task.task_id,
                        at=finding.created_at,
                        severity=Severity.MEDIUM,
                    )
                )
        return out

    def _experiment_observations(self) -> list[_Observation]:
        out: list[_Observation] = []
        for experiment in sorted(self.kernel.experiments.all(), key=lambda e: e.experiment_id):
            mapped = _FAILED_EXPERIMENT_CAPABILITY.get(experiment.status)
            if mapped is None:
                continue
            capability, severity = mapped
            out.append(
                _Observation(
                    key=f"EXPERIMENT_{experiment.status.value}",
                    capability=capability,
                    code=f"EXPERIMENT_{experiment.status.value}",
                    origin="experiment",
                    evidence_id=experiment.experiment_id,
                    at=experiment.created_at,
                    severity=severity,
                )
            )
        return out

    def _conflict_observations(self) -> list[_Observation]:
        out: list[_Observation] = []
        for conflict in sorted(self.kernel.ledger.blocking(), key=lambda c: c.conflict_id):
            capability = _CONFLICT_CAPABILITIES.get(conflict.kind, "research engineering")
            out.append(
                _Observation(
                    key=f"CONFLICT_UNRESOLVED_{conflict.kind.value}",
                    capability=capability,
                    code=f"CONFLICT_UNRESOLVED_{conflict.kind.value}",
                    origin="conflict",
                    evidence_id=conflict.conflict_id,
                    at=conflict.created_at,
                    severity=Severity.HIGH,
                )
            )
        return out

    def _file_gap(self, principal: Principal, group: _Group, *, task_id: str | None) -> SkillGap:
        occurrences = len(group.observations)
        severity = _max_severity([o.severity for o in group.observations])
        if occurrences >= 3 and _severity_rank(severity) < _severity_rank(Severity.HIGH):
            severity = Severity.HIGH
        first_seen = min(o.at for o in group.observations)
        last_seen = max(o.at for o in group.observations)

        existing = self._open_gap(group.key)
        description = (
            f"{group.key}: {group.capability} — {occurrences} occurrence(s) in "
            f"{', '.join(group.origins)}; the project needed this capability and no usable skill "
            f"provides it (signal: {', '.join(group.codes)})"
        )
        payload: dict[str, Any] = {
            "description": description,
            "missing_capability": group.capability,
            "occurrences": occurrences,
            "severity": severity,
            "last_seen": last_seen,
            "search_queries": self.capability_queries(group.capability),
        }
        if existing is None:
            gap = SkillGap(
                evidence=group.evidence,
                first_seen=first_seen,
                status="OPEN",
                **payload,
            )
        else:
            gap = existing.with_updates(
                evidence=dedupe_preserve_order([*existing.evidence, *group.evidence]),
                first_seen=min(existing.first_seen, first_seen),
                **payload,
            )
        self.kernel.skill_gaps.save(gap)
        self.kernel.events.append(
            "skill.gap.detected",
            actor=principal.name,
            task_id=task_id,
            payload={
                "gap_id": gap.gap_id,
                "key": group.key,
                "missing_capability": gap.missing_capability,
                "occurrences": gap.occurrences,
                "severity": gap.severity.value,
                "origins": group.origins,
                "evidence": gap.evidence,
                "refiled": existing is not None,
            },
        )
        return gap

    def _open_gap(self, key: str) -> SkillGap | None:
        """The live gap already filed for ``key``, recognised by its machine-readable prefix.

        Re-running detection must not file a second copy of the same gap: the key is the first token
        of the description, which is why the description is built as ``"<KEY>: ..."``.
        """
        prefix = f"{key}: "
        for gap in sorted(self.kernel.skill_gaps.all(), key=lambda g: g.gap_id):
            if gap.status in _LIVE_GAP_STATUSES and gap.description.startswith(prefix):
                return gap
        return None

    def _mark_searching(
        self, principal: Principal, gap: SkillGap, *, queries: Sequence[str], found: int
    ) -> SkillGap:
        stored = self.kernel.skill_gaps.get(gap.gap_id) or gap
        updated = stored.with_updates(
            status="SEARCHING",
            search_queries=dedupe_preserve_order([*stored.search_queries, *queries]),
            last_seen=utcnow(),
        )
        self.kernel.skill_gaps.save(updated)
        self.kernel.events.append(
            "skill.discovery.search",
            actor=principal.name,
            payload={
                "gap_id": gap.gap_id,
                "queries": list(queries),
                "candidates_registered": found,
            },
        )
        return updated

    def _card_from_result(
        self,
        principal: Principal,
        result: Mapping[str, object],
        *,
        query: str,
        gap: SkillGap,
    ) -> SkillCard:
        name = _first_str(result, _NAME_KEYS)
        if not name:
            raise SkillError(
                f"provider result for query {query!r} names no repository or skill; a card without "
                "a name cannot be reviewed",
                resource=gap.gap_id,
            )
        repository = _first_str(result, _REPO_KEYS)
        url = repository or _first_str(result, ("url", "html_url"))
        source = (
            SkillSource.GITHUB
            if url and "github.com" in url.lower()
            else SkillSource.WEB
        )
        signals: dict[str, object] = {}
        for label, keys in (
            ("stars", _STARS_KEYS),
            ("forks", ("forks", "forks_count")),
            ("pushed_at", ("pushed_at", "updated_at")),
        ):
            value = next((result[k] for k in keys if k in result), None)
            if value is not None:
                signals[label] = value
        return SkillCard(
            name=name,
            description=_first_str(result, _DESCRIPTION_KEYS) or f"candidate for {query}",
            version=str(result.get("version") or "0.0.0"),
            source=source,
            source_url=url,
            repository=repository,
            commit=_first_str(result, _COMMIT_KEYS),
            license=_first_str(result, _LICENSE_KEYS),
            entrypoint=_first_str(result, _ENTRYPOINT_KEYS),
            documentation_url=_first_str(result, _DOCS_KEYS),
            status=SkillStatus.DISCOVERED,
            trust=SkillTrust.UNTRUSTED,
            discovered_by=principal.name,
            discovery_query=query,
            discovery_rationale=(
                f"keyword hit for capability gap {gap.gap_id} ({gap.missing_capability}); "
                "untrusted until it passes the sandbox and the benchmark"
            ),
            tags=[gap.missing_capability, "discovered"],
            limitations=[
                "discovered by keyword search: not sandboxed, not benchmarked, not usable",
                "popularity is recorded for review and is never a capability metric",
            ],
            benchmark={"source_signals": signals} if signals else {},
        )

    def _already_registered(self, card: SkillCard) -> bool:
        for existing in self.kernel.skills.all():
            if existing.repository and card.repository:
                if existing.repository.lower() == card.repository.lower() and (
                    existing.commit or ""
                ) == (card.commit or ""):
                    return True
            if existing.name.lower() == card.name.lower() and (
                existing.repository or ""
            ).lower() == (card.repository or "").lower():
                return True
        return False

    @staticmethod
    def _merged_capabilities(parents: Sequence[SkillCard]) -> list[SkillCapability]:
        merged: dict[str, SkillCapability] = {}
        for parent in parents:
            for capability in parent.capabilities:
                merged.setdefault(capability.name, capability)
        return list(merged.values())


__all__ = [
    "CAPABILITY_AREAS",
    "CAPABILITY_QUERY_TEMPLATES",
    "FINDING_CODE_CAPABILITIES",
    "SkillDiscoveryAgent",
]

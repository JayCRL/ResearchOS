"""The skill lifecycle: the gates between "we found this" and "the project uses this".

Why this module exists
----------------------
Status is where a skill meta-system usually lies to itself. A manifest says ``ACTIVE``; someone flips
a flag; the skill is now in the research path. This module makes each rung *earned*:

* ``DISCOVERED -> CANDIDATE -> SANDBOX -> EVALUATED -> VERIFIED -> ACTIVE`` — one step at a time.
  Skipping rungs is refused, including ``DISCOVERED -> ACTIVE``.
* **No ACTIVE without a passing sandbox.** A skill that was never inspected statically cannot reach
  ACTIVE even if its benchmark score is perfect, so trust can never be claimed without evidence.
* **No ACTIVE without a ``BenchmarkRun`` whose ``activatable()`` is true**, and **never while
  ``regression_passed is False``**. Upgrade safety is the same rule: re-activating an already-ACTIVE
  skill on a new commit requires a run that was compared with the incumbent
  (``regression_passed is True``), which is invariant 8 — "a skill upgrade cannot break existing
  benchmarks" — enforced where the upgrade happens rather than promised in a docstring.
* **Activation and deprecation are human decisions** (``skill.activate`` / ``skill.deprecate`` held
  only by the human principal). The layer that measures is not the layer that promotes.
* Every transition writes an event *and* a timeline entry: "why is this skill in the research path?"
  must be answerable from the project's own history, not from a chat log.

What is deliberately *not* automatic: no transition sets ``SkillTrust.TRUSTED``. Trust beyond
"benchmarked" is a human judgement, and nothing in this layer should manufacture it.
"""

from __future__ import annotations

from typing import Mapping, cast

from ..kernel.errors import SkillError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import HUMAN_ONLY_CAPABILITIES, Cap, Principal
from ..models import BenchmarkRun, SandboxReport, SkillCard, SkillTrust, SkillVersion
from ..models.common import SkillStatus
from ..models.timeline import TimelineEventKind
from .registry import SkillRegistry

#: The legal edges. Forward edges are the documented ladder; the extra edges are the honest loops
#: (re-sandboxing after a change, recovering a DEGRADED skill, replacing the active version of an
#: ACTIVE skill behind the regression gate). ``DEPRECATED`` is terminal.
LEGAL_TRANSITIONS: Mapping[SkillStatus, frozenset[SkillStatus]] = {
    SkillStatus.DISCOVERED: frozenset({SkillStatus.CANDIDATE, SkillStatus.DEPRECATED}),
    SkillStatus.CANDIDATE: frozenset({SkillStatus.SANDBOX, SkillStatus.DEPRECATED}),
    SkillStatus.SANDBOX: frozenset(
        {SkillStatus.EVALUATED, SkillStatus.CANDIDATE, SkillStatus.DEPRECATED}
    ),
    SkillStatus.EVALUATED: frozenset(
        {SkillStatus.VERIFIED, SkillStatus.SANDBOX, SkillStatus.DEPRECATED}
    ),
    SkillStatus.VERIFIED: frozenset(
        {SkillStatus.ACTIVE, SkillStatus.EVALUATED, SkillStatus.DEPRECATED}
    ),
    # ACTIVE -> ACTIVE is only legal as a regression-gated version upgrade (see `transition`).
    SkillStatus.ACTIVE: frozenset(
        {SkillStatus.ACTIVE, SkillStatus.DEGRADED, SkillStatus.DEPRECATED}
    ),
    SkillStatus.DEGRADED: frozenset({SkillStatus.ACTIVE, SkillStatus.DEPRECATED}),
    SkillStatus.DEPRECATED: frozenset(),
}

#: Capability required for each target status. "Who may say this is true?"
_CAP_FOR_STATUS: Mapping[SkillStatus, Cap] = {
    SkillStatus.DISCOVERED: Cap.SKILL_PROPOSE,
    SkillStatus.CANDIDATE: Cap.SKILL_PROPOSE,
    SkillStatus.SANDBOX: Cap.SKILL_EVALUATE,
    SkillStatus.EVALUATED: Cap.SKILL_EVALUATE,
    SkillStatus.VERIFIED: Cap.SKILL_EVALUATE,
    SkillStatus.ACTIVE: Cap.SKILL_ACTIVATE,
    SkillStatus.DEGRADED: Cap.SKILL_EVALUATE,
    SkillStatus.DEPRECATED: Cap.SKILL_DEPRECATE,
}


def _legal_targets(status: SkillStatus) -> str:
    targets = sorted(t.value for t in LEGAL_TRANSITIONS[status])
    return ", ".join(targets) if targets else "none (terminal state)"


class SkillLifecycle:
    """Owns every status change of a skill. The registry never moves status by itself."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel
        self.registry = SkillRegistry(kernel)

    # ================================================================== transitions

    def transition(
        self,
        principal: Principal,
        skill_id: str,
        new_status: SkillStatus,
        *,
        benchmark_run_id: str | None = None,
        sandbox_report_id: str | None = None,
        reason: str = "",
        commit: str | None = None,
    ) -> SkillCard:
        """Move a skill to ``new_status`` if every gate for that status holds.

        Order of checks is deliberate: capability first (authority), then legality (can this edge be
        walked at all), then evidence (does the destination's gate hold). Each failure raises
        ``SkillError`` naming the invariant it protects.
        """
        target = SkillStatus(new_status) if isinstance(new_status, str) else new_status
        required_cap = _CAP_FOR_STATUS[target]
        principal.require(required_cap, f"skill.transition:{skill_id}->{target.value}")
        if required_cap in HUMAN_ONLY_CAPABILITIES and not principal.is_human:
            raise SkillError(
                f"only a human may move a skill to {target.value}; {principal.name!r} is a "
                f"{principal.kind.value} (activation and deprecation are human decisions)",
                principal=principal.name,
                resource=skill_id,
            )

        card = self.registry.require(skill_id)
        current = card.status
        if target not in LEGAL_TRANSITIONS[current]:
            raise SkillError(
                f"illegal skill transition {current.value} -> {target.value} for {skill_id}; "
                f"legal targets from {current.value} are: {_legal_targets(current)}. "
                "The ladder is DISCOVERED -> CANDIDATE -> SANDBOX -> EVALUATED -> VERIFIED -> ACTIVE; "
                "rungs are earned one at a time.",
                principal=principal.name,
                resource=skill_id,
            )

        updates: dict[str, object] = {"status": target}
        run: BenchmarkRun | None = None
        if benchmark_run_id:
            run = self.kernel.benchmark_runs.get(benchmark_run_id)
            if run is None:
                raise SkillError(
                    f"benchmark run {benchmark_run_id!r} does not exist", resource=skill_id
                )
            if run.skill_id != skill_id:
                raise SkillError(
                    f"benchmark run {benchmark_run_id} was produced for skill {run.skill_id}, "
                    f"not {skill_id}",
                    resource=skill_id,
                )

        if target is SkillStatus.VERIFIED:
            report = self._sandbox_report(card, sandbox_report_id)
            if report is None:
                raise SkillError(
                    f"cannot mark {skill_id} VERIFIED: the sandbox has not passed. Run "
                    "SkillSandbox.inspect (static analysis) first — verification without a sandbox "
                    "verdict would be a claim with no evidence",
                    principal=principal.name,
                    resource=skill_id,
                )
            if card.trust is SkillTrust.UNTRUSTED:
                updates["trust"] = SkillTrust.SANDBOXED

        if target is SkillStatus.ACTIVE:
            report = self._sandbox_report(card, sandbox_report_id)
            if report is None:
                raise SkillError(
                    f"cannot activate {skill_id}: no passing sandbox report exists for it "
                    "(invariant 7: an external skill is inspected before it is used)",
                    principal=principal.name,
                    resource=skill_id,
                )
            if run is None:
                run = self.registry._latest_run(skill_id)  # noqa: SLF001 - same plane, one source
            if run is None:
                raise SkillError(
                    f"cannot activate {skill_id}: no benchmark run exists "
                    "(invariant 8: activation is earned by a benchmark, never declared)",
                    principal=principal.name,
                    resource=skill_id,
                )
            if run.regression_passed is False:
                raise SkillError(
                    f"cannot activate {skill_id}: the regression gate failed against the incumbent "
                    f"({', '.join(run.regression_regressions) or 'aggregate drop'}) — an upgrade may "
                    "not break a benchmark the previous version passed",
                    principal=principal.name,
                    resource=skill_id,
                )
            ok, reasons = run.activatable()
            if not ok:
                raise SkillError(
                    f"cannot activate {skill_id}: " + "; ".join(reasons),
                    principal=principal.name,
                    resource=skill_id,
                )
            if (
                current is SkillStatus.ACTIVE
                and commit is not None
                and card.commit is not None
                and commit != card.commit
                and run.regression_passed is not True
            ):
                raise SkillError(
                    f"cannot move the active version of {skill_id} to commit {commit[:12]}: the "
                    "benchmark run was never compared with the incumbent (regression_passed is not "
                    "True). Re-run the suite with `incumbent=<current run>` before switching "
                    "versions (invariant 8).",
                    principal=principal.name,
                    resource=skill_id,
                )
            updates["last_benchmark_run_id"] = run.benchmark_run_id
            updates["trust"] = (
                SkillTrust.BENCHMARKED if card.trust is not SkillTrust.TRUSTED else card.trust
            )
            updates["quality_metrics"] = self._metrics_after_activation(card, run)
            updates["benchmark"] = {
                **card.benchmark,
                f"run:{run.suite.value}": run.score,
                "last_run_id": run.benchmark_run_id,
            }
            if commit:
                updates["commit"] = commit

        if target is SkillStatus.DEPRECATED and not reason.strip():
            raise SkillError(
                f"deprecating {skill_id} requires a reason: a skill removed from the research path "
                "must explain what replaces it",
                principal=principal.name,
                resource=skill_id,
            )
        if target is SkillStatus.DEGRADED and not reason.strip():
            raise SkillError(
                f"degrading {skill_id} requires a reason (what broke, and what it affects)",
                principal=principal.name,
                resource=skill_id,
            )

        updated = cast(SkillCard, card.with_updates(**updates))
        self.kernel.skills.save(updated)

        self.kernel.events.append(
            f"skill.status.{target.value.lower()}",
            actor=principal.name,
            payload={
                "skill_id": skill_id,
                "from": current.value,
                "to": target.value,
                "reason": reason[:300],
                "benchmark_run_id": run.benchmark_run_id if run else benchmark_run_id,
                "sandbox_report_id": sandbox_report_id,
                "commit": updated.commit,
                "trust": updated.trust.value,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill {current.value} -> {target.value}: {card.name}",
            detail=reason or f"{skill_id} moved to {target.value}",
            actor=principal.name,
            refs=[skill_id, *([run.benchmark_run_id] if run else [])],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    # ================================================================== conveniences

    def activate(
        self, principal: Principal, skill_id: str, *, benchmark_run_id: str
    ) -> SkillCard:
        """Activate a skill on the strength of one named benchmark run."""
        return self.transition(
            principal,
            skill_id,
            SkillStatus.ACTIVE,
            benchmark_run_id=benchmark_run_id,
            reason=f"activated on benchmark run {benchmark_run_id}",
        )

    def degrade(self, principal: Principal, skill_id: str, *, reason: str) -> SkillCard:
        """Take a skill out of the research path *without* discarding it.

        Degradation is the honest middle ground between "working" and "deleted": the skill stays,
        its history stays, and a passing re-benchmark can bring it back.
        """
        return self.transition(principal, skill_id, SkillStatus.DEGRADED, reason=reason)

    def deprecate(self, principal: Principal, skill_id: str, *, reason: str) -> SkillCard:
        """Terminally retire a skill. Terminal on purpose: un-retiring would re-route history."""
        return self.transition(principal, skill_id, SkillStatus.DEPRECATED, reason=reason)

    def supersede(
        self, principal: Principal, old_skill_id: str, new_skill_id: str
    ) -> tuple[SkillCard, SkillCard]:
        """Deprecate ``old`` in favour of ``new`` and link both directions.

        Supersession is provenance, not deletion: the old card keeps its history, gains
        ``superseded_by``, and the replacement gains ``supersedes`` so a reader can follow the swap in
        either direction.
        """
        principal.require(Cap.SKILL_DEPRECATE, f"skill.supersede:{old_skill_id}")
        if old_skill_id == new_skill_id:
            raise SkillError(
                "a skill cannot supersede itself", principal=principal.name, resource=old_skill_id
            )
        old = self.registry.require(old_skill_id)
        new = self.registry.require(new_skill_id)

        retired = self.transition(
            principal,
            old_skill_id,
            SkillStatus.DEPRECATED,
            reason=f"superseded by {new_skill_id} ({new.name})",
        )
        retired = cast(SkillCard, retired.with_updates(superseded_by=new_skill_id))
        self.kernel.skills.save(retired)

        replacement = cast(SkillCard, new.with_updates(supersedes=old_skill_id))
        self.kernel.skills.save(replacement)

        self.kernel.events.append(
            "skill.superseded",
            actor=principal.name,
            payload={"old_skill_id": old_skill_id, "new_skill_id": new_skill_id},
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill superseded: {old.name} -> {new.name}",
            detail=f"{old_skill_id} retired in favour of {new_skill_id}",
            actor=principal.name,
            refs=[old_skill_id, new_skill_id],
            state_revision=self.kernel.state.revision(),
        )
        return retired, replacement

    def rollback(
        self,
        principal: Principal,
        skill_id: str,
        *,
        to_version_id: str,
        benchmark_run_id: str,
    ) -> SkillCard:
        """Return a skill to an earlier *pinned* version, with the benchmark run that justifies it.

        This is the recovery path for a bad upgrade: the pinned version and its passing run already
        exist, so rollback re-points the active revision rather than re-evaluating anything. A
        version with no pin (commit/content hash) cannot be rolled back to — "we rolled back to
        something" must name something.
        """
        principal.require(Cap.SKILL_ACTIVATE, f"skill.rollback:{skill_id}")
        if not principal.is_human:
            raise SkillError(
                f"only a human may roll back an active skill; {principal.name!r} is a "
                f"{principal.kind.value}",
                principal=principal.name,
                resource=skill_id,
            )
        card = self.registry.require(skill_id)
        if card.status not in (SkillStatus.ACTIVE, SkillStatus.DEGRADED, SkillStatus.VERIFIED):
            raise SkillError(
                f"cannot roll back {skill_id}: it is {card.status.value}; rollback only makes sense "
                "for a skill that was already in the research path",
                resource=skill_id,
            )
        version: SkillVersion | None = self.kernel.skill_versions.get(to_version_id)
        if version is None:
            raise SkillError(f"skill version {to_version_id!r} does not exist", resource=skill_id)
        if version.skill_id != skill_id:
            raise SkillError(
                f"skill version {to_version_id} belongs to {version.skill_id}, not {skill_id}",
                resource=skill_id,
            )
        if not version.commit or not version.content_hash:
            raise SkillError(
                f"cannot roll back to unpinned version {to_version_id}; pin it first so the "
                "rollback target is a named revision",
                resource=skill_id,
            )
        run: BenchmarkRun | None = self.kernel.benchmark_runs.get(benchmark_run_id)
        if run is None:
            raise SkillError(f"benchmark run {benchmark_run_id!r} does not exist", resource=skill_id)
        if run.skill_id != skill_id:
            raise SkillError(
                f"benchmark run {benchmark_run_id} belongs to {run.skill_id}, not {skill_id}",
                resource=skill_id,
            )
        if run.skill_version_id not in (None, to_version_id):
            raise SkillError(
                f"benchmark run {benchmark_run_id} was produced by version "
                f"{run.skill_version_id}, not {to_version_id}",
                resource=skill_id,
            )
        ok, reasons = run.activatable()
        if not ok:
            raise SkillError(
                f"cannot roll back {skill_id} on run {benchmark_run_id}: " + "; ".join(reasons),
                resource=skill_id,
            )

        updated = cast(
            SkillCard,
            card.with_updates(
                status=SkillStatus.ACTIVE,
                version=version.version,
                commit=version.commit,
                content_hash=version.content_hash,
                last_benchmark_run_id=run.benchmark_run_id,
                trust=(
                    SkillTrust.BENCHMARKED if card.trust is not SkillTrust.TRUSTED else card.trust
                ),
                benchmark={
                    **card.benchmark,
                    f"run:{run.suite.value}": run.score,
                    "last_run_id": run.benchmark_run_id,
                    "rolled_back_to_version": version.skill_version_id,
                },
            ),
        )
        self.kernel.skills.save(updated)
        if run.benchmark_run_id not in version.benchmark_run_ids:
            linked = cast(
                SkillVersion,
                version.with_updates(
                    benchmark_run_ids=[*version.benchmark_run_ids, run.benchmark_run_id]
                ),
            )
            self.kernel.skill_versions.save(linked)

        self.kernel.events.append(
            "skill.rolled_back",
            actor=principal.name,
            payload={
                "skill_id": skill_id,
                "to_version_id": to_version_id,
                "commit": version.commit,
                "benchmark_run_id": run.benchmark_run_id,
                "from_commit": card.commit,
            },
        )
        self.kernel.timeline.record(
            TimelineEventKind.SKILL,
            f"Skill rolled back: {card.name}",
            detail=(
                f"{skill_id} back to {version.version} @ {version.commit[:12]} "
                f"(was {card.commit[:12] if card.commit else 'unpinned'})"
            ),
            actor=principal.name,
            refs=[skill_id, to_version_id, run.benchmark_run_id],
            state_revision=self.kernel.state.revision(),
        )
        return updated

    # ================================================================== internals

    def _sandbox_report(
        self, card: SkillCard, sandbox_report_id: str | None
    ) -> SandboxReport | None:
        """Resolve a *passing* sandbox report for this skill, or ``None``."""
        if sandbox_report_id:
            report = self.kernel.sandbox_reports.get(sandbox_report_id)
            if report is None:
                raise SkillError(
                    f"sandbox report {sandbox_report_id!r} does not exist", resource=card.skill_id
                )
            if report.skill_id != card.skill_id:
                raise SkillError(
                    f"sandbox report {sandbox_report_id} was produced for skill "
                    f"{report.skill_id}, not {card.skill_id}",
                    resource=card.skill_id,
                )
            return report if report.passed else None
        reports = [
            r
            for r in self.kernel.sandbox_reports.all()
            if r.skill_id == card.skill_id and r.passed
        ]
        if not reports:
            return None
        return max(reports, key=lambda r: (r.created_at, r.sandbox_report_id))

    @staticmethod
    def _metrics_after_activation(card: SkillCard, run: BenchmarkRun):
        """Record what the benchmark measured without inventing what it did not.

        ``regression_safety`` is set only when the gate actually ran (an unmeasured skill is not
        credited *or* punished). ``accuracy`` is filled from the measured suite score only when the
        card declared no accuracy of its own: measurement answers silence, and never overwrites a
        declared value.
        """
        metrics = card.quality_metrics
        if metrics.accuracy == 0.0:
            metrics = metrics.with_updates(accuracy=run.score)
        if run.regression_passed is not None:
            metrics = metrics.with_updates(regression_safety=1.0 if run.regression_passed else 0.0)
        return metrics


__all__ = ["LEGAL_TRANSITIONS", "SkillLifecycle"]

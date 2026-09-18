"""The ResearchOS CLI — the first-class interface (UI comes later, deliberately).

Every command is a thin, explicit wrapper over a kernel or module operation. Two rules the CLI
follows without exception:

* it never approves anything on the user's behalf unless the *user* is the principal and the action
  is an explicit command (`researchos claim approve`, `researchos task transition approve`);
* it prints what was refused as loudly as what succeeded. A blocked overclaim is a feature.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..kernel.errors import ResearchOSError
from ..kernel.kernel import ResearchKernel
from ..kernel.paths import ProjectPaths

app = typer.Typer(
    name="researchos",
    help="ResearchOS — research state, evidence and claims that humans and machines can audit.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

task_app = typer.Typer(help="Bounded research tasks (the anti-drift container).", no_args_is_help=True)
experiment_app = typer.Typer(help="Experiment registry and provenance.", no_args_is_help=True)
evidence_app = typer.Typer(help="Evidence registry and verification.", no_args_is_help=True)
claim_app = typer.Typer(help="Claim registry, lifecycle and language calibration.", no_args_is_help=True)
literature_app = typer.Typer(help="Literature OS: search, map, prior art.", no_args_is_help=True)
novelty_app = typer.Typer(help="Novelty audits (coverage-gated).", no_args_is_help=True)
skill_app = typer.Typer(help="Skill Intelligence Layer.", no_args_is_help=True)
paper_app = typer.Typer(help="Paper compiler and auditors.", no_args_is_help=True)
audit_app = typer.Typer(help="Audits, and the append-only event log.", no_args_is_help=True)
agent_app = typer.Typer(help="Agent inventory and capability boundaries.", no_args_is_help=True)
review_app = typer.Typer(help="Human review queue produced by Research Import.", no_args_is_help=True)

app.add_typer(task_app, name="task")
app.add_typer(experiment_app, name="experiment")
app.add_typer(evidence_app, name="evidence")
app.add_typer(claim_app, name="claim")
app.add_typer(literature_app, name="literature")
app.add_typer(novelty_app, name="novelty")
app.add_typer(skill_app, name="skill")
app.add_typer(paper_app, name="paper")
app.add_typer(audit_app, name="audit")
app.add_typer(agent_app, name="agent")
app.add_typer(review_app, name="review")


# ======================================================================================
# helpers
# ======================================================================================


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _kernel(root: Optional[Path] = None) -> ResearchKernel:
    try:
        return ResearchKernel.open(root)
    except ResearchOSError as exc:
        _fail(str(exc))


def _ok(message: str) -> None:
    console.print(f"[green]ok[/green] {message}")


def _warn(message: str) -> None:
    console.print(f"[yellow]warn[/yellow] {message}")


def _json(data: object) -> None:
    console.print_json(json.dumps(data, default=str))


def short_id(identifier: str | None) -> str:
    """Display form of an id: the *tail*, which is the random, unique part.

    The head of a ULID is a millisecond timestamp, so two objects created in the same millisecond
    share it. Showing the head makes different records look identical in a table, which is the one
    thing an audit view must never do.
    """
    if not identifier:
        return ""
    return identifier.split("_", 1)[1][-8:] if "_" in identifier else identifier[-8:]


def _run(fn, *args, **kwargs):
    """Execute a kernel operation, turning ResearchOS errors into clean CLI failures."""
    try:
        return fn(*args, **kwargs)
    except ResearchOSError as exc:
        _fail(str(exc))


# ======================================================================================
# init / import / status
# ======================================================================================


@app.command("init")
def init(
    path: Path = typer.Argument(Path("."), help="Directory to initialise."),
    name: Optional[str] = typer.Option(None, "--name", help="Project name."),
    domain: str = typer.Option("", "--domain", help="Research domain, e.g. 'AI/ML'."),
    description: str = typer.Option("", "--description"),
) -> None:
    """Create ``.researchos/`` and the initial research state."""
    kernel = _run(ResearchKernel.create, path, name=name, domain=domain, description=description)
    _ok(f"initialised {kernel.ros_dir} (project {kernel.project().name})")
    console.print(
        "next: [bold]researchos import <dir>[/bold] to take over existing research, or "
        "[bold]researchos task create[/bold]"
    )


@app.command("import")
def import_command(
    source: Path = typer.Argument(..., help="Directory or repository to reconstruct research state from."),
    into: Optional[Path] = typer.Option(None, "--into", help="Project root (default: this directory)."),
    task: Optional[str] = typer.Option(None, "--task", help="Task id to attach the import to."),
    no_approve: bool = typer.Option(
        False, "--no-approve", help="Leave the recovered core question as a pending state transition."
    ),
    max_claims: int = typer.Option(60, "--max-claims"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Reconstruct research state from an existing project (Research Archaeology)."""
    root = into or Path(".")
    paths = ProjectPaths.discover_or_none(root)
    kernel = (
        _run(ResearchKernel.open, root)
        if paths
        else _run(ResearchKernel.create, root, name=Path(source).resolve().name)
    )

    from ..importer import ImportPipeline

    pipeline = ImportPipeline(kernel, approve_core_question=not no_approve, max_claims=max_claims)
    result = _run(pipeline.run, source, task_id=task)
    if json_out:
        _json(result.report.model_dump(mode="json"))
    else:
        console.print(result.report.render())
    _ok(result.summary_line())


@app.command("status")
def status(
    root: Optional[Path] = typer.Option(None, "--root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show the global research state (the dashboard's text version)."""
    kernel = _kernel(root)
    state = kernel.research_state()
    if json_out:
        _json(kernel.describe())
        return

    question = state.core_question.statement if state.core_question else "(not set)"
    placeholder = " [unconfirmed placeholder]" if state.core_question and state.core_question.is_placeholder else ""
    console.print(Panel.fit(f"[bold]{question}[/bold]{placeholder}", title="CORE QUESTION"))

    claims = kernel.claims.all()
    if claims:
        table = Table(title="CURRENT CLAIMS", show_lines=False)
        table.add_column("id", style="dim", no_wrap=True)
        table.add_column("status")
        table.add_column("level")
        table.add_column("statement", overflow="fold")
        from ..claims.language import permitted_verb

        for claim in claims:
            table.add_row(
                claim.claim_id.split("_")[1][-8:],
                claim.status.value,
                claim.evidence_level.value,
                claim.statement[:90],
            )
        console.print(table)
        console.print(
            f"[dim]permitted language per claim comes from the evidence level: "
            f"{permitted_verb(claims[0].evidence_level)!r} for the first claim[/dim]"
        )
    else:
        console.print("[dim]no claims registered[/dim]")

    frontier = state.current_frontier
    console.print(Panel.fit(frontier.statement or "(no frontier recorded)", title="CURRENT FRONTIER"))

    counts = kernel.counts()
    interesting = {k: v for k, v in counts.items() if v}
    console.print("[bold]REGISTRY[/bold] " + ", ".join(f"{k}={v}" for k, v in sorted(interesting.items())))

    integrity = kernel.integrity()
    if integrity["ok"]:
        _ok(f"integrity ok · state rev {integrity['revision']} · event head {integrity['head']}")
    else:
        _warn(f"integrity issues ({len(integrity['issues'])})")
        for issue in integrity["issues"][:10]:
            console.print(f"  - {issue}")

    pending = kernel.pending_reviews()
    if pending:
        console.print(f"[yellow]{len(pending)} review item(s) awaiting a human decision[/yellow]")


@app.command("dashboard")
def dashboard(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """Everything the first UI version must show: state, evidence, literature, skills, readiness."""
    kernel = _kernel(root)
    state = kernel.research_state()
    console.print(Panel.fit(state.core_question.statement if state.core_question else "(no core question)", title="CORE QUESTION"))

    from ..evidence import EvidenceGraph, EvidenceRegistry

    console.print(Panel.fit(
        json.dumps(EvidenceRegistry(kernel).summary(), indent=2), title="EVIDENCE COVERAGE"
    ))
    console.print(Panel.fit(
        json.dumps(_literature_state(state), indent=2), title="LITERATURE MAP"
    ))

    experiments = Table(title="CURRENT EXPERIMENTS")
    for column in ("id", "title", "status", "level", "seeds"):
        experiments.add_column(column, overflow="fold")
    for experiment in kernel.experiments.all():
        experiments.add_row(
            experiment.experiment_id.split("_")[1][-8:],
            experiment.title[:50],
            experiment.status.value,
            experiment.evidence_level().value,
            str(len(experiment.seeds)),
        )
    console.print(experiments)

    conflicts = kernel.ledger.summary()
    console.print(Panel.fit(
        "unresolved conflicts: "
        + str(conflicts.get("UNRESOLVED", 0) + conflicts.get("A_WINS", 0) + conflicts.get("B_WINS", 0))
        + f" · blocking claim promotion: {conflicts.get('blocking', 0)}",
        title="CONFLICTS",
    ))

    try:
        from ..paper import ReadinessAssessor

        readiness = _run(ReadinessAssessor(kernel).assess, kernel.human())
        table = Table(title="PAPER READINESS (no single score, by design)")
        table.add_column("dimension")
        table.add_column("score")
        table.add_column("basis", overflow="fold")
        for dimension in readiness.dimensions:
            table.add_row(dimension.label(), f"{dimension.score:.2f}", dimension.basis[:80])
        console.print(table)
    except Exception as exc:  # noqa: BLE001 - readiness is optional for the dashboard
        _warn(f"readiness unavailable: {type(exc).__name__}: {exc}")

    try:
        from ..skills.registry import SkillRegistry

        console.print(Panel.fit(json.dumps(SkillRegistry(kernel).health(), indent=2, default=str), title="SKILL HEALTH"))
    except Exception as exc:  # noqa: BLE001
        _warn(f"skill health unavailable: {type(exc).__name__}: {exc}")

    if state.skill_state.gaps:
        console.print("[yellow]SKILL GAPS[/yellow]")
        for gap_id in state.skill_state.gaps[:10]:
            record = kernel.skill_gaps.get(gap_id)
            if record:
                console.print(f"  - ({record.occurrences}x) {record.missing_capability}")

    graph = EvidenceGraph(kernel)
    orphans = graph.orphan_evidence()
    if orphans:
        _warn(f"{len(orphans)} evidence record(s) not linked to any claim: {orphans[:5]}")


def _literature_state(state) -> dict:
    literature = state.literature_state
    return {
        "coverage_score": literature.coverage_score,
        "basis": literature.coverage_basis,
        "queries_executed": literature.queries_executed,
        "providers_used": literature.providers_used,
        "papers_screened": literature.papers_screened,
        "fulltext_verified": literature.fulltext_verified,
        "closest_prior_work": [
            {"paper_id": p.paper_id, "title": p.title, "threat": p.threat_level}
            for p in literature.closest_prior_work[:5]
        ],
    }


# ======================================================================================
# task
# ======================================================================================


@task_app.command("create")
def task_create(
    objective: str = typer.Argument(..., help="One sentence: what is this task for?"),
    purpose: str = typer.Option("ANALYSE", "--purpose", help="DEBUG|EXPLAIN|DESIGN|RUN|ANALYSE|LITERATURE|WRITE|IMPORT|AUDIT|SKILL"),
    priority: str = typer.Option("EXPLORATORY", "--priority", help="PRIMARY|SECONDARY|EXPLORATORY"),
    stop_condition: str = typer.Option("", "--stop", help="When should this task stop?"),
    parent_claim: Optional[str] = typer.Option(None, "--claim"),
    parent_experiment: Optional[str] = typer.Option(None, "--experiment"),
    allowed: Optional[str] = typer.Option(None, "--allowed", help="Comma-separated capability names."),
    forbidden: Optional[str] = typer.Option(None, "--forbidden"),
    touches_core: bool = typer.Option(False, "--touches-core"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Create a bounded task. A task's findings never redefine the project on their own."""
    kernel = _kernel(root)
    task = _run(
        kernel.task_manager.create,
        kernel.principal("planner"),
        objective=objective,
        purpose=purpose,
        priority=priority,
        stop_condition=stop_condition,
        parent_claim=parent_claim,
        parent_experiment=parent_experiment,
        allowed_actions=[a.strip() for a in (allowed or "").split(",") if a.strip()],
        forbidden_actions=[a.strip() for a in (forbidden or "").split(",") if a.strip()],
        touches_core=touches_core,
    )
    _ok(f"task {task.task_id} created ({task.priority.value}/{task.purpose.value})")
    if not task.may_request_core_change():
        console.print(
            "[dim]this task may NOT request a change to the core research question; "
            "record a finding with suggests_core_change=True and let a human decide[/dim]"
        )


@task_app.command("list")
def task_list(
    open_only: bool = typer.Option(False, "--open"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """List tasks with their boundaries."""
    kernel = _kernel(root)
    table = Table("id", "priority", "purpose", "status", "objective")
    for task in kernel.task_manager.list(open_only=open_only):
        table.add_row(
            task.task_id.split("_")[1][-8:], task.priority.value, task.purpose.value,
            task.status.value, task.objective[:60],
        )
    console.print(table)


@task_app.command("run")
def task_run(
    task_id: str = typer.Argument(...),
    finding: Optional[str] = typer.Option(None, "--finding", help="Record a finding and close the run."),
    kind: str = typer.Option("OBSERVATION", "--kind"),
    suggests_core_change: bool = typer.Option(False, "--suggests-core-change"),
    close: Optional[str] = typer.Option(None, "--close", help="Close the task with this summary."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Run a bounded task: print its scope, then record what was found.

    The scope printed here is the *only* context an agent should load for this task — bounded
    context is how research direction stops drifting with the chat history.
    """
    kernel = _kernel(root)
    scope = _run(kernel.task_manager.context_scope, task_id)
    console.print(Panel.fit(json.dumps(scope, indent=2, default=str), title=f"TASK SCOPE {task_id}"))
    _run(kernel.task_manager.start, kernel.principal("planner"), task_id)
    if finding:
        _run(
            kernel.task_manager.add_finding,
            kernel.principal("planner"),
            task_id,
            statement=finding,
            kind=kind,
            suggests_core_change=suggests_core_change,
        )
        _ok("finding recorded on the task (NOT on the research state)")
        if suggests_core_change:
            console.print(
                "[yellow]this finding suggests a core change: file a state transition request and "
                "have a human approve it (`researchos task transition request ...`)[/yellow]"
            )
    if close:
        _run(
            kernel.task_manager.close,
            kernel.principal("planner"),
            task_id,
            summary=close,
        )
        _ok("task closed")


@task_app.command("transition")
def task_transition(
    action: str = typer.Argument(..., help="request | approve | reject | list | preview"),
    path: Optional[str] = typer.Option(None, "--path", help="Guarded path, e.g. core_question."),
    value: Optional[str] = typer.Option(None, "--value", help="New value (JSON for structured fields)."),
    reason: str = typer.Option("", "--reason"),
    evidence: Optional[str] = typer.Option(None, "--evidence", help="Comma-separated evidence ids."),
    task_id: Optional[str] = typer.Option(None, "--task"),
    str_id: Optional[str] = typer.Option(None, "--str"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """State Transition Requests: the only legal path to a research-direction change."""
    kernel = _kernel(root)

    if action == "list":
        table = Table("str", "change", "risk", "by", "status", "reason")
        for request in kernel.transitions.history():
            table.add_row(
                request.str_id.split("_")[1][-8:], request.change_class.value, request.risk.value,
                request.requested_by, request.status.value, request.reason[:50],
            )
        console.print(table)
        return

    if action == "preview":
        if not str_id:
            _fail("--str is required for preview")
        for row in _run(kernel.transitions.diff_preview, str_id):
            guarded = " [red]guarded[/red]" if row["guarded"] else ""
            console.print(f"{row['op']} {row['path']}{guarded}\n  before: {row['before']}\n  after:  {row['after']}")
        return

    if action == "request":
        if not path:
            _fail("--path is required for request")
        parsed_value: object = value
        if value and value.strip().startswith(("{", "[")):
            parsed_value = json.loads(value)
        operations = [{"op": "set", "path": path, "value": parsed_value}]
        request = _run(
            kernel.transitions.request,
            kernel.principal("planner"),
            operations=operations,
            reason=reason or "state transition requested from the CLI",
            evidence_ids=[e.strip() for e in (evidence or "").split(",") if e.strip()],
            task_id=task_id,
        )
        _ok(f"state transition request {request.str_id} filed (status PROPOSED)")
        console.print("a human must approve it: researchos task transition approve --str <id> --reason '...'")
        return

    if action in {"approve", "reject"}:
        if not str_id:
            _fail("--str is required")
        if action == "approve":
            request, decision = _run(
                kernel.transitions.approve, kernel.human(), str_id, note=reason
            )
            _ok(f"applied at revision {request.applied_revision}; decision {decision.decision_id}")
        else:
            request = _run(kernel.transitions.reject, kernel.human(), str_id, reason=reason or "rejected")
            _ok(f"rejected {request.str_id}")
        return

    _fail(f"unknown action {action!r}: expected request|approve|reject|list|preview")


# ======================================================================================
# experiment
# ======================================================================================


@experiment_app.command("register")
def experiment_register(
    title: str = typer.Argument(...),
    model: Optional[str] = typer.Option(None, "--model"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    scale: Optional[str] = typer.Option(None, "--scale"),
    seeds: str = typer.Option("", "--seeds", help="Comma-separated seeds."),
    n: Optional[int] = typer.Option(None, "--n"),
    control: Optional[str] = typer.Option(None, "--control", help="Control arm name."),
    treatment: Optional[str] = typer.Option(None, "--treatment"),
    matched: str = typer.Option("", "--matched", help="Comma-separated matched conditions."),
    has_intervention: bool = typer.Option(False, "--intervention"),
    necessity: bool = typer.Option(False, "--necessity"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file to hash into provenance."),
    claim: Optional[str] = typer.Option(None, "--claim", help="Claim this experiment supports."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Register an experiment. The design flags decide the evidence level it can ever support."""
    from ..kernel.provenance import capture_provenance
    from ..models.experiment import Arm, Experiment, ExperimentDesign, TrainingBudget

    kernel = _kernel(root)
    config_data: dict = {}
    if config:
        import yaml

        config_data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    seed_list = [int(s) for s in seeds.split(",") if s.strip().isdigit()]

    experiment = Experiment(
        title=title,
        model=model,
        dataset=dataset,
        scale=scale,
        seeds=seed_list,
        n=n,
        treatment=Arm(name=treatment or "treatment", is_control=False) if treatment else None,
        control=Arm(name=control, is_control=True) if control else None,
        matched_conditions=[m.strip() for m in matched.split(",") if m.strip()],
        design=ExperimentDesign(
            has_control=bool(control),
            has_matched_conditions=bool(matched.strip()),
            has_intervention=has_intervention,
            has_necessity_design=necessity,
        ),
        training_budget=TrainingBudget(
            steps=int(config_data["steps"]) if str(config_data.get("steps", "")).isdigit() else None
        ),
        config=config_data,
        claim_ids=[claim] if claim else [],
        provenance=capture_provenance(kernel.root, config=config_data or None, seeds=seed_list),
        status=__import__("researchos.models", fromlist=["ExperimentStatus"]).ExperimentStatus.PLANNED,
    )
    from ..agents import ExperimentDesigner

    designer = ExperimentDesigner(kernel)
    _run(designer.register, experiment)
    _ok(f"experiment {experiment.experiment_id} registered; design ceiling {experiment.evidence_level().value}")
    gaps = designer.design_gaps(experiment)
    if gaps:
        _warn("design gaps (these cap the claim you can make):")
        for gap in gaps:
            console.print(f"  - {gap}")


@experiment_app.command("list")
def experiment_list(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    kernel = _kernel(root)
    table = Table("id", "title", "status", "ceiling", "matched", "seeds")
    for experiment in kernel.experiments.all():
        table.add_row(
            experiment.experiment_id.split("_")[1][-8:], experiment.title[:44], experiment.status.value,
            experiment.evidence_level().value, str(len(experiment.matched_conditions)), str(experiment.seeds),
        )
    console.print(table)


@experiment_app.command("run")
def experiment_run(
    experiment_id: str = typer.Argument(...),
    artifact: Optional[Path] = typer.Option(None, "--artifact", help="Raw artifact to hash into provenance."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Record the execution of a registered experiment (artifact hashing, never result editing)."""
    kernel = _kernel(root)
    agent = __import__("researchos.agents", fromlist=["ExperimentAgent"]).ExperimentAgent(kernel)
    experiment = _run(kernel.experiments.require, experiment_id)
    if artifact:
        ref = _run(agent.register_raw_artifact, experiment_id, str(artifact))
        _ok(f"raw artifact registered: {ref.path} sha256={ref.sha256[:16]}…")
    else:
        _warn("no --artifact given: the run has no hashed raw data behind it yet")
    console.print(
        "[dim]ResearchOS does not execute your training code: register the artifact your runner "
        "produced so every number traces back to bytes.[/dim]"
    )
    _ = experiment


# ======================================================================================
# evidence
# ======================================================================================


@evidence_app.command("list")
def evidence_list(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    kernel = _kernel(root)
    table = Table("id", "type", "level", "verification", "statement")
    for item in kernel.evidence.all():
        table.add_row(
            item.evidence_id.split("_")[1][-8:], item.evidence_type.value, item.evidence_level.value,
            item.verification_status.value, item.statement[:60],
        )
    console.print(table)


@evidence_app.command("verify")
def evidence_verify(
    evidence_id: str = typer.Argument(...),
    method: str = typer.Option("re-hash artifacts and check provenance", "--method"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Verify evidence by re-hashing its artifacts. A changed file becomes a conflict, not a warning."""
    from ..evidence import EvidenceRegistry

    kernel = _kernel(root)
    updated = _run(EvidenceRegistry(kernel).verify, kernel.principal("analysis"), evidence_id, method=method)
    _ok(
        f"evidence {evidence_id} -> verification {updated.verification_status.value}, "
        f"type {updated.evidence_type.value}"
    )
    for note in updated.verification_notes:
        console.print(f"  - {note}")


@evidence_app.command("trace")
def evidence_trace(
    claim_id: str = typer.Argument(...),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Print the provenance chain: claim → evidence → analysis → experiment → artifacts → code."""
    from ..evidence import EvidenceGraph

    kernel = _kernel(root)
    result = _run(EvidenceGraph(kernel).trace_claim, claim_id)
    console.print(Panel.fit(json.dumps(result, indent=2, default=str), title=f"PROVENANCE {claim_id}"))


# ======================================================================================
# claim
# ======================================================================================


@claim_app.command("list")
def claim_list(
    status: Optional[str] = typer.Option(None, "--status"),
    live: bool = typer.Option(False, "--live"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """List claims with their status, evidence level and permitted language."""
    from ..claims import ClaimRegistry

    kernel = _kernel(root)
    registry = ClaimRegistry(kernel)
    claims = registry.by_status(status) if status else (registry.live() if live else registry.all())
    table = Table("id", "status", "level", "may say", "ev", "statement")
    for claim in claims:
        table.add_row(
            claim.claim_id.split("_")[1][-8:], claim.status.value, claim.evidence_level.value,
            claim.language_strength or "", str(len(claim.evidence_ids)), claim.statement[:60],
        )
    console.print(table)
    _json(registry.summary())


@claim_app.command("propose")
def claim_propose(
    statement: str = typer.Argument(...),
    scope: str = typer.Option(..., "--scope", help="Where the claim is asserted to hold."),
    evidence: str = typer.Option("", "--evidence", help="Comma-separated evidence ids."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Create a claim as IDEA. Promotion is a separate, evidence-gated step."""
    from ..claims import ClaimLifecycle

    kernel = _kernel(root)
    claim = _run(
        ClaimLifecycle(kernel).create,
        kernel.principal("claim_manager"),
        statement=statement,
        scope=scope,
        evidence_ids=[e.strip() for e in evidence.split(",") if e.strip()],
    )
    _ok(f"claim {claim.claim_id} created as IDEA (scope: {scope})")


@claim_app.command("transition")
def claim_transition(
    claim_id: str = typer.Argument(...),
    to_status: str = typer.Argument(...),
    reason: str = typer.Option("", "--reason"),
    evidence: str = typer.Option("", "--evidence"),
    superseded_by: Optional[str] = typer.Option(None, "--superseded-by"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Move a claim along the lifecycle. Every rung checks its own obligation."""
    from ..claims import ClaimLifecycle

    kernel = _kernel(root)
    principal = (
        kernel.human() if to_status.upper() in {"SUPPORTED", "ROBUST"} else kernel.principal("claim_manager")
    )
    claim = _run(
        ClaimLifecycle(kernel).transition,
        principal,
        claim_id,
        to_status,
        reason=reason,
        evidence_ids=[e.strip() for e in evidence.split(",") if e.strip()],
        superseded_by=superseded_by,
    )
    _ok(f"claim {claim_id} -> {claim.status.value} (language now: {claim.language_strength!r})")


@claim_app.command("audit")
def claim_audit(
    claim_id: Optional[str] = typer.Option(None, "--claim"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Check claim/evidence/language obligations and show what would move each claim forward."""
    from ..claims import ClaimLifecycle, ClaimRegistry

    kernel = _kernel(root)
    registry = ClaimRegistry(kernel)
    if claim_id:
        console.print_json(json.dumps(_run(ClaimLifecycle(kernel).explain, claim_id), indent=2, default=str))
        return
    overclaiming = registry.overclaiming()
    if overclaiming:
        console.print("[yellow]claims whose wording exceeds their evidence (with the calibrated rewrite):[/yellow]")
        for claim, safe in overclaiming:
            console.print(f"  - {claim.claim_id}: {claim.statement[:80]}\n    -> {safe[:90]}")
    blocked = registry.blocked()
    for claim, reasons in blocked:
        _warn(f"{claim.claim_id} is blocked by unresolved conflict(s): {reasons[0][:100]}")
    _json(registry.summary())


# ======================================================================================
# literature / novelty / prior art
# ======================================================================================


@literature_app.command("search")
def literature_search(
    target: str = typer.Argument(..., help="Question, mechanism or claim to search for."),
    limit: int = typer.Option(10, "--limit"),
    offline: bool = typer.Option(False, "--offline", help="Use only local/cached sources (no network)."),
    bib: Optional[Path] = typer.Option(None, "--bib", help="Local .bib file to include."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Plan and execute a broad literature search across all query families."""
    from ..literature.providers.registry import default_providers

    kernel = _kernel(root)
    try:
        from ..literature.query_planner import QueryPlanner
    except ImportError:
        _fail("literature query planner unavailable (module missing)")
    planner = QueryPlanner(kernel)
    plan = _run(planner.plan, kernel.principal("literature_researcher"), target)
    console.print(f"planned {len(plan.queries)} queries across {len({q.family for q in plan.queries})} families")
    providers = default_providers(kernel, offline=offline, bib_paths=[bib] if bib else [])
    plan = _run(planner.execute, kernel.principal("literature_researcher"), plan, providers, limit=limit)
    coverage = planner.compute_coverage(plan)
    _ok(f"{coverage.queries_executed} queries executed; {coverage.papers_screened} papers screened")
    for line in coverage.deficits():
        _warn(f"coverage deficit: {line}")
    console.print(f"coverage score {coverage.score:.2f} — {coverage.basis}")


@literature_app.command("map")
def literature_map(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """Show the literature graph: clusters, closest prior work, coverage."""
    kernel = _kernel(root)
    try:
        from ..literature.graph import LiteratureGraph
    except ImportError:
        _fail("literature graph unavailable (module missing)")
    graph = LiteratureGraph(kernel)
    console.print_json(json.dumps(graph.summary(), indent=2, default=str))
    clusters = graph.clusters()
    console.print(f"[bold]clusters[/bold]: {len(clusters)}")
    for index, cluster in enumerate(clusters[:10], start=1):
        console.print(f"  {index}: {', '.join(c[:12] for c in cluster[:8])}")


@novelty_app.command("audit")
def novelty_audit(
    statement: str = typer.Argument(..., help="The novelty claim to test, e.g. 'X has not been compared under Y'."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Run a novelty audit. Coverage decides whether a 'no match found' verdict is even expressible."""
    kernel = _kernel(root)
    try:
        from ..literature.novelty import NoveltyAuditor
    except ImportError:
        _fail("novelty auditor unavailable (module missing)")
    audit = _run(NoveltyAuditor(kernel).audit, kernel.principal("novelty_auditor"), statement)
    console.print(f"[bold]verdict:[/bold] {audit.verdict.value}")
    console.print(f"gap kind: {audit.gap_kind.value}")
    console.print(f"coverage: {audit.coverage.basis or 'no coverage measured'}")
    if not audit.coverage_is_sufficient():
        _warn("coverage below thresholds: the verdict is forced to NOVELTY_UNCERTAIN")
    console.print(f"[bold]safe sentence:[/bold] {audit.verdict_sentence()}")


@literature_app.command("prior-art")
def literature_prior_art(
    title: str = typer.Option("Prior-art comparison", "--title"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Render the prior-art matrix for the imported/registered papers (UNKNOWN stays UNKNOWN)."""
    kernel = _kernel(root)
    try:
        from ..literature.prior_art import PriorArtMatrixBuilder, render_markdown
    except ImportError:
        _fail("prior-art builder unavailable (module missing)")
    papers = kernel.papers.all()
    if not papers:
        _fail("no papers registered yet: run `researchos literature search` first")
    builder = PriorArtMatrixBuilder(kernel)
    matrix = _run(
        builder.build,
        kernel.human(),
        title=title,
        rows=[(p.paper_id, p.title[:40]) for p in papers],
    )
    console.print(render_markdown(matrix))


# ======================================================================================
# audits
# ======================================================================================


@audit_app.command("stats")
def audit_stats(
    experiment: Optional[str] = typer.Option(None, "--experiment"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Statistical audit: recompute from artifacts and check the design's arithmetic."""
    from ..analysis import StatisticalAuditor

    kernel = _kernel(root)
    audit = _run(
        StatisticalAuditor(kernel).audit,
        kernel.principal("analysis"),
        experiment_ids=[experiment] if experiment else (),
    )
    _print_audit(audit)


@app.command("mechanism")
def mechanism_audit(
    claim: Optional[str] = typer.Option(None, "--claim"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Mechanism audit: does the language match the rung of the ladder the evidence reaches?"""
    from ..analysis import MechanismAuditor

    kernel = _kernel(root)
    audit = _run(
        MechanismAuditor(kernel).audit,
        kernel.principal("mechanism_auditor"),
        claim_ids=[claim] if claim else (),
    )
    _print_audit(audit)


@app.command("timeline")
def timeline_command(
    action: str = typer.Argument("show", help="show | summary | claim | why | ask"),
    ref: Optional[str] = typer.Option(None, "--ref", help="Explain one object (claim/experiment/str id)."),
    claim_id: Optional[str] = typer.Option(None, "--claim", help="Claim history."),
    question: Optional[str] = typer.Option(None, "--question", help="One of the standing questions."),
    limit: int = typer.Option(30, "--limit"),
    json_out: bool = typer.Option(False, "--json"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """The research timeline: why the claim was cancelled, the route changed, the control was added."""
    kernel = _kernel(root)
    view = kernel.history

    if action == "summary":
        _json(view.summary())
        return

    if action == "claim":
        if not claim_id:
            _fail("--claim is required")
        _json(view.claim_history(claim_id))
        return

    if action == "why":
        if not ref and not question:
            _fail("--ref or --question is required")
        records = view.why(ref=ref, text=question)
        if not records:
            _warn("no decision, transition or timeline event matches")
        for record in records:
            console.print(
                f"[bold]{record['kind']}[/bold] {record.get('at', '')} {record.get('id', '')}\n"
                f"  {record.get('summary', '')}\n  rationale: {record.get('rationale', '')}"
            )
        return

    if action == "ask":
        if not question:
            _fail("--question is required")
        _json(view.answer(question))
        return

    if json_out:
        _json(
            [
                {"phase": phase, "events": [event.model_dump(mode="json") for event in events]}
                for phase, events in view.narrative()
            ]
        )
        return

    for phase, events in view.narrative():
        table = Table(title=f"{phase.upper()} ({len(events)})")
        table.add_column("when")
        table.add_column("kind")
        table.add_column("who")
        table.add_column("what", overflow="fold")
        for event in events[-limit:]:
            marker = "[dim](imported)[/dim]" if event.imported else ""
            table.add_row(
                f"{event.at:%Y-%m-%d %H:%M}", event.kind.value, event.actor,
                f"{event.title} {marker}",
            )
        console.print(table)


@app.command("redteam")
def redteam(
    claim: Optional[str] = typer.Option(None, "--claim"),
    subject: str = typer.Option("current research state", "--subject"),
    json_out: bool = typer.Option(False, "--json"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Adversarial review before a claim or a release: what would falsify this, and which weakness is first."""
    from ..agents import RedTeamAgent

    kernel = _kernel(root)
    agent = RedTeamAgent(kernel)
    report = _run(agent.review, subject=subject, claim_ids=[claim] if claim else ())
    if json_out:
        _json(report.model_dump(mode="json"))
        return
    console.print(f"[bold]RED TEAM REPORT[/bold] — {report.verdict} ({len(report.questions)} objection(s))")
    console.print(f"strongest objection: {report.strongest_objection}")
    console.print(f"weakest experiment: {report.weakest_experiment_id or 'n/a'}")
    table = Table("severity", "category", "question")
    for item in report.questions:
        table.add_row(item.severity.value, item.category, item.question[:96])
    console.print(table)
    if report.blocking():
        _warn(f"{len(report.blocking())} high-severity objection(s) are unaddressed")
    console.print(
        "[dim]the red team reports; it never edits research state (the report is stored under "
        ".researchos/paper/audits/)[/dim]"
    )


def _print_audit(audit) -> None:
    console.print(f"[bold]{audit.kind.value} audit: {audit.title}[/bold] -> {audit.verdict.value}")
    if audit.summary:
        console.print(audit.summary)
    table = Table("code", "severity", "blocks", "message")
    for finding in audit.findings:
        table.add_row(
            finding.code, finding.severity.value, "yes" if finding.blocks_publication else "",
            finding.message[:80],
        )
    console.print(table)
    if audit.findings:
        console.print("[dim]suggestions:[/dim]")
        for finding in audit.findings[:6]:
            if finding.suggestion:
                console.print(f"  - {finding.code}: {finding.suggestion[:110]}")


@audit_app.command("log")
def audit_log(
    action: str = typer.Argument("verify", help="verify | head | tail"),
    limit: int = typer.Option(20, "--limit"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Inspect and verify the append-only, hash-chained event log."""
    kernel = _kernel(root)
    if action == "verify":
        result = kernel.events.verify()
        if result.ok:
            _ok(result.summary())
        else:
            _fail(result.summary() + "\n" + "\n".join(result.problems[:10]))
        return
    records = list(kernel.events.iter_records())
    if action == "head":
        records = records[-limit:]
    else:
        records = records[:limit]
    table = Table("seq", "kind", "actor", "task", "payload")
    for record in records:
        table.add_row(
            str(record.seq), record.kind, record.actor, (record.task_id or "")[:12],
            json.dumps(record.payload, default=str)[:70],
        )
    console.print(table)


# ======================================================================================
# skills
# ======================================================================================


@skill_app.command("list")
def skill_list(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    from ..skills.registry import SkillRegistry

    kernel = _kernel(root)
    registry = SkillRegistry(kernel)
    table = Table("id", "status", "trust", "score", "source", "name")
    for skill in registry.list():
        table.add_row(
            skill.skill_id.split("_")[1][-8:], skill.status.value, skill.trust.value,
            f"{skill.quality_metrics.score():.2f}", skill.source.value, skill.name[:40],
        )
    console.print(table)


@skill_app.command("search")
def skill_search(
    capability: str = typer.Argument(..., help="Capability to find, e.g. 'citation verification'."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Search for candidate skills by *capability* (not by 'research agent')."""
    kernel = _kernel(root)
    try:
        from ..skills.discovery import SkillDiscoveryAgent
    except ImportError:
        _fail("skill discovery unavailable (module missing)")
    agent = SkillDiscoveryAgent(kernel)
    from ..kernel.permissions import Principal
    from ..models.skill import SkillGap

    gap = SkillGap(description=capability, missing_capability=capability)
    queries = agent.capability_queries(capability)
    console.print("[bold]capability queries[/bold] (searched all of these, not one keyword):")
    for query in queries:
        console.print(f"  - {query}")
    console.print(
        "[dim]pass a provider to execute the search; ResearchOS does not invent candidates "
        "it did not actually retrieve.[/dim]"
    )
    _ = gap, Principal


@skill_app.command("evaluate")
def skill_evaluate(
    skill_id: str = typer.Argument(...),
    content: Optional[Path] = typer.Option(None, "--content", help="Skill source file to inspect."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Static sandbox evaluation of an external skill (never executed)."""
    from ..skills.registry import SkillRegistry
    from ..skills.sandbox import SkillSandbox

    kernel = _kernel(root)
    card = _run(SkillRegistry(kernel).get, skill_id)
    if card is None:
        _fail(f"skill {skill_id} not found")
    text = content.read_text(encoding="utf-8") if content else None
    report = _run(
        SkillSandbox(kernel).inspect, kernel.principal("skill_evaluator"), card,
        content=text, path=content,
    )
    console.print(f"sandbox passed: [bold]{report.passed}[/bold] (static analysis only)")
    for finding in report.findings:
        console.print(f"  - [{finding.severity.value}] {finding.code}: {finding.message[:90]}")
    if report.protected_paths_touched:
        _warn(f"touched protected paths: {report.protected_paths_touched}")


@skill_app.command("benchmark")
def skill_benchmark(
    suite: str = typer.Option("LITERATURE", "--suite", help="LITERATURE|EXPERIMENT|MECHANISM|WRITING"),
    seed_tasks: bool = typer.Option(False, "--seed-tasks", help="Install the default task catalogue."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """List the fixed benchmark tasks for a suite (activation requires a real, executed run)."""
    from ..skills.benchmark_tasks import seed_default_tasks
    from ..skills.benchmark import BenchmarkHarness

    kernel = _kernel(root)
    if seed_tasks:
        tasks = _run(seed_default_tasks, kernel)
        _ok(f"{len(tasks)} benchmark task(s) registered")
    harness = BenchmarkHarness(kernel)
    tasks = harness.tasks_for(suite)
    table = Table("id", "name", "requires", "forbids")
    for task_ in tasks:
        table.add_row(
            task_.benchmark_task_id.split("_")[1][-8:], task_.name[:44],
            ", ".join(task_.required_findings[:3]), ", ".join(task_.forbidden_findings[:2]),
        )
    console.print(table)
    console.print(
        "[dim]a skill becomes ACTIVE only after a benchmark run clears the activation score AND "
        "passes the regression gate against the incumbent[/dim]"
    )


@skill_app.command("evolve")
def skill_evolve(
    parents: Optional[str] = typer.Option(None, "--from", help="Comma-separated parent skill ids for synthesis."),
    rule: str = typer.Option("", "--rule", help="The research-specific rule to inject."),
    name: Optional[str] = typer.Option(None, "--name"),
    description: str = typer.Option("", "--description"),
    suite: str = typer.Option("LITERATURE", "--suite"),
    findings: Optional[str] = typer.Option(
        None, "--findings", help="JSON map task_id -> [finding codes], for a deterministic dry run."
    ),
    content: Optional[Path] = typer.Option(None, "--content", help="Candidate source file to inspect."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Run a candidate skill through sandbox → benchmark → regression → ACTIVE or REJECT.

    With ``--findings`` the benchmark uses a deterministic runner (no model, no network), which is how
    the *gates* are tested; without it, a caller-supplied runner is required through the API.
    """
    from ..skills.benchmark_tasks import seed_default_tasks
    from ..skills.evolution import SkillEvolution, deterministic_runner
    from ..skills.registry import SkillRegistry

    kernel = _kernel(root)
    ensure = _run(seed_default_tasks, kernel)
    if ensure:
        console.print(f"[dim]{len(ensure)} benchmark task(s) available[/dim]")
    if not findings:
        _fail("--findings is required for a dry run (a real evaluation needs an injected runner)")
    import json as _jsonlib

    mapping = _jsonlib.loads(findings)
    runner = deterministic_runner(mapping)

    if not parents:
        _fail("--from is required: synthesis combines existing skills plus a research rule")
    parent_ids = [p.strip() for p in parents.split(",") if p.strip()]
    evolution = SkillEvolution(kernel)
    card, result = _run(
        evolution.synthesise_and_evaluate,
        kernel.principal("skill_synthesizer"),
        parents=parent_ids,
        rule=rule or "unspecified rule",
        name=name or "synthesised-skill",
        description=description,
        suite=suite,
        runner=runner,
        content=content.read_text(encoding="utf-8") if content else None,
    )
    console.print(result.summary())
    for note in result.notes:
        console.print(f"  [dim]{note}[/dim]")
    if result.sandbox and result.sandbox.findings:
        for finding in result.sandbox.findings:
            console.print(f"  sandbox: [{finding.severity.value}] {finding.code}: {finding.message[:80]}")
    if result.benchmark:
        console.print(
            f"  benchmark score {result.benchmark.score:.2f} "
            f"(regression passed: {result.benchmark.regression_passed})"
        )
    _json(evolution.history(card.skill_id))


@skill_app.command("install")
def skill_install(skill_id: str = typer.Argument(...), root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """Install a skill into the registry (human decision only)."""
    from ..skills.registry import SkillRegistry

    kernel = _kernel(root)
    card = _run(SkillRegistry(kernel).install, kernel.human(), skill_id)
    _ok(f"installed {card.skill_id} (status {card.status.value}, trust {card.trust.value})")


@skill_app.command("update")
def skill_update(
    skill_id: str = typer.Argument(...),
    benchmark_run: str = typer.Option(..., "--benchmark-run", help="Passing benchmark run id."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Activate a new skill version. Regressions against the incumbent block the upgrade."""
    from ..skills.lifecycle import SkillLifecycle

    kernel = _kernel(root)
    card = _run(SkillLifecycle(kernel).activate, kernel.human(), skill_id, benchmark_run_id=benchmark_run)
    _ok(f"{card.skill_id} is now ACTIVE (trust {card.trust.value})")


# ======================================================================================
# paper
# ======================================================================================


@paper_app.command("compile")
def paper_compile(
    title: Optional[str] = typer.Option(None, "--title"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the compiled markdown here."),
    format: str = typer.Option("markdown", "--format"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Compile a paper from approved claims, verified evidence and analysis artifacts."""
    from ..paper import PaperCompiler

    kernel = _kernel(root)
    artifact = _run(PaperCompiler(kernel).compile, kernel.principal("paper_writer"), title=title, format=format)
    report = artifact.grounding_report
    console.print(f"[bold]{artifact.title}[/bold]")
    console.print(f"sections: {len(artifact.sections)} · words: {artifact.word_count()}")
    if report:
        console.print(f"grounding: {report.summary()}")
    if artifact.refused_additions:
        _warn("the compiler refused to write:")
        for refusal in artifact.refused_additions[:8]:
            console.print(f"  - {refusal[:120]}")
    if report and not report.passed:
        _fail(f"compilation blocked by {len(report.blockers())} grounding violation(s)")
    if out:
        out.write_text(artifact.render(), encoding="utf-8")
        _ok(f"written to {out}")


@paper_app.command("audit")
def paper_audit(
    paper_id: Optional[str] = typer.Option(None, "--paper"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Run the AI-style / overclaim auditor over a compiled paper."""
    from ..paper import StyleAuditor

    kernel = _kernel(root)
    artifact = kernel.paper_artifacts.get(paper_id) if paper_id else (
        kernel.paper_artifacts.all()[-1] if kernel.paper_artifacts.all() else None
    )
    if artifact is None:
        _fail("no compiled paper found: run `researchos paper compile` first")
    audit = _run(
        StyleAuditor().audit_paper, kernel.principal("paper_auditor"), artifact, kernel=kernel
    )
    _print_audit(audit)


@paper_app.command("readiness")
def paper_readiness(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """Seven separate readiness dimensions. Deliberately no single total score."""
    from ..paper import ReadinessAssessor

    kernel = _kernel(root)
    readiness = _run(ReadinessAssessor(kernel).assess, kernel.human())
    table = Table("dimension", "score", "basis")
    for dimension in readiness.dimensions:
        table.add_row(dimension.label(), f"{dimension.score:.2f}", dimension.basis[:90])
    console.print(table)
    for blocker in readiness.blockers()[:10]:
        _warn(blocker)


# ======================================================================================
# agents / review
# ======================================================================================


@paper_app.command("voice")
def paper_voice(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """Show the real research arc and the researcher's own voice fingerprint."""
    from ..paper import ResearcherVoice

    kernel = _kernel(root)
    voice = ResearcherVoice(kernel)
    arc = voice.arc()
    console.print(f"[bold]REAL RESEARCH ARC[/bold] ({len(arc)} step(s))")
    table = Table("phase", "when", "what")
    for step in arc:
        table.add_row(step.phase, (step.at or "")[:16], step.title[:88])
    console.print(table)
    _json(voice.profile().as_dict())
    sentences = voice.narrative_sentences()
    if sentences:
        console.print("[bold]narrative sentences the compiler may use[/bold]")
        for text, section, _notes, _decisions, _claims in sentences:
            console.print(f"  [{section.value}] {text}")


@app.command("api")
def api_command(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    enable_actions: bool = typer.Option(
        False,
        "--enable-actions",
        help="Also mount the human-in-the-loop actions (review decisions, transition approval, "
        "evidence verification). Local clients only; every action runs as the human principal "
        "through the same kernel gate as the CLI.",
    ),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Serve the dashboard and the read-only JSON API (requires ``pip install researchos[api]``)."""
    try:
        import uvicorn  # type: ignore
    except ImportError:
        _fail("uvicorn is not installed: pip install 'researchos[api]'")
    from ..api.app import app_for_project

    kernel = _kernel(root)
    console.print(f"[bold]ResearchOS dashboard[/bold]  http://{host}:{port}/")
    console.print(f"[dim]project: {kernel.root} · JSON API: /docs · state plane: .researchos/[/dim]")
    if enable_actions:
        console.print(
            "[yellow]human actions enabled[/yellow] (review decisions, transition approval, evidence "
            "verification) — local clients only, executed as the human principal through the kernel gate"
        )
    else:
        console.print(
            "[dim]read-only: every mutation stays in the CLI, where the acting principal is explicit. "
            "Add --enable-actions for the human-in-the-loop actions.[/dim]"
        )
    uvicorn.run(
        app_for_project(str(kernel.root), enable_actions=enable_actions),
        host=host,
        port=port,
        log_level="info",
    )


@agent_app.command("list")
def agent_list(root: Optional[Path] = typer.Option(None, "--root")) -> None:
    """List the fifteen agents with their capability counts and what they may NOT do."""
    from ..agents import agent_table

    kernel = _kernel(root)
    table = Table("agent", "caps", "denied", "purpose")
    for name, purpose, caps, denied in agent_table(kernel):
        table.add_row(name, str(caps), ", ".join(denied[:2]), purpose[:60])
    console.print(table)


@review_app.command("list")
def review_list(
    pending_only: bool = typer.Option(True, "--pending/--all"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Show the human review queue produced by Research Import."""
    kernel = _kernel(root)
    items = kernel.pending_reviews() if pending_only else kernel.reviews.all()
    table = Table("id", "kind", "confidence", "title")
    for item in items:
        table.add_row(
            item.review_item_id.split("_")[1][-8:], item.kind.value, f"{item.confidence:.2f}", item.title[:70]
        )
    console.print(table)
    if not items and pending_only:
        _ok("review queue is empty")


@review_app.command("show")
def review_show(review_id: str = typer.Argument(...), root: Optional[Path] = typer.Option(None, "--root")) -> None:
    kernel = _kernel(root)
    item = _run(kernel.reviews.require, review_id)
    console.print(Panel.fit(json.dumps(item.model_dump(mode="json"), indent=2, default=str), title=item.title))


@review_app.command("decide")
def review_decide(
    review_id: str = typer.Argument(...),
    decision: str = typer.Argument(..., help="ACCEPTED | EDITED | REJECTED | DEFERRED"),
    note: str = typer.Option("", "--note"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Record a human decision on an imported candidate."""
    from ..models.review import ReviewDecision

    kernel = _kernel(root)
    item = _run(kernel.reviews.require, review_id)
    updated = item.with_updates(
        status=ReviewDecision(decision.upper()),
        decided_by=kernel.human().name,
        decided_at=__import__("researchos.models", fromlist=["utcnow"]).utcnow(),
        decision_note=note,
    )
    kernel.reviews.save(updated)
    kernel.events.append(
        "review.decided",
        actor=kernel.human().name,
        payload={"review_item_id": review_id, "decision": updated.status.value, "note": note[:200]},
    )
    _ok(f"{review_id} -> {updated.status.value}")


@app.command("agents")
def agents_alias(root: Optional[Path] = typer.Option(None, "--root")) -> None:  # pragma: no cover
    """Alias for `researchos agent list`."""
    agent_list(root=root)


def main(argv: Optional[list[str]] = None) -> None:
    """Console entry point."""
    try:
        app(args=argv)
    except ResearchOSError as exc:  # safety net: kernel errors never traceback at the user
        err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

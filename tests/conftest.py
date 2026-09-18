"""Shared pytest fixtures for the ResearchOS test suite.

Every test runs against a *real* project on a temporary filesystem: real YAML, real event log,
real hashes. Mocking the kernel would make the invariants untested, and the invariants are the
product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchos.kernel import ResearchKernel  # noqa: E402
from researchos.models import (  # noqa: E402
    Arm,
    Evidence,
    EvidenceType,
    Experiment,
    ExperimentDesign,
    ExperimentStatus,
    MetricRef,
    Provenance,
    TaskPriority,
    TaskPurpose,
    TrainingBudget,
)


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture()
def kernel(project_root: Path) -> ResearchKernel:
    return ResearchKernel.create(project_root, name="TestProject", domain="AI/ML")


@pytest.fixture()
def human(kernel: ResearchKernel):
    return kernel.human()


@pytest.fixture()
def planner(kernel: ResearchKernel):
    return kernel.principal("planner")


@pytest.fixture()
def analyst(kernel: ResearchKernel):
    return kernel.principal("analysis")


@pytest.fixture()
def writer(kernel: ResearchKernel):
    return kernel.principal("paper_writer")


def write_artifact(root: Path, relative: str, content: str) -> Path:
    """Create a file under the project root, making parents as needed."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def artifact_writer(project_root: Path):
    def _write(relative: str, content: str) -> Path:
        return write_artifact(project_root, relative, content)

    return _write


def make_experiment(
    *,
    title: str = "direct vs shufwrite",
    model: str = "gpt2-small",
    dataset: str = "wikitext-103",
    scale: str = "124M",
    seeds: list[int] | None = None,
    n: int = 3,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    control: bool = True,
    matched: bool = True,
    intervention: bool = False,
    necessity: bool = False,
    sufficiency: bool = False,
    rescue: bool = False,
    settings: list[str] | None = None,
    metrics: list[str] | None = None,
    **overrides,
) -> Experiment:
    """Build a valid experiment. Defaults are the *weakest* legal design; flags opt into strength."""
    seeds = [0, 1, 2] if seeds is None else seeds
    payload = dict(
        title=title,
        research_question="Does write placement in fast weights affect retention?",
        hypothesis="Direct writeback retains more than shuffled writeback.",
        independent_variables=[],
        dependent_variables=[],
        treatment=Arm(name="direct", description="write back to the originating positions", is_control=False),
        control=Arm(name="shufwrite", description="shuffled writeback", is_control=True) if control else None,
        matched_conditions=["tokenizer", "data order", "optimizer"] if matched else [],
        design=ExperimentDesign(
            has_control=control,
            has_matched_conditions=matched,
            has_intervention=intervention,
            has_necessity_design=necessity,
            has_sufficiency_design=sufficiency,
            has_rescue=rescue,
            settings=settings or [],
        ),
        model=model,
        dataset=dataset,
        scale=scale,
        optimizer="adamw",
        learning_rate=3e-4,
        seeds=seeds,
        n=n,
        training_budget=TrainingBudget(steps=10000, hardware="1x A100"),
        metrics=[MetricRef(name=m, definition=f"{m} computed on the held-out split") for m in (metrics or ["retention"])],
        provenance=Provenance(code_commit="deadbeef", random_seeds=seeds, config_hash="c0ffee"),
        status=status,
    )
    payload.update(overrides)
    return Experiment(**payload)


@pytest.fixture()
def experiment_factory():
    return make_experiment


def register_experiment(kernel: ResearchKernel, experiment: Experiment, principal=None) -> Experiment:
    """Persist an experiment as the experiment agent would."""
    principal = principal or kernel.principal("experiment")
    principal.require("experiment.register", "experiment")
    kernel.experiments.save(experiment)
    kernel.events.append(
        "experiment.registered",
        actor=principal.name,
        payload={"experiment_id": experiment.experiment_id, "status": experiment.status.value},
    )
    kernel.timeline.record(
        "EXPERIMENT_REGISTERED",
        f"Registered experiment: {experiment.title}",
        actor=principal.name,
        refs=[experiment.experiment_id],
        state_revision=kernel.state.revision(),
    )
    return experiment


def make_raw_evidence(
    *,
    statement: str = "direct beats shufwrite by 0.31",
    experiment_id: str | None = None,
    payload: dict | None = None,
    **overrides,
) -> Evidence:
    data = dict(
        statement=statement,
        evidence_type=EvidenceType.RAW,
        payload=payload if payload is not None else {"delta": 0.31},
        source_experiment=experiment_id,
        experiment_ids=[experiment_id] if experiment_id else [],
    )
    data.update(overrides)
    return Evidence(**data)


@pytest.fixture()
def evidence_factory():
    return make_raw_evidence


@pytest.fixture()
def register_experiment_fixture():
    """The module-level ``register_experiment`` helper, exposed as a callable fixture."""
    return register_experiment


@pytest.fixture(name="register_experiment")
def register_experiment_fixture_named():
    """Persist an experiment the way the experiment agent would."""
    return register_experiment


@pytest.fixture()
def import_task(kernel: ResearchKernel):
    """A pre-made IMPORT task, since the importer refuses to work without a task boundary."""
    return kernel.task_manager.create(
        kernel.principal("planner"),
        objective="reconstruct research state from imported material",
        purpose=TaskPurpose.IMPORT,
        priority=TaskPriority.PRIMARY,
        stop_condition="all material classified and review queue emitted",
    )

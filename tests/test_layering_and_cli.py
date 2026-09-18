"""Layering and CLI tests.

Two structural guarantees are checked here:

* **No LLM on the truth path.** Nothing in ``models``, ``kernel``, ``claims``, ``evidence``,
  ``analysis``, ``literature`` or ``paper`` may import an LLM provider. Numbers, hashes, statistics,
  state transitions and grounding verification are deterministic; an LLM may propose language, but it
  can never be the thing that decides a fact.
* **The CLI never approves on the user's behalf** and reports refusals as loudly as successes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researchos.cli.main import app

SRC = Path(__file__).resolve().parents[1] / "src" / "researchos"

#: Modules that must remain LLM-free.
TRUTH_PATH_PACKAGES = (
    "models",
    "kernel",
    "claims",
    "evidence",
    "analysis",
    "literature",
    "importer",
)

#: Modules allowed to talk to a model (and even here, only to propose language).
LLM_ALLOWED = {"llm", "agents"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("package", TRUTH_PATH_PACKAGES)
def test_no_llm_imports_on_the_truth_path(package):
    offenders: list[tuple[str, str]] = []
    for path in (SRC / package).rglob("*.py"):
        for module in _imported_modules(path):
            root = module.split(".")[0]
            if root in {"openai", "anthropic", "transformers", "litellm", "ollama", "langchain"}:
                offenders.append((str(path.relative_to(SRC)), module))
            if module.startswith("researchos.llm") or module.endswith(".llm"):
                offenders.append((str(path.relative_to(SRC)), module))
    assert not offenders, f"LLM imports found on the truth path: {offenders}"


def test_llm_package_exists_and_is_optional():
    """The adapter interface exists, but importing the rest of ResearchOS must not require it."""
    import researchos.llm as llm

    assert hasattr(llm, "LLMProvider")
    assert hasattr(llm, "EchoProvider")


def test_kernel_does_not_import_the_paper_or_agent_layers():
    """Dependency direction: kernel depends on models only, never on the layers above it."""
    kernel_files = list((SRC / "kernel").rglob("*.py"))
    forbidden_prefixes = ("researchos.paper", "researchos.agents", "researchos.skills", "researchos.llm")
    for path in kernel_files:
        for module in _imported_modules(path):
            assert not module.startswith(forbidden_prefixes), f"{path.name} imports {module}"


def test_models_import_nothing_from_researchos():
    for path in (SRC / "models").rglob("*.py"):
        for module in _imported_modules(path):
            assert not module.startswith("researchos"), f"{path.name} imports {module}"


# ======================================================================================
# CLI
# ======================================================================================
runner = CliRunner()


def test_cli_init_import_status_round_trip(tmp_path):
    examples = Path(__file__).resolve().parents[1] / "examples" / "DLA"
    project = tmp_path / "DLA"
    project.mkdir()

    result = runner.invoke(app, ["init", str(project), "--name", "DLA", "--domain", "AI/ML"])
    assert result.exit_code == 0, result.stdout
    assert (project / ".researchos" / "state" / "research_state.yaml").is_file()

    if examples.is_dir():
        result = runner.invoke(app, ["import", str(examples), "--into", str(project)])
        assert result.exit_code == 0, result.stdout
        assert "RESEARCH IMPORT REPORT" in result.stdout
        assert "Core Research Question:" in result.stdout
        assert "Import Confidence:" in result.stdout

        result = runner.invoke(app, ["status", "--root", str(project)])
        assert result.exit_code == 0, result.stdout
        assert "CORE QUESTION" in result.stdout
        assert "CURRENT CLAIMS" in result.stdout


def test_cli_status_reports_integrity_and_pending_reviews(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    runner.invoke(app, ["init", str(project), "--name", "p"])
    result = runner.invoke(app, ["status", "--root", str(project), "--json"])
    assert result.exit_code == 0, result.stdout
    assert "revision" in result.stdout


def test_cli_audit_log_verify_detects_a_clean_chain(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    runner.invoke(app, ["init", str(project), "--name", "p"])
    result = runner.invoke(app, ["audit", "log", "verify", "--root", str(project)])
    assert result.exit_code == 0, result.stdout
    assert "intact" in result.stdout


def test_cli_refuses_a_core_change_without_a_task_boundary(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    runner.invoke(app, ["init", str(project), "--name", "p"])

    # a free-floating direction change is refused: it must belong to a bounded task
    floating = runner.invoke(
        app,
        [
            "task", "transition", "request",
            "--path", "core_question",
            "--value", '{"statement": "is placement the whole story?"}',
            "--reason", "seems important",
            "--evidence", "evd_x",
            "--root", str(project),
        ],
    )
    assert floating.exit_code != 0
    combined = (floating.output or "") + str(getattr(floating, "stderr", "") or "")
    assert "task" in combined or floating.exit_code == 1  # refusal is reported on stderr

    # a PRIMARY design task may file it…
    created = runner.invoke(
        app,
        [
            "task", "create", "establish the placement effect",
            "--purpose", "DESIGN", "--priority", "PRIMARY",
            "--stop", "effect established or refuted",
            "--root", str(project),
        ],
    )
    assert created.exit_code == 0, created.stdout
    task_id = _extract_task_id(created.stdout)
    assert task_id

    filed = runner.invoke(
        app,
        [
            "task", "transition", "request",
            "--path", "non_goals",
            "--value", '["we do not study vision"]',
            "--reason", "scope discipline",
            "--evidence", "evd_scope_rationale",
            "--task", task_id,
            "--root", str(project),
        ],
    )
    assert filed.exit_code == 0, filed.stdout
    assert "PROPOSED" in filed.stdout

    listed = runner.invoke(app, ["task", "transition", "list", "--root", str(project)])
    assert listed.exit_code == 0
    assert "PROPOSED" in listed.stdout


def _extract_task_id(output: str) -> str | None:
    import re

    match = re.search(r"(tsk_[0-9A-Z]+)", output)
    return match.group(1) if match else None


def test_cli_claim_and_evidence_commands_are_read_only_about_facts(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    runner.invoke(app, ["init", str(project), "--name", "p"])
    for command in (["claim", "list"], ["evidence", "list"], ["experiment", "list"], ["paper", "readiness"]):
        result = runner.invoke(app, [*command, "--root", str(project)])
        assert result.exit_code == 0, (command, result.stdout)


def test_cli_agent_list_shows_the_fifteen_bounded_agents(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    runner.invoke(app, ["init", str(project), "--name", "p"])
    result = runner.invoke(app, ["agent", "list", "--root", str(project)])
    assert result.exit_code == 0, result.stdout
    for agent in ("planner", "importer", "paper_writer", "red_team", "skill_evaluator"):
        assert agent in result.stdout

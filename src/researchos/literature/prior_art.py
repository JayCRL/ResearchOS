"""The prior-art matrix: comparison with three-valued honesty.

The rule this module exists to enforce: **``UNKNOWN`` is never converted to ``FALSE``.**
"Not described in the abstract" is not evidence of absence, and a comparison table that collapses the
two produces exactly the false novelty claims ResearchOS is built to prevent.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..kernel.errors import LiteratureError
from ..kernel.kernel import ResearchKernel
from ..kernel.permissions import Cap, Principal
from ..models.common import Tri
from ..models.literature import (
    MatrixCell,
    MatrixColumn,
    MatrixRow,
    PriorArtMatrix,
    ProviderKind,
)

#: The comparison axes for fast-weight / memory research. Each column states the question it answers
#: so a cell can never be filled by pattern-matching on the column label.
DEFAULT_COLUMNS: tuple[MatrixColumn, ...] = (
    MatrixColumn(key="fast_state", label="Fast state", question="Is there a distinct fast (rapidly updated) state?", requires_fulltext=True),
    MatrixColumn(key="slow_state", label="Slow state", question="Is there a separate slowly-updated parameter set?", requires_fulltext=True),
    MatrixColumn(key="explicit_selector", label="Explicit selector", question="Is there an explicit mechanism selecting what is written?", requires_fulltext=True),
    MatrixColumn(key="parameter_level_selection", label="Parameter-level selection", question="Is selection at parameter level rather than token/sequence level?", requires_fulltext=True),
    MatrixColumn(key="writeback", label="Writeback", question="Is there an explicit write-back step into the fast state?", requires_fulltext=True),
    MatrixColumn(key="alignment", label="Alignment", question="Is write position aligned with read position (as opposed to arbitrary)?", requires_fulltext=True),
    MatrixColumn(key="sleep_boundary", label="Sleep / boundary", question="Is there an offline consolidation or boundary step?", requires_fulltext=True),
    MatrixColumn(key="replay", label="Replay", question="Is past experience replayed during consolidation?", requires_fulltext=True),
    MatrixColumn(key="matched_energy_test", label="Matched-energy test", question="Is the comparison controlled for compute/parameter budget?", requires_fulltext=False),
)

CURRENT_WORK_REF = "current_work"


def _blank_cell(column: MatrixColumn) -> MatrixCell:
    return MatrixCell(value=Tri.UNKNOWN, note=f"not established for column {column.key!r}")


class PriorArtMatrixBuilder:
    """Builds and mutates prior-art matrices without ever inventing an answer."""

    def __init__(self, kernel: ResearchKernel) -> None:
        self.kernel = kernel

    # ------------------------------------------------------------------ build

    def build(
        self,
        principal: Principal,
        *,
        title: str,
        columns: Sequence[MatrixColumn] = DEFAULT_COLUMNS,
        rows: Sequence[tuple[str, str]],
        cells: Mapping[tuple[str, str], MatrixCell] | None = None,
        current_work_label: str = "This work",
    ) -> PriorArtMatrix:
        principal.require(Cap.LITERATURE_WRITE, "prior_art.build")
        if not rows:
            raise LiteratureError("a prior-art matrix needs at least one row")

        built_rows: list[MatrixRow] = []
        for work_ref, label in rows:
            is_current = work_ref == CURRENT_WORK_REF
            row_cells: dict[str, MatrixCell] = {}
            for column in columns:
                provided = (cells or {}).get((work_ref, column.key))
                if provided is not None:
                    row_cells[column.key] = provided
                elif is_current:
                    # We know our own work; we do not know the literature's internals.
                    row_cells[column.key] = MatrixCell(
                        value=Tri.UNKNOWN,
                        note="current work: fill from the registered design, not from prose",
                    )
                else:
                    row_cells[column.key] = _blank_cell(column)
            built_rows.append(
                MatrixRow(
                    work_ref=work_ref,
                    work_label=label,
                    cells=row_cells,
                    is_current_work=is_current,
                )
            )

        matrix = PriorArtMatrix(
            title=title,
            columns=list(columns),
            rows=built_rows,
            notes=[
                "UNKNOWN is a first-class answer: it is not FALSE.",
                "every TRUE/FALSE should name a paper, page or section where possible.",
            ],
        )
        self.kernel.matrices.save(matrix)
        self.kernel.events.append(
            "prior_art.built",
            actor=principal.name,
            payload={
                "matrix_id": matrix.matrix_id,
                "rows": len(matrix.rows),
                "columns": len(matrix.columns),
                "coverage": matrix.coverage(),
            },
        )
        return matrix

    # ------------------------------------------------------------------ mutate

    def set_cell(
        self, matrix: PriorArtMatrix, work_ref: str, column_key: str, cell: MatrixCell
    ) -> PriorArtMatrix:
        """Replace one cell. Converting UNKNOWN to FALSE requires documented absence evidence."""
        current = matrix.cell(work_ref, column_key)
        if current.value is Tri.UNKNOWN and cell.value is Tri.FALSE and not cell.explicit_absence_evidence:
            raise LiteratureError(
                f"refusing to record UNKNOWN -> FALSE for {work_ref}/{column_key} without "
                "explicit_absence_evidence=True. Absence of description in an abstract is not "
                "evidence that the feature is absent."
            )
        return self._replace(matrix, work_ref, column_key, cell, event="prior_art.cell_set")

    def upgrade_unknown(
        self,
        matrix: PriorArtMatrix,
        work_ref: str,
        column_key: str,
        *,
        value: Tri,
        note: str,
        paper_id: str | None = None,
        page: str | None = None,
        section: str | None = None,
        explicit_absence_evidence: bool = False,
    ) -> PriorArtMatrix:
        """The documented, auditable path from UNKNOWN to TRUE/FALSE."""
        if value is Tri.FALSE and not explicit_absence_evidence:
            raise LiteratureError(
                f"UNKNOWN -> FALSE for {work_ref}/{column_key} requires explicit_absence_evidence=True "
                "(the source states the feature is absent) — otherwise it stays UNKNOWN"
            )
        cell = MatrixCell(
            value=value,
            paper_id=paper_id,
            page=page,
            section=section,
            note=note,
            explicit_absence_evidence=explicit_absence_evidence,
            confidence=0.8 if (page or section) else 0.5,
        )
        return self._replace(matrix, work_ref, column_key, cell, event="prior_art.cell_upgraded")

    def _replace(
        self,
        matrix: PriorArtMatrix,
        work_ref: str,
        column_key: str,
        cell: MatrixCell,
        *,
        event: str,
    ) -> PriorArtMatrix:
        if column_key not in {c.key for c in matrix.columns}:
            raise LiteratureError(f"unknown matrix column {column_key!r}")
        rows: list[MatrixRow] = []
        found = False
        for row in matrix.rows:
            if row.work_ref != work_ref:
                rows.append(row)
                continue
            found = True
            rows.append(row.with_updates(cells={**row.cells, column_key: cell}))
        if not found:
            raise LiteratureError(f"unknown matrix row {work_ref!r}")
        updated = matrix.with_updates(rows=rows)
        self.kernel.matrices.save(updated)
        self.kernel.events.append(
            event,
            actor="literature",
            payload={
                "matrix_id": updated.matrix_id,
                "work_ref": work_ref,
                "column": column_key,
                "value": cell.value.value,
                "paper_id": cell.paper_id,
                "page": cell.page,
            },
        )
        return updated


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

_VALUE_GLYPH = {Tri.TRUE: "yes", Tri.FALSE: "no", Tri.UNKNOWN: "?"}


def coverage_line(matrix: PriorArtMatrix) -> str:
    coverage = matrix.coverage()
    return (
        f"coverage: {coverage['answered']}/{coverage['total_cells']} cells answered "
        f"({coverage['answered_fraction']:.0%}); {coverage['unknown']} UNKNOWN; "
        f"{coverage['fully_answered_rows']}/{len(matrix.rows)} rows fully answered"
    )


def render_markdown(matrix: PriorArtMatrix) -> str:
    """Render the matrix. UNKNOWN shows as ``?`` and the coverage line always follows."""
    header = ["Work"] + [column.label for column in matrix.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in matrix.rows:
        cells = [row.work_label]
        for column in matrix.columns:
            cell = row.cells.get(column.key, _blank_cell(column))
            glyph = _VALUE_GLYPH[cell.value]
            if cell.value in (Tri.TRUE, Tri.FALSE) and (cell.page or cell.section):
                glyph += f" ({cell.section or ''}{' p.' + cell.page if cell.page else ''})".strip()
            cells.append(glyph)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(coverage_line(matrix))
    if matrix.unknown_cells():
        lines.append(
            "UNKNOWN cells are unresolved: " + "; ".join(f"{w}/{c}" for w, c in matrix.unknown_cells()[:8])
        )
    return "\n".join(lines)


def cell_provenance(matrix: PriorArtMatrix, work_ref: str, column_key: str) -> dict[str, object]:
    cell = matrix.cell(work_ref, column_key)
    return {
        "work_ref": work_ref,
        "column": column_key,
        "value": cell.value.value,
        "paper_id": cell.paper_id,
        "page": cell.page,
        "section": cell.section,
        "confidence": cell.confidence,
        "quote": cell.quote,
        "literature_claim_id": cell.literature_claim_id,
        "explicit_absence_evidence": cell.explicit_absence_evidence,
    }


def provider_defaults() -> list[ProviderKind]:
    return [ProviderKind.LOCAL_BIB, ProviderKind.SEMANTIC_SCHOLAR, ProviderKind.ARXIV]

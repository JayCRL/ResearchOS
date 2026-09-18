"""Analysis artifacts — the *only* permitted source of numbers in a paper."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import ArtifactRef, RosModel, StrEnum, new_id, utcnow


class StatMethod(StrEnum):
    DESCRIPTIVE = "descriptive"
    STUDENT_TTEST = "student_t_test"
    WELCH_TTEST = "welch_t_test"
    PAIRED_TTEST = "paired_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP_CI = "bootstrap_ci"
    EFFECT_SIZE = "effect_size"
    HOLM_BONFERRONI = "holm_bonferroni"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    CORRELATION = "correlation"
    PROPORTION = "proportion"
    CUSTOM = "custom"


class StatResult(RosModel):
    """One named number with its statistical enclosures.

    ``analysis_id`` + ``result_id`` is the address the paper compiler uses for a numeral
    (``NumberRef``). A numeral in prose with no such address is a grounding violation.
    """

    result_id: str = Field(default_factory=lambda: new_id("finding"))
    name: str = Field(min_length=1, description="e.g. 'direct_minus_shufwrite.mean'")
    value: float | None = None
    unit: str | None = None

    n: int | None = Field(default=None, ge=0)
    mean: float | None = None
    std: float | None = None
    se: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None

    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None

    effect_size: float | None = None
    effect_size_kind: str | None = Field(
        default=None, description="cohens_d | hedges_g | cles | odds_ratio | relative_change"
    )

    test: StatMethod | None = None
    statistic: float | None = None
    df: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    alpha: float | None = 0.05
    significant: bool | None = None

    paired: bool | None = Field(
        default=None, description="Paired vs unpaired must be stated, not inferred from prose."
    )
    group_a: str | None = None
    group_b: str | None = None
    comparison_family: str | None = Field(
        default=None, description="Group id for multiple-comparison correction."
    )
    corrected_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    correction_method: StatMethod | None = None

    seed: int | None = None
    method: StatMethod = StatMethod.DESCRIPTIVE
    inputs: list[str] = Field(
        default_factory=list, description="Artifact paths or 'path#column' selectors actually read."
    )
    n_inputs: int | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "StatResult":
        if self.method is StatMethod.PAIRED_TTEST and self.paired is False:
            raise ValueError(f"{self.name}: paired t-test cannot be declared unpaired")
        if self.method in (StatMethod.STUDENT_TTEST, StatMethod.WELCH_TTEST) and self.paired is True:
            raise ValueError(f"{self.name}: unpaired test declared as paired — use the paired test")
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(f"{self.name}: ci_low > ci_high")
        return self

    def is_inferential(self) -> bool:
        return self.p_value is not None or self.test not in (None, StatMethod.DESCRIPTIVE)


class Analysis(RosModel):
    """A deterministic computation over artifacts, plus its own provenance."""

    analysis_id: str = Field(default_factory=lambda: new_id("analysis"))
    title: str = Field(min_length=1)
    question: str = ""
    claim_ids: list[str] = Field(default_factory=list)

    experiment_ids: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)

    method: StatMethod = StatMethod.DESCRIPTIVE
    method_description: str = ""
    parameters: dict[str, object] = Field(default_factory=dict)
    code_artifact: ArtifactRef | None = None
    script_hash: str | None = None
    seed: int | None = None
    deterministic: bool = Field(
        default=False, description="True only when no RNG/network/clock entered the computation."
    )

    results: list[StatResult] = Field(default_factory=list)
    output_artifact: ArtifactRef | None = None
    produced_evidence_ids: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    excluded_inputs: list[tuple[str, str]] = Field(
        default_factory=list, description="(input, reason) for every dropped run — exclusions are recorded."
    )

    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "analysis"
    task_id: str | None = None
    verified_by: str | None = None
    verified_at: datetime | None = None
    verification_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Analysis":
        if not self.results and not self.warnings:
            raise ValueError(
                f"analysis {self.analysis_id}: must produce results or explain why it produced none"
            )
        if self.verified_by is not None and self.verified_at is None:
            raise ValueError(
                f"analysis {self.analysis_id}: verification requires a timestamp, not just an author"
            )
        return self

    @property
    def is_verified(self) -> bool:
        return self.verified_by is not None and self.verified_at is not None

    def result(self, name_or_id: str) -> StatResult | None:
        for result in self.results:
            if result.name == name_or_id or result.result_id == name_or_id:
                return result
        return None

    def numeric_map(self) -> dict[str, float]:
        """Flatten every numeric value into ``name -> value`` — the allowed-number set for papers."""
        out: dict[str, float] = {}
        for result in self.results:
            for key in (
                "value",
                "mean",
                "std",
                "se",
                "median",
                "min",
                "max",
                "ci_low",
                "ci_high",
                "effect_size",
                "statistic",
                "p_value",
                "corrected_p_value",
            ):
                value = getattr(result, key)
                if isinstance(value, (int, float)):
                    out[f"{result.name}.{key}"] = float(value)
        return out

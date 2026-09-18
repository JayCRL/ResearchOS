"""Analysis OS: statistics, statistical audit, mechanism audit.

The rule this package exists to enforce: **numbers come from artifacts, not from prose**. Every
statistic here is computed from raw values, every inferential result declares whether it was paired
or unpaired, and a p-value is only ever reported when it was actually computed.
"""

from .mechanism_audit import MechanismAuditor
from .statistical_audit import StatisticalAuditor
from .statistics import (
    MannWhitneyResult,
    TTestResult,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    difference_ci,
    hedges_g,
    holm_bonferroni,
    mann_whitney_u,
    mean,
    mean_ci,
    median,
    normal_cdf,
    normal_ppf,
    one_sample_t_test,
    paired_t_test,
    quantile,
    std,
    stderr,
    student_t_test,
    t_cdf,
    t_ppf,
    t_sf,
    to_stat_result,
    variance,
    welch_t_test,
)

__all__ = [
    "MechanismAuditor",
    "MannWhitneyResult",
    "StatisticalAuditor",
    "TTestResult",
    "benjamini_hochberg",
    "bootstrap_ci",
    "cohens_d",
    "difference_ci",
    "hedges_g",
    "holm_bonferroni",
    "mann_whitney_u",
    "mean",
    "mean_ci",
    "median",
    "normal_cdf",
    "normal_ppf",
    "one_sample_t_test",
    "paired_t_test",
    "quantile",
    "std",
    "stderr",
    "student_t_test",
    "t_cdf",
    "t_ppf",
    "t_sf",
    "to_stat_result",
    "variance",
    "welch_t_test",
]

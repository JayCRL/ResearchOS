"""Deterministic statistics — the arithmetic an LLM is never allowed to perform.

**Why this module exists.** ResearchOS separates *computing* a number from *writing* a sentence
about it. Every p-value, confidence interval, effect size and corrected p-value that can appear in
a compiled paper has to be produced by code that a reader can re-run and re-check. An LLM in this
system may propose hypotheses and draft prose; it may not produce a t statistic. This module is the
deterministic half of that boundary: standard library only, no network, no LLM, no global RNG.

Three rules shape the design, and they are the reason some functions raise while others return a
result object full of ``None``:

1. **Never return a p-value you did not compute.** If a sample is too small, has zero variance, or
   is empty, no test was performed. The result object says so via ``p_value=None`` and a
   human-readable ``notes`` entry. Returning ``0.0``, ``1.0`` or ``nan`` would let a degenerate run
   masquerade as evidence.
2. **Never propagate NaN.** Descriptive helpers (``mean``, ``variance``, ``std``, ``stderr``,
   ``quantile``) raise :class:`ValueError` on input for which the quantity is genuinely undefined
   (an empty sample, a one-element sample variance). That is a loud failure at the call site, which
   is what a determinism-critical library wants; silent NaN is what a paper compiler must never see.
   The *inferential* entry points take the opposite stance, because they are called from audits that
   must finish: they return a degenerate result with notes instead of raising.
3. **Determinism is explicit.** The only randomised routine (:func:`bootstrap_ci`) takes a ``seed``
   and builds its own :class:`random.Random`; it never touches module-level RNG state, so two calls
   with the same seed are bit-identical and no other code can perturb them.

Hand-verifiable reference points (asserted in ``tests/test_statistics.py``):
``t_ppf(0.975, 10) = 2.228138852``, ``t_ppf(0.975, 1) = 12.706204736``, ``t_ppf(0.95, 30) = 1.697260887``,
``t_cdf(0.0, df) = 0.5``, and ``t_ppf(0.975, df) -> 1.959963985`` as ``df -> inf``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from ..models.analysis import StatMethod, StatResult

# --------------------------------------------------------------------------------------
# Tunables, all named so they appear in the audit trail rather than as magic numbers
# --------------------------------------------------------------------------------------

#: Below this many observations a t-interval's normality assumption is untestable; we say so.
SMALL_SAMPLE_N = 5

#: Relative/absolute slack used when comparing a numeral in prose with an artifact value at tol=0.
#: This is *float-representation* repair (``0.1 + 0.2`` style), not a significance fudge.
_FLOAT_REL_TOL = 1e-9
_FLOAT_ABS_TOL = 1e-12

#: Degrees of freedom above which the t distribution is indistinguishable from the normal one at
#: double precision; the shortcut avoids a slowly converging continued fraction.
_NORMAL_LIMIT_DF = 1.0e9

#: |t| above which the t CDF is 0/1 to double precision; avoids ``1 - 0.0`` cancellation.
_T_OVERFLOW = 1.0e10

__all__ = [
    "MannWhitneyResult",
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
    "regularized_incomplete_beta",
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


# --------------------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------------------


def _floats(values: Sequence[float], *, what: str) -> list[float]:
    """Materialise a sequence of floats, refusing NaN and infinities.

    A non-finite input makes almost every downstream statistic meaningless (``inf - inf`` is NaN),
    so it is rejected here rather than silently averaged.
    """
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"{what}: input contains a non-finite value ({number!r}); ResearchOS never lets "
                "NaN/inf reach a reported statistic"
            )
        out.append(number)
    return out


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Raises on an empty sample: the mean of nothing is not a number."""
    data = _floats(values, what="mean")
    if not data:
        raise ValueError("mean of an empty sample is undefined (refusing to return 0.0 or NaN)")
    return math.fsum(data) / len(data)


def variance(values: Sequence[float], *, sample: bool = True) -> float:
    """Variance. ``sample=True`` uses the ``n-1`` (Bessel-corrected) estimator.

    The uncorrected ``n`` estimator is available because it is occasionally the honest one (a
    complete population of seeds), but the default is the estimator a paper must report.
    """
    data = _floats(values, what="variance")
    n = len(data)
    if n == 0:
        raise ValueError("variance of an empty sample is undefined")
    if sample and n < 2:
        raise ValueError(
            "sample variance requires n >= 2; with a single observation the quantity is undefined "
            "(use sample=False only if the single value is a whole population)"
        )
    centre = math.fsum(data) / n
    return math.fsum((x - centre) ** 2 for x in data) / (n - 1 if sample else n)


def std(values: Sequence[float], *, sample: bool = True) -> float:
    """Standard deviation. See :func:`variance` for the estimator and the degenerate cases."""
    return math.sqrt(variance(values, sample=sample))


def stderr(values: Sequence[float]) -> float:
    """Standard error of the mean, ``sd / sqrt(n)`` (sample SD)."""
    data = _floats(values, what="stderr")
    if len(data) < 2:
        raise ValueError("standard error requires n >= 2 observations")
    return std(data) / math.sqrt(len(data))


def quantile(values: Sequence[float], q: float) -> float:
    """Quantile with linear interpolation between order statistics (the numpy/``method='linear'`` rule).

    ``h = (n-1) * q``; the result interpolates between ``x[floor(h)]`` and ``x[ceil(h)]``. Chosen
    over the nearest-rank rule because it is continuous in ``q``, which keeps bootstrap intervals
    from jumping between order statistics.
    """
    data = sorted(_floats(values, what="quantile"))
    if not data:
        raise ValueError("quantile of an empty sample is undefined")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile requires 0 <= q <= 1, got {q!r}")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[int(position)]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def median(values: Sequence[float]) -> float:
    """Median, i.e. ``quantile(values, 0.5)``."""
    return quantile(values, 0.5)


# --------------------------------------------------------------------------------------
# The t and normal distributions (stdlib only: no scipy)
# --------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function, evaluated with modified Lentz.

    Lentz's method avoids the explicit ``0/0`` that the naive recursion hits and converges in a few
    tens of iterations for the parameter ranges the t distribution needs.
    """
    max_iterations = 400
    epsilon = 3.0e-16
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        # Even step of the standard three-term recurrence.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` — the regularised incomplete beta function, in ``[0, 1]``.

    This is the primitive behind :func:`t_cdf`: ``P(T <= t)`` for a t distribution with ``df``
    degrees of freedom equals ``1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)`` for ``t > 0``.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"regularized_incomplete_beta requires a, b > 0, got a={a!r}, b={b!r}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_front += a * math.log(x) + b * math.log1p(-x)
    # Use the continued fraction directly when it converges fast, else the symmetric identity.
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def _check_df(df: float) -> float:
    value = float(df)
    if math.isnan(value) or value <= 0.0:
        raise ValueError(f"degrees of freedom must be a positive number, got {df!r}")
    return value


def _t_beta_term(t: float, df: float) -> float:
    """``I_x(df/2, 1/2)`` with ``x = df / (df + t^2)`` — the shared tail term of CDF and SF."""
    x = df / (df + t * t) if math.isfinite(t) else 0.0
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def t_cdf(t: float, df: float) -> float:
    """``P(T <= t)`` for Student's t with ``df`` degrees of freedom.

    Verified numerically against direct quadrature of the t density in the test suite, so the
    continued-fraction implementation cannot silently drift.
    """
    if math.isnan(t):
        raise ValueError("t_cdf received NaN")
    df = _check_df(df)
    if t == 0.0:
        return 0.5
    if not math.isfinite(t):
        return 1.0 if t > 0 else 0.0
    if df > _NORMAL_LIMIT_DF:
        return normal_cdf(t)
    term = _t_beta_term(t, df)
    return 1.0 - 0.5 * term if t > 0 else 0.5 * term


def t_sf(t: float, df: float) -> float:
    """``P(T > t)`` — the survival function.

    Computed directly rather than as ``1 - t_cdf`` so far-tail p-values keep their precision
    (subtracting from 1 destroys every significant digit below ~1e-16).
    """
    if math.isnan(t):
        raise ValueError("t_sf received NaN")
    df = _check_df(df)
    if t == 0.0:
        return 0.5
    if not math.isfinite(t):
        # A diverging statistic is "more extreme than any computable p"; 0.0 is the honest limit.
        return 0.0 if t > 0 else 1.0
    if df > _NORMAL_LIMIT_DF:
        return 0.5 * math.erfc(t / math.sqrt(2.0))
    term = _t_beta_term(t, df)
    return 0.5 * term if t > 0 else 1.0 - 0.5 * term


def t_ppf(p: float, df: float) -> float:
    """Inverse t CDF (quantile), used for every t-based confidence interval.

    Solved by bisection on a bracket that is expanded until it straddles ``p``. Bisection was
    chosen over Newton because the t quantile function is extremely flat for large ``df`` and
    Newton's step there is numerically unstable; 200 halvings take the bracket far below double
    precision, and the bracket is widened geometrically so that ``pf(0.999999, 1)`` still lands.
    """
    if math.isnan(p):
        raise ValueError("t_ppf received NaN probability")
    df = _check_df(df)
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p == 0.5:
        return 0.0
    if df > _NORMAL_LIMIT_DF:
        return normal_ppf(p)

    low, high = -1.0, 1.0
    while t_cdf(low, df) > p:
        low *= 2.0
        if low < -_T_OVERFLOW:
            return -math.inf
    while t_cdf(high, df) < p:
        high *= 2.0
        if high > _T_OVERFLOW:
            return math.inf
    for _ in range(200):
        middle = 0.5 * (low + high)
        if t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
        if high - low <= 1e-16 * max(1.0, abs(middle)):
            break
    return 0.5 * (low + high)


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: float) -> float:
    """Standard normal CDF, ``0.5 * erfc(-z / sqrt(2))``."""
    if math.isnan(z):
        raise ValueError("normal_cdf received NaN")
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    """Standard normal quantile.

    Acklam's rational approximation seeds a single Newton step against the *exact* CDF
    (:func:`normal_cdf`), which pushes the result to full double precision without any iteration
    count that could differ between platforms.
    """
    if math.isnan(p):
        raise ValueError("normal_ppf received NaN probability")
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p == 0.5:
        return 0.0

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    lower, upper = 0.02425, 1.0 - 0.02425
    if p < lower:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= upper:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # One Halley refinement against the exact CDF: e = F(x) - p, u = e * sqrt(2pi) * exp(x^2/2).
    error = normal_cdf(x) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


# --------------------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------------------


def _pooled_sd(a: list[float], b: list[float]) -> float | None:
    """Pooled SD with the ``(n-1)`` weights; ``None`` when neither sample varies."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    var_a = variance(a)
    var_b = variance(b)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    return math.sqrt(pooled)


def cohens_d(a: Sequence[float], b: Sequence[float], *, paired: bool = False) -> float:
    """Standardised mean difference ``(mean_a - mean_b) / sd``.

    Unpaired: pooled within-group SD (Cohen's *d*). Paired: SD of the within-pair differences
    (Cohen's *d_z*) — using the pooled SD for paired data would understate the effect whenever
    pairing removes between-subject variance, which is exactly the case pairing exists for.

    With a zero denominator the value is the mathematical limit: ``0.0`` when the means coincide,
    otherwise ``+/-inf``. ``inf`` is returned (never NaN) and the calling result object records a
    note, because "the groups are constant and differ" is a fact worth stating plainly.
    """
    left = _floats(a, what="cohens_d(a)")
    right = _floats(b, what="cohens_d(b)")
    if not left or not right:
        raise ValueError("cohens_d requires two non-empty samples")
    diff = mean(left) - mean(right)
    if paired:
        if len(left) != len(right):
            raise ValueError(
                f"cohens_d(paired=True) needs equal-length samples, got {len(left)} and {len(right)}"
            )
        if len(left) < 2:
            raise ValueError("cohens_d(paired=True) requires n >= 2 pairs")
        differences = [x - y for x, y in zip(left, right)]
        spread = std(differences)
    else:
        spread = _pooled_sd(left, right)
        if spread is None:
            raise ValueError("cohens_d requires n >= 2 in both samples")
    if spread == 0.0:
        return 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    return diff / spread


def hedges_g(a: Sequence[float], b: Sequence[float], *, paired: bool = False) -> float:
    """Bias-corrected standardised mean difference (Hedges' *g*).

    ``g = J * d`` with the exact correction factor ``J = 1 - 3 / (4 * df - 1)``, which shrinks the
    small-sample upward bias of Cohen's *d*. Reported alongside *d* rather than instead of it,
    because *d* is what most readers can check by hand.
    """
    left = _floats(a, what="hedges_g(a)")
    right = _floats(b, what="hedges_g(b)")
    d = cohens_d(left, right, paired=paired)
    if paired:
        df = len(left) - 1
    else:
        df = len(left) + len(right) - 2
    if df <= 1:
        return d
    correction = 1.0 - 3.0 / (4.0 * df - 1)
    return d * correction


# --------------------------------------------------------------------------------------
# Inferential results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TTestResult:
    """Outcome of a t test, including what the test *could not* do.

    Every field that could not be computed is ``None`` and every reason is in ``notes``. An audit
    that finds ``p_value is None`` knows immediately that the number was not computed, which is the
    whole point: ResearchOS prefers an explicit hole to a plausible-looking number.
    """

    t: float | None
    df: float | None
    p_value: float | None
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    std_a: float | None
    std_b: float | None
    se_diff: float | None
    paired: bool
    method: str
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    effect_size: float | None
    effect_size_kind: str | None
    notes: tuple[str, ...] = field(default=())

    @property
    def statistic(self) -> float | None:
        """The test statistic under a name shared with :class:`MannWhitneyResult`."""
        return self.t

    @property
    def difference(self) -> float | None:
        """Point estimate of ``mean_a - mean_b``, or ``None`` when a mean is unknown."""
        if self.mean_a is None or self.mean_b is None:
            return None
        return self.mean_a - self.mean_b

    def is_computed(self) -> bool:
        return self.p_value is not None


@dataclass(frozen=True)
class MannWhitneyResult:
    """Outcome of a Mann-Whitney U test (rank-based, no normality assumption)."""

    u: float | None
    df: float | None
    p_value: float | None
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    std_a: float | None
    std_b: float | None
    se_diff: float | None
    paired: bool
    method: str
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    effect_size: float | None
    effect_size_kind: str | None
    notes: tuple[str, ...] = field(default=())

    @property
    def statistic(self) -> float | None:
        return self.u

    @property
    def difference(self) -> float | None:
        if self.mean_a is None or self.mean_b is None:
            return None
        return self.mean_a - self.mean_b

    def is_computed(self) -> bool:
        return self.p_value is not None


def _describe(values: list[float]) -> tuple[int, float | None, float | None]:
    """``(n, mean, sample_sd)`` with ``None`` where the statistic is undefined."""
    n = len(values)
    if n == 0:
        return 0, None, None
    return n, mean(values), (std(values) if n >= 2 else None)


def _too_few_result(
    a: list[float],
    b: list[float],
    *,
    paired: bool,
    method: str,
    level: float,
    extra_notes: tuple[str, ...] = (),
) -> TTestResult:
    """Result for a comparison that was never performed because the sample is too small."""
    n_a, mean_a, std_a = _describe(a)
    n_b, mean_b, std_b = _describe(b)
    notes = (
        f"no test performed: {method} requires n >= 2 per sample (n_a={n_a}, n_b={n_b})",
        "p_value, test statistic, confidence interval and effect size are all None because no "
        "number was computed",
        *extra_notes,
    )
    return TTestResult(
        t=None,
        df=None,
        p_value=None,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        se_diff=None,
        paired=paired,
        method=method,
        ci_low=None,
        ci_high=None,
        ci_level=level,
        effect_size=None,
        effect_size_kind=None,
        notes=notes,
    )


def _small_sample_note(n_a: int, n_b: int) -> tuple[str, ...]:
    if min(n_a, n_b) < SMALL_SAMPLE_N:
        return (
            f"small sample (n_a={n_a}, n_b={n_b} < {SMALL_SAMPLE_N}): the t interval leans on an "
            "approximate-normality assumption that this sample cannot check",
        )
    return ()


def _effect_size_fields(
    a: list[float], b: list[float], *, paired: bool
) -> tuple[float | None, str | None, tuple[str, ...]]:
    """Cohen's *d* plus a note whenever the value is degenerate (infinite)."""
    try:
        d = cohens_d(a, b, paired=paired)
    except ValueError:
        return None, None, ("effect size not computed: the samples are too small",)
    if not math.isfinite(d):
        return (
            d,
            "cohens_d",
            (
                "effect size is infinite: the comparison has zero variance, so the standardised "
                "difference diverges (the groups are constant and their means differ)",
            ),
        )
    return d, "cohens_d", ()


def _finish(
    *,
    statistic: float | None,
    df: float | None,
    se_diff: float | None,
    a: list[float],
    b: list[float],
    paired: bool,
    method: str,
    level: float,
    notes: tuple[str, ...],
) -> TTestResult:
    """Assemble a completed t result from a statistic, its df and its standard error."""
    n_a, mean_a, std_a = _describe(a)
    n_b, mean_b, std_b = _describe(b)
    difference = (mean_a - mean_b) if (mean_a is not None and mean_b is not None) else None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    degenerate = se_diff == 0.0
    if degenerate:
        notes = (
            *notes,
            "zero standard error: the t statistic is undefined (a 0/0 or x/0 form), so no p-value "
            "was computed; the effect is either non-existent or numerically exact within this sample",
        )
    elif statistic is not None and df is not None and math.isfinite(statistic):
        p_value = 2.0 * t_sf(abs(statistic), df)
        critical = t_ppf(1.0 - (1.0 - level) / 2.0, df)
        if difference is not None:
            ci_low = difference - critical * se_diff
            ci_high = difference + critical * se_diff
    effect, kind, effect_notes = _effect_size_fields(a, b, paired=paired)
    if degenerate and difference is not None:
        # Interval of zero width is the honest statement: this sample contains no variability.
        ci_low = ci_high = difference
    notes = (*notes, *effect_notes, *_small_sample_note(n_a, n_b), "two-sided test")
    return TTestResult(
        t=statistic,
        df=df,
        p_value=p_value,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        se_diff=se_diff,
        paired=paired,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=level,
        effect_size=effect,
        effect_size_kind=kind,
        notes=notes,
    )


def student_t_test(
    a: Sequence[float], b: Sequence[float], *, paired: bool = False
) -> TTestResult:
    """Student's t test: pooled-variance two-sample test, or the paired test when ``paired=True``.

    ``paired=True`` delegates to :func:`paired_t_test`, because "Student's t" and "the paired t"
    are the same statistic applied to the within-pair differences; keeping one entry point makes it
    impossible for a caller to ask for a paired comparison and silently receive an unpaired one.
    """
    if paired:
        return paired_t_test(a, b)
    left = _floats(a, what="student_t_test(a)")
    right = _floats(b, what="student_t_test(b)")
    if len(left) < 2 or len(right) < 2:
        return _too_few_result(left, right, paired=False, method="student", level=0.95)
    pooled = _pooled_sd(left, right)
    assert pooled is not None  # guaranteed by the n >= 2 check above
    se_diff = pooled * math.sqrt(1.0 / len(left) + 1.0 / len(right))
    df = float(len(left) + len(right) - 2)
    difference = mean(left) - mean(right)
    statistic = None if se_diff == 0.0 else difference / se_diff
    return _finish(
        statistic=statistic,
        df=df,
        se_diff=se_diff,
        a=left,
        b=right,
        paired=False,
        method="student",
        level=0.95,
        notes=(
            "pooled-variance (Student) t test: assumes equal population variances",
        ),
    )


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> TTestResult:
    """Welch's t test: unequal-variance two-sample test with the Welch–Satterthwaite df.

    This is the default comparison in the statistical auditor because unequal variances between
    arms are the norm in ML experiments and the pooled test is anti-conservative when they differ.
    """
    left = _floats(a, what="welch_t_test(a)")
    right = _floats(b, what="welch_t_test(b)")
    if len(left) < 2 or len(right) < 2:
        return _too_few_result(left, right, paired=False, method="welch", level=0.95)
    n_a, n_b = len(left), len(right)
    var_a, var_b = variance(left), variance(right)
    term_a, term_b = var_a / n_a, var_b / n_b
    se_diff = math.sqrt(term_a + term_b)
    difference = mean(left) - mean(right)
    statistic = None if se_diff == 0.0 else difference / se_diff
    # Welch–Satterthwaite: df = (A + B)^2 / (A^2/(na-1) + B^2/(nb-1)).
    denominator = term_a**2 / (n_a - 1) + term_b**2 / (n_b - 1)
    notes: tuple[str, ...] = (
        "Welch t test: does not assume equal variances (Welch-Satterthwaite df)",
    )
    if denominator == 0.0:
        df: float | None = None
        notes = (*notes, "degrees of freedom undefined (both arms have zero variance)")
        statistic = None
    else:
        df = (term_a + term_b) ** 2 / denominator
    return _finish(
        statistic=statistic,
        df=df,
        se_diff=se_diff,
        a=left,
        b=right,
        paired=False,
        method="welch",
        level=0.95,
        notes=notes,
    )


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> TTestResult:
    """Paired t test on the within-pair differences.

    Used when each observation in ``a`` is matched to one in ``b`` (same seed, same unit, same
    starting point). Pairing removes between-unit variance from the standard error, so on genuinely
    paired data this test is strictly more powerful than the unpaired one — which is why the
    statistical auditor treats "was this paired?" as a question that must be *declared*.
    """
    left = _floats(a, what="paired_t_test(a)")
    right = _floats(b, what="paired_t_test(b)")
    if len(left) != len(right):
        return _too_few_result(
            left,
            right,
            paired=True,
            method="paired",
            level=0.95,
            extra_notes=(
                f"paired test needs equal-length samples: got {len(left)} and {len(right)}; "
                "re-pair the observations (or use the unpaired test) instead of truncating",
            ),
        )
    if len(left) < 2:
        return _too_few_result(left, right, paired=True, method="paired", level=0.95)
    differences = [x - y for x, y in zip(left, right)]
    n = len(differences)
    spread = std(differences)
    se_diff = spread / math.sqrt(n)
    statistic = None if se_diff == 0.0 else mean(differences) / se_diff
    return _finish(
        statistic=statistic,
        df=float(n - 1),
        se_diff=se_diff,
        a=left,
        b=right,
        paired=True,
        method="paired",
        level=0.95,
        notes=(f"paired t test on {n} within-pair differences (df = n - 1)",),
    )


def one_sample_t_test(a: Sequence[float], mu: float = 0.0) -> TTestResult:
    """One-sample t test of ``mean(a)`` against a hypothesised value ``mu``.

    ``mu`` is not a field of :class:`TTestResult`, so it is carried in ``notes`` and recorded as
    ``mean_b`` — the null value is part of the result's meaning and must not be lost. ``n_b`` is 0
    because there is no second sample.
    """
    left = _floats(a, what="one_sample_t_test(a)")
    target = float(mu)
    if not math.isfinite(target):
        raise ValueError(f"one_sample_t_test requires a finite mu, got {mu!r}")
    if len(left) < 2:
        degenerate = _too_few_result(left, [], paired=False, method="student", level=0.95)
        return replace(
            degenerate,
            mean_b=target,
            n_b=0,
            notes=(
                f"no test performed: a one-sample t test requires n >= 2 (n={len(left)})",
                f"the hypothesised value mu={target} is recorded even though no test ran",
            ),
        )
    n = len(left)
    spread = std(left)
    se_diff = spread / math.sqrt(n)
    statistic = None if se_diff == 0.0 else (mean(left) - target) / se_diff
    result = _finish(
        statistic=statistic,
        df=float(n - 1),
        se_diff=se_diff,
        a=left,
        b=[],
        paired=False,
        method="student",
        level=0.95,
        notes=(f"one-sample t test against mu={target}",),
    )
    # ``mean_b``/``std_b`` must describe the null hypothesis, not a replicated constant sample.
    return replace(result, mean_b=target, std_b=None, n_b=0)


# --------------------------------------------------------------------------------------
# Mann-Whitney U (rank-based, no normality assumption)
# --------------------------------------------------------------------------------------


def _ranks(values: list[float]) -> tuple[list[float], int]:
    """Average ranks for ties; returns ``(ranks, tie_correction_term)``.

    The tie term ``sum(t^3 - t)`` is what the variance of U must be corrected by — ignoring ties
    understates the variance and therefore overstates significance.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_term = 0
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0  # ranks are 1-based
        for position in range(index, end + 1):
            ranks[order[position]] = average
        size = end - index + 1
        if size > 1:
            tie_term += size**3 - size
        index = end + 1
    return ranks, tie_term


def _rank_sum_counts(n_total: int, size: int) -> list[float]:
    """Exact null distribution of the rank sum: ``counts[s]`` = number of ``size``-subsets of
    ``{1..n_total}`` whose rank sum is ``s``.

    Built by the standard subset-sum DP (walk the ranks downwards so each rank is used once).
    Counts stay integers well below ``2**53`` for the sizes we allow, so float accumulation is exact.
    """
    max_sum = size * (2 * n_total - size + 1) // 2
    table = [[0.0] * (max_sum + 1) for _ in range(size + 1)]
    table[0][0] = 1.0
    for rank in range(1, n_total + 1):
        for chosen in range(min(size, rank), 0, -1):
            row = table[chosen]
            previous = table[chosen - 1]
            for total in range(max_sum, rank - 1, -1):
                if previous[total - rank]:
                    row[total] += previous[total - rank]
    return table[size]


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> MannWhitneyResult:
    """Mann-Whitney U test with an exact null distribution for small tie-free samples.

    Why an exact branch exists at all: with 3 seeds per arm — routine in ML — the normal
    approximation's p-values are noticeably wrong, and a wrong p-value that looks plausible is the
    most dangerous kind. When ``n_a + n_b <= 24`` and there are no ties, the p-value is enumerated
    from the exact null distribution of the rank sum. Larger or tied samples fall back to the normal
    approximation with tie correction and a continuity correction, and the ``notes`` say which path
    was taken so the reader never has to guess.
    """
    left = _floats(a, what="mann_whitney_u(a)")
    right = _floats(b, what="mann_whitney_u(b)")
    n_a, n_b = len(left), len(right)
    if n_a < 1 or n_b < 1:
        return MannWhitneyResult(
            u=None,
            df=None,
            p_value=None,
            n_a=n_a,
            n_b=n_b,
            mean_a=mean(left) if left else None,
            mean_b=mean(right) if right else None,
            std_a=std(left) if n_a >= 2 else None,
            std_b=std(right) if n_b >= 2 else None,
            se_diff=None,
            paired=False,
            method="mann_whitney",
            ci_low=None,
            ci_high=None,
            ci_level=0.95,
            effect_size=None,
            effect_size_kind=None,
            notes=("no test performed: Mann-Whitney U needs at least one observation per sample",),
        )

    combined = [*left, *right]
    ranks, tie_term = _ranks(combined)
    rank_sum_a = math.fsum(ranks[:n_a])
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    n_total = n_a + n_b
    notes: list[str] = [
        f"U reported for the first sample (U_a={u_a:g}, U_b={u_b:g}); the two are mirror images",
        "no confidence interval is produced: Mann-Whitney estimates a rank shift, not a mean "
        "difference — use bootstrap_ci/difference_ci for an interval",
    ]
    p_value: float | None = None
    exact_possible = tie_term == 0 and n_total <= 24
    if exact_possible:
        counts = _rank_sum_counts(n_total, n_a)
        total = float(math.comb(n_total, n_a))
        observed = int(round(rank_sum_a))
        upper = sum(counts[observed:]) / total
        lower = sum(counts[: observed + 1]) / total
        p_value = min(1.0, 2.0 * min(upper, lower))
        notes.append(
            f"exact two-sided p from the enumerated null distribution of the rank sum "
            f"(C({n_total},{n_a})={int(total)} arrangements, no ties)"
        )
    else:
        mean_u = n_a * n_b / 2.0
        variance_u = (n_a * n_b / 12.0) * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
        if variance_u <= 0.0:
            notes.append("zero variance under the null: the U statistic is degenerate")
        else:
            # Continuity correction: move the observed U half a unit toward its null mean.
            deviation = u_a - mean_u
            corrected = deviation - 0.5 * math.copysign(1.0, deviation) if deviation else 0.0
            z = corrected / math.sqrt(variance_u)
            p_value = min(1.0, 2.0 * (0.5 * math.erfc(abs(z) / math.sqrt(2.0))))
            reason = "ties present" if tie_term else f"n_a+n_b={n_total} > 24"
            notes.append(
                f"normal approximation with tie correction and continuity correction ({reason}); "
                "the exact distribution is not enumerated for this sample size"
            )
    effect_size = 1.0 - 2.0 * u_a / (n_a * n_b)  # rank-biserial correlation
    return MannWhitneyResult(
        u=u_a,
        df=float(n_total - 2),
        p_value=p_value,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean(left),
        mean_b=mean(right),
        std_a=std(left) if n_a >= 2 else None,
        std_b=std(right) if n_b >= 2 else None,
        se_diff=None,
        paired=False,
        method="mann_whitney",
        ci_low=None,
        ci_high=None,
        ci_level=0.95,
        effect_size=effect_size,
        effect_size_kind="rank_biserial",
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------------------


def _check_level(level: float) -> float:
    value = float(level)
    if not 0.0 < value < 1.0:
        raise ValueError(f"confidence level must lie strictly between 0 and 1, got {level!r}")
    return value


def mean_ci(values: Sequence[float], *, level: float = 0.95) -> tuple[float, float]:
    """t-based confidence interval for the mean.

    With fewer than two observations the interval is ``(-inf, +inf)`` rather than a zero-width
    interval around the single value: one observation carries no information about sampling
    variability, and reporting ``(x, x)`` would claim a precision we do not have. An infinite
    interval is impossible to mistake for a result.
    """
    data = _floats(values, what="mean_ci")
    level = _check_level(level)
    if len(data) < 2:
        return (-math.inf, math.inf)
    centre = mean(data)
    se = std(data) / math.sqrt(len(data))
    critical = t_ppf(1.0 - (1.0 - level) / 2.0, len(data) - 1)
    return (centre - critical * se, centre + critical * se)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    The bootstrap is the only randomised routine in this module, so it takes an explicit ``seed``
    and constructs its own :class:`random.Random`: no module-level state is read or written, and two
    calls with the same seed return identical bounds. ``statistic`` is applied to resamples of the
    *same size* as the input (the standard non-parametric bootstrap; resampling fewer points would
    estimate the wrong sampling distribution).
    """
    data = _floats(values, what="bootstrap_ci")
    level = _check_level(level)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}")
    if not data:
        return (-math.inf, math.inf)
    if len(data) == 1:
        # The resampling distribution is a point mass: the honest bootstrap answer is that point.
        return (data[0], data[0])
    rng = random.Random(seed)
    n = len(data)
    draws: list[float] = []
    for _ in range(n_resamples):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        value = float(statistic(resample))
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return (-math.inf, math.inf)
    alpha = 1.0 - level
    return (quantile(draws, alpha / 2.0), quantile(draws, 1.0 - alpha / 2.0))


def difference_ci(
    a: Sequence[float], b: Sequence[float], *, paired: bool = False, level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for ``mean_a - mean_b`` (Welch when unpaired, paired t when paired)."""
    level = _check_level(level)
    if paired:
        result = paired_t_test(a, b)
    else:
        result = welch_t_test(a, b)
    if result.ci_low is None or result.ci_high is None:
        return (-math.inf, math.inf)
    return (result.ci_low, result.ci_high)


# --------------------------------------------------------------------------------------
# Multiplicity corrections
# --------------------------------------------------------------------------------------


def _check_pvalues(pvalues: Sequence[float]) -> list[float]:
    out = _floats(pvalues, what="p-values")
    for value in out:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-values must lie in [0, 1], got {value!r}")
    return out


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, returned in the input order.

    Adjusted ``p_(i) = max_{j <= i} min(1, (m - j + 1) * p_(j))`` over p-values sorted ascending.
    The running maximum enforces monotonicity, which is what makes the adjusted values directly
    comparable with a fixed alpha. Holm controls the family-wise error rate without the
    independence assumption that plain Bonferroni needs no matter what.
    """
    values = _check_pvalues(pvalues)
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for position, index in enumerate(order):
        candidate = (m - position) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini–Hochberg step-up adjusted p-values (FDR), in the input order.

    Adjusted ``p_(i) = min_{j >= i} min(1, (m / j) * p_(j))`` over p-values sorted ascending.
    Reported alongside Holm rather than instead of it: BH controls the false discovery rate and is
    the right correction when many comparisons are exploratory, Holm is the right one when any
    single false positive would be costly. The auditor never chooses for the researcher — it
    requires that *a* correction was applied and names both.
    """
    values = _check_pvalues(pvalues)
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (values[i], i))
    adjusted = [0.0] * m
    running = 1.0
    for position in range(m - 1, -1, -1):
        index = order[position]
        rank = position + 1
        candidate = (m / rank) * values[index]
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------------------
# Adapter into the frozen StatResult model
# --------------------------------------------------------------------------------------

_METHOD_TO_STAT_METHOD: dict[str, StatMethod] = {
    "student": StatMethod.STUDENT_TTEST,
    "welch": StatMethod.WELCH_TTEST,
    "paired": StatMethod.PAIRED_TTEST,
    "mann_whitney": StatMethod.MANN_WHITNEY_U,
}


def to_stat_result(
    name: str, result: TTestResult | MannWhitneyResult, **extra: object
) -> StatResult:
    """Adapt a test result into the persisted :class:`StatResult` model.

    Two conventions are encoded here and are worth reading before trusting the fields:

    * ``value`` and ``mean`` are both the *difference of means*, because ``name`` is expected to
      name a difference (the model's own example is ``direct_minus_shufwrite.mean``). ``std`` is
      therefore left ``None``: there is one dispersion that matters for a comparison, and it is
      ``se`` (the standard error of the difference).
    * ``n`` is the **effective per-arm n** (``min(n_a, n_b)``; ``n_a`` for a paired test). A
      per-arm n of 2 is not rescued by having 40 observations in the other arm, and an audit that
      reads total n would miss exactly that failure. Per-arm counts are spelled out in ``notes``.

    A degenerate comparison yields a ``StatResult`` with ``p_value=None`` and the explanatory notes
    attached, never a fabricated number.
    """
    n_effective = result.n_a if result.paired else min(result.n_a, result.n_b)
    known = set(StatResult.model_fields)
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ValueError(
            f"to_stat_result({name!r}): unknown StatResult field(s) {unknown}; "
            "the model forbids extra keys by design"
        )
    difference = result.difference
    data: dict[str, object] = {
        "name": name,
        "value": difference,
        "n": n_effective,
        "mean": difference,
        "std": None,
        "se": result.se_diff,
        "median": None,
        "min": None,
        "max": None,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "ci_level": result.ci_level,
        "effect_size": result.effect_size,
        "effect_size_kind": result.effect_size_kind,
        "test": _METHOD_TO_STAT_METHOD.get(result.method),
        "statistic": result.statistic,
        "df": result.df,
        "p_value": result.p_value,
        "paired": result.paired,
        "warnings": [],
        "notes": [
            f"n_a={result.n_a}, n_b={result.n_b} (the 'n' field is the effective per-arm n)",
            *result.notes,
        ],
    }
    data.update(extra)
    alpha = float(data.get("alpha") or 0.05)  # type: ignore[arg-type]
    p_value = data.get("p_value")
    data["significant"] = (
        None if p_value is None else bool(float(p_value) < alpha)  # type: ignore[arg-type]
    )
    return StatResult.model_validate(data)

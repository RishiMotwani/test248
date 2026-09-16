"""Statistical rigor helpers (task F).

Pure, dependency-light routines the paper pipeline calls at aggregation time:

- ``bootstrap_mean_ci`` / ``bootstrap_ci_mean_diff``: percentile bootstrap
  confidence intervals (deterministic under a random-seeded RNG so repeated
  runs differ only by the seed).
- ``cohens_d_paired``: paired (within-run) effect size on the E1 token series.
- ``holms_correct`` / ``bonferroni_correct``: family-wise multiple-comparison
  correction for the method-vs-baseline wilcoxon comparisons.

Module-level ``_selftest`` is the --quick gate: small synthetic arrays, asserts
the CI brackets the sample mean and that Holm thresholds are monotone
non-decreasing in rank.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np


def bootstrap_mean_ci(xs: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                      rng: random.Random = None) -> List[float]:
    """Percentile-bootstrap 100*(1-alpha)% CI on the mean of ``xs``."""
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return [None, None]
    if rng is None:
        rng = random.Random(0)
    means = np.empty(n_boot)
    rint = np.random.default_rng(rng.randint(0, 2**32 - 1))
    for i in range(n_boot):
        means[i] = rint.choice(xs, size=xs.size, replace=True).mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return [round(float(lo), 3), round(float(hi), 3)]


def bootstrap_ci_mean_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 1000,
                           alpha: float = 0.05, rng: random.Random = None) -> List[float]:
    """Percentile-bootstrap CI on the paired mean difference (a - b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return [None, None]
    if rng is None:
        rng = random.Random(1)
    rint = np.random.default_rng(rng.randint(0, 2**32 - 1))
    idx = rint.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return [round(float(np.percentile(diffs, 100 * alpha / 2)), 3),
            round(float(np.percentile(diffs, 100 * (1 - alpha / 2))), 3)]


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d on the within-run difference (a - b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return None
    d = a[:n] - b[:n]
    sd = d.std(ddof=1)
    if sd == 0:
        return None
    return round(float(d.mean() / sd), 3)


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon p-value without rejecting constant arms."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return None
    d = a[:n] - b[:n]
    if np.allclose(d, 0.0):
        return 1.0
    from scipy.stats import wilcoxon
    try:
        _, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(p)
    except ValueError:
        return None


def _rank_ascending(p: float) -> int:
    return int(np.round(p * 10**9))


def holms_correct(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (family-wise, monotone non-decreasing)."""
    ps = list(p_values)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    out = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, ps[i] * (m - rank))
        out[i] = max(adjusted, running)
        running = out[i]
    return [round(float(x), 4) for x in out]


def bonferroni_correct(p_values: Sequence[float]) -> List[float]:
    m = max(1, len(p_values))
    return [round(float(min(1.0, p * m)), 4) for p in p_values]


def _selftest() -> None:
    rng = random.Random(7)
    a = [rng.random() * 100 for _ in range(30)]
    b = [x + rng.gauss(0, 10) for x in a]

    lo, hi = bootstrap_mean_ci(a, n_boot=500)
    mean = float(np.mean(a))
    assert lo <= mean <= hi, (lo, mean, hi)

    cl, ch = bootstrap_ci_mean_diff(a[:20], b[:20], n_boot=500)
    assert cl is not None and cl <= ch

    d = cohens_d_paired(a, b)
    assert d is not None and abs(d) < 5

    raw = [0.001, 0.04, 0.5, 0.01]
    holm = holms_correct(raw)
    bonf = bonferroni_correct(raw)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    ranked_holm = [holm[i] for i in order]
    assert all(ranked_holm[i] <= ranked_holm[i + 1] + 1e-9 for i in range(len(ranked_holm) - 1))
    assert all(h >= r for h, r in zip(holm, raw))
    assert all(b >= r for b, r in zip(bonf, raw))
    print("statistics._selftest OK")


if __name__ == "__main__":
    _selftest()

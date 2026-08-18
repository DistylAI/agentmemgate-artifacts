"""Statistics for the memory-gating pilot (role I1).

Wilson score CIs, Mann-Whitney AUC, paired bootstrap on delta-AUC
(protocol_spec section 3.3 / 4.2), cluster bootstrap on rates (section 4.1),
exact McNemar (section 4.2), and Fleiss' kappa (section 2.4).

Every protocol-quoted reference value in protocol_spec section 4 reproduces
from these functions (tests/test_stats.py). Bootstraps are deterministic
under their seed (default 20260702 per INTERFACES.md shared conventions).

v2 extension (role W1, protocol_spec_v2 section 1.1 / freeze item 10.1/10):
cluster_bootstrap_mean generalizes cluster_bootstrap_rate to bounded-integer
paired contrasts Z_i in {-2..2}; its 0/1 path is byte-compatible with
cluster_bootstrap_rate (tests/test_stats_v2.py).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

DEFAULT_SEED = 20260702


def _validate_binary(values, name: str) -> None:
    bad = set(values) - {0, 1}
    if bad:
        raise ValueError(f"{name} must contain only 0/1 values; got extras {sorted(bad)}")


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for x successes out of n (95% at the default z).

    Clamped to [0, 1]: the exact interval always lies there, but float
    rounding can stray by ~1e-17 at the x=0 / x=n endpoints.
    """
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC via midranks; y must contain both classes."""
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: y_true must contain both classes")
    ranks = rankdata(s)  # midranks handle ties: counted as 1/2
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def mann_whitney_auc(y_true: list[int], scores: list[float]) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 * P(tie) over all pos/neg pairs."""
    _validate_binary(y_true, "y_true")
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"y_true and scores lengths differ: {y.size} vs {s.size}")
    return _auc(y, s)


def _cluster_members(clusters) -> list[np.ndarray]:
    """Item indices grouped by cluster id, in first-appearance order (deterministic)."""
    members: dict = {}
    for i, cluster_id in enumerate(clusters):
        members.setdefault(cluster_id, []).append(i)
    return [np.asarray(idx, dtype=int) for idx in members.values()]


def paired_bootstrap_delta_auc(
    y: list[int],
    s_a: list[float],
    s_b: list[float],
    n_boot: int = 10000,
    seed: int = DEFAULT_SEED,
    clusters=None,
) -> dict:
    """Percentile bootstrap CI on delta = AUC(s_a) - AUC(s_b), paired by statement.

    Resamples statement indices i.i.d. (protocol section 3.3: 10k
    statement-level resamples, percentile CI). When ``clusters`` is given,
    resamples unique clusters with replacement instead (section 4.2 M9
    by-substrate sensitivity line; natural items = singleton clusters).
    Resamples in which y is single-class (AUC undefined) are skipped;
    ``n_boot_used`` reports how many contributed. Deterministic under seed.
    """
    _validate_binary(y, "y")
    y_arr = np.asarray(y, dtype=int)
    a_arr = np.asarray(s_a, dtype=float)
    b_arr = np.asarray(s_b, dtype=float)
    if not (y_arr.size == a_arr.size == b_arr.size):
        raise ValueError("y, s_a, s_b must have equal length")
    if clusters is not None and len(clusters) != y_arr.size:
        raise ValueError("clusters must align with y")

    n = y_arr.size
    auc_a = _auc(y_arr, a_arr)
    auc_b = _auc(y_arr, b_arr)

    members = _cluster_members(clusters) if clusters is not None else None
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        if members is None:
            idx = rng.integers(0, n, n)
        else:
            draw = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[c] for c in draw])
        y_b = y_arr[idx]
        if y_b.min() == y_b.max():  # single class in resample; AUC undefined
            continue
        deltas.append(_auc(y_b, a_arr[idx]) - _auc(y_b, b_arr[idx]))

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": auc_a - auc_b,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "lo": float(lo),
        "hi": float(hi),
        "n": n,
        "n_clusters": None if members is None else len(members),
        "n_boot": n_boot,
        "n_boot_used": len(deltas),
    }


def cluster_bootstrap_rate(
    successes: list[int],
    clusters: list[str],
    n_boot: int = 10000,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Cluster-bootstrap percentile CI on a success rate (protocol section 4.1).

    Resamples the unique clusters with replacement (as many draws as there
    are clusters), pools the drawn clusters' items, and takes the percentile
    CI of the pooled rate. Deterministic under seed.
    """
    _validate_binary(successes, "successes")
    if len(successes) != len(clusters):
        raise ValueError("successes and clusters must have equal length")
    x = np.asarray(successes, dtype=float)
    members = _cluster_members(clusters)
    n_clusters = len(members)
    rng = np.random.default_rng(seed)
    rates = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in draw])
        rates[b] = x[idx].mean()
    lo, hi = np.quantile(rates, [0.025, 0.975])
    return {
        "point": float(x.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(x.size),
        "n_clusters": n_clusters,
        "n_boot": n_boot,
    }


def cluster_bootstrap_mean(
    values: list[int],
    clusters: list[str],
    n_boot: int = 10000,
    seed: int = DEFAULT_SEED,
    bounds: tuple[int, int] = (-2, 2),
) -> dict:
    """Cluster-bootstrap percentile CI on the mean of bounded-integer values
    (protocol_spec_v2 §1.1: per-injection paired interaction contrasts
    Z_i ∈ {-2..2}).

    Generalizes ``cluster_bootstrap_rate``'s 0/1 validation to integers in
    ``[bounds[0], bounds[1]]``; the resampling scheme, seed/CI conventions,
    and RNG consumption are identical, so on 0/1 inputs the result is
    byte-identical to ``cluster_bootstrap_rate`` (regression-tested against
    pilot-quoted values in tests/test_stats_v2.py). Deterministic under seed.
    """
    lo_bound, hi_bound = bounds
    bad = set(values) - set(range(lo_bound, hi_bound + 1))
    if bad:
        raise ValueError(
            f"values must be integers in [{lo_bound}, {hi_bound}]; got extras {sorted(bad)}"
        )
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    x = np.asarray(values, dtype=float)
    members = _cluster_members(clusters)
    n_clusters = len(members)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in draw])
        means[b] = x[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "point": float(x.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(x.size),
        "n_clusters": n_clusters,
        "n_boot": n_boot,
    }


def exact_mcnemar(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value on discordant-pair counts.

    Exact binomial: under H0 the smaller discordant count k = min(n01, n10)
    is Binomial(n01 + n10, 1/2); p = min(1, 2 * P(X <= k)). With zero
    discordant pairs there is no evidence against H0 and p = 1.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def mcnemar_power(n_d: int, pi: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test at n_d discordant pairs.

    ``pi`` is the true probability that a discordant pair favors the better
    method (e.g. 0.7 for a 70/30 split). Power = P(reject at level alpha)
    with X ~ Binomial(n_d, pi) discordant pairs favoring the better method.
    Reproduces protocol section 4.2: n_d=30 -> 0.59/0.80/0.94 at 70/30,
    75/25, 80/20.
    """
    return sum(
        math.comb(n_d, k) * pi**k * (1 - pi) ** (n_d - k)
        for k in range(n_d + 1)
        if exact_mcnemar(k, n_d - k) <= alpha
    )


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss' kappa, standard formula. Rows = items, columns = raters."""
    n_raters = len(ratings[0])
    if any(len(row) != n_raters for row in ratings):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    categories = sorted({label for row in ratings for label in row})
    counts = np.array(
        [[row.count(category) for category in categories] for row in ratings],
        dtype=float,
    )
    n_items = len(ratings)
    p_j = counts.sum(axis=0) / (n_items * n_raters)  # category marginals
    p_i = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = float((p_j**2).sum())
    return float((p_bar - p_e) / (1 - p_e))

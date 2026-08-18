"""v2 endpoint metrics and preregistered decision rules (owner: W5).

Implements protocol_spec_v2.md §5.1 (one-screen decision-rule table), the §1.1
H-INT estimand, and the §8 natural-first reporting rules as CODE: every rule
helper returns a uniform dict

    {"id", "rule", "point", "ci": [lo, hi] | None, "pass": bool | None, ...}

where ``pass`` is None for report-only endpoints (no preregistered go bar —
GATE-EM, CANON, NAT-CENS, MISATTR, TIME-IDX, QA-NAT) and the rule text is the
§5.1 wording. Every rate carries its denominator (INTERFACES shared
conventions). Benchmarks are never pooled here except NAT-RES (necessity,
flagged — FM-22).

H-INT (§1.1): per-injection paired contrast
    Z_i = (Y_i^EZ,CW − Y_i^EZ,CG) − (Y_i^EM,CW − Y_i^EM,CG)  ∈ {−2..2},
mean over injections; CI by cluster bootstrap via ``cluster_bootstrap_mean`` —
the Z_i-range generalization of ``stats.cluster_bootstrap_rate`` pinned by
§1.1/§10.1 item 10 (0/1 validation generalized to bounded integers). The
resampling scheme, seed discipline, and percentile CI are IDENTICAL to
``cluster_bootstrap_rate`` so 0/1 inputs reproduce it exactly
(tests/test_metrics_v2.py asserts this). Clusters: by-substrate for
LongMemEval; conversation-level (n=10) for LoCoMo — the V2-P2 primary unit
that DECIDES REP-LOCO (V2-F5); speaker-level (n=20) is sensitivity only.

Natural/injected split everywhere (V2-P9): ``stratify_by_provenance`` runs any
endpoint per stratum, and ``r_nat_assignment`` applies the §8 rule-2 headline
assignment mechanically (natural carries iff n ≥ 40 AND cluster-CI half-width
≤ 15 pp).

No LLM calls in this module.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

import stats

SEED = 20260702
N_BOOT = 10000

# The four cells of the 2×2 (§1.1): extraction family × commit policy.
CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")

# §6 generative-model anchors (pilot-anchored, V2-P6) for true_interaction().
ANCHOR_C_EM = 0.42
ANCHOR_C_EZ = 0.85
ANCHOR_Q_EM = 0.16
ANCHOR_Q_GATE = 0.05


# ---------------------------------------------------------------------------
# Cluster bootstrap on a bounded-integer mean (§1.1 / §10.1 item 10)
# ---------------------------------------------------------------------------

def _cluster_members(clusters: Sequence) -> list[np.ndarray]:
    """Item indices grouped by cluster id, first-appearance order — the exact
    grouping of stats.cluster_bootstrap_rate (deterministic)."""
    members: dict = {}
    for i, cluster_id in enumerate(clusters):
        members.setdefault(cluster_id, []).append(i)
    return [np.asarray(idx, dtype=int) for idx in members.values()]


def cluster_bootstrap_mean(
    values: Sequence[int],
    clusters: Sequence,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    bounds: tuple[int, int] = (-2, 2),
) -> dict:
    """Cluster-bootstrap percentile CI on the mean of bounded INTEGER values.

    The §1.1-pinned generalization of ``stats.cluster_bootstrap_rate``: the
    0/1 validation becomes a bounded-integer validation (H-INT's Z_i take
    values in {−2..2}; gate contrasts in {−1..1}); resampling scheme (unique
    clusters with replacement, as many draws as clusters, pooled mean),
    percentile CI, and seed discipline are identical, so 0/1 inputs with
    bounds=(0, 1) reproduce cluster_bootstrap_rate bit-for-bit.

    Delegates to ``stats.cluster_bootstrap_mean`` — the §10.1-item-10 freeze
    change (role W1) — when present; the local implementation below is the
    identical fallback so this module also runs against pre-freeze stats.py.
    """
    if hasattr(stats, "cluster_bootstrap_mean"):
        return stats.cluster_bootstrap_mean(list(values), list(clusters),
                                            n_boot=n_boot, seed=seed,
                                            bounds=bounds)
    lo_b, hi_b = bounds
    bad = [v for v in values if int(v) != v or not lo_b <= v <= hi_b]
    if bad:
        raise ValueError(
            f"values must be integers in [{lo_b}, {hi_b}]; got extras {sorted(set(bad))}")
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


# ---------------------------------------------------------------------------
# Uniform rule-result shape
# ---------------------------------------------------------------------------

def _result(rule_id: str, rule: str, point: Optional[float],
            ci: Optional[Sequence[float]], passed: Optional[bool],
            **extra: Any) -> dict:
    return {
        "id": rule_id,
        "rule": rule,
        "point": point,
        "ci": None if ci is None else [float(ci[0]), float(ci[1])],
        "pass": passed,
        **extra,
    }


def _wilson_result(rule_id: str, rule: str, x: int, n: int,
                   passed: Optional[bool], **extra: Any) -> dict:
    if n == 0:
        return _result(rule_id, rule, None, None, None, x=x, n=0, **extra)
    return _result(rule_id, rule, x / n, stats.wilson_ci(x, n), passed,
                   x=x, n=n, ci_kind="wilson", **extra)


# ---------------------------------------------------------------------------
# H-INT / REP-LOCO — the interaction estimand (§1.1, §5.1)
# ---------------------------------------------------------------------------

def z_interaction(y: dict) -> int:
    """Per-injection paired contrast Z_i ∈ {−2..2} from the four cell outcomes
    Y_i^cell ∈ {0,1} (§1.1). ``y`` keys: em_cw, em_cg, ez_cw, ez_cg."""
    for cell in CELLS:
        if y[cell] not in (0, 1):
            raise ValueError(f"Y_i^{cell} must be 0/1; got {y[cell]!r}")
    return (y["ez_cw"] - y["ez_cg"]) - (y["em_cw"] - y["em_cg"])


def _interaction_ci(rows: list[dict], cluster_field: str,
                    n_boot: int, seed: int) -> dict:
    z = [z_interaction(r["y"]) for r in rows]
    return cluster_bootstrap_mean(z, [r[cluster_field] for r in rows],
                                  n_boot=n_boot, seed=seed, bounds=(-2, 2))


def h_int(rows: list[dict], *, rule_id: str = "H-INT",
          cluster_field: str = "cluster",
          n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """H-INT (confirmatory, LME-only by rule — FM-22): Δ̂_int > 0 AND the 95%
    by-substrate cluster-bootstrap CI excludes 0.

    ``rows``: one dict per injection with ``y`` (four 0/1 cell outcomes at the
    manifest-designated probe point), a cluster id under ``cluster_field``,
    and ``provenance`` in {"injected", "natural"}. The natural/injected strata
    are reported alongside (V2-P9); the §8 rule-5 consistency check is
    ``natural_consistency_check`` on the two strata.
    """
    boot = _interaction_ci(rows, cluster_field, n_boot, seed)
    passed = boot["point"] > 0 and (boot["lo"] > 0 or boot["hi"] < 0)
    strata = {}
    for prov in ("injected", "natural"):
        sub = [r for r in rows if r.get("provenance") == prov]
        if sub:
            b = _interaction_ci(sub, cluster_field, n_boot, seed)
            strata[prov] = {"point": b["point"], "ci": [b["lo"], b["hi"]],
                            "n": b["n"], "n_clusters": b["n_clusters"]}
        else:
            strata[prov] = {"point": None, "ci": None, "n": 0, "n_clusters": 0}
    return _result(
        rule_id,
        "delta_int_hat > 0 AND 95% cluster-bootstrap CI excludes 0 (§5.1 H-INT)",
        boot["point"], (boot["lo"], boot["hi"]), passed,
        n=boot["n"], n_clusters=boot["n_clusters"], ci_kind="cluster_bootstrap",
        strata_V2P9=strata,
    )


def rep_loco(rows: list[dict], *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """REP-LOCO (secondary, replication): same estimand on LoCoMo; 'replicated'
    is decided on the CONVERSATION-level (n=10, V2-P2 primary unit) cluster CI
    (V2-F5); the speaker-level (n=20) CI is reported as sensitivity ONLY.

    ``rows`` carry ``conversation`` and ``speaker`` cluster ids (+ ``y``,
    ``provenance``)."""
    primary = h_int(rows, rule_id="REP-LOCO", cluster_field="conversation",
                    n_boot=n_boot, seed=seed)
    sens = _interaction_ci(rows, "speaker", n_boot, seed)
    primary["rule"] = ("replicated = delta_int_hat > 0 AND conversation-level "
                       "(n=10, V2-P2 primary) cluster CI excludes 0 (V2-F5); "
                       "speaker-level CI is sensitivity only")
    primary["speaker_sensitivity_only"] = {
        "point": sens["point"], "ci": [sens["lo"], sens["hi"]],
        "n_clusters": sens["n_clusters"],
        "note": "never decides the verdict (V2-F5)",
    }
    return primary


def true_interaction(q_ez: float, *, c_ez: float = ANCHOR_C_EZ,
                     c_em: float = ANCHOR_C_EM, q_em: float = ANCHOR_Q_EM,
                     q_gate: float = ANCHOR_Q_GATE) -> float:
    """§6.1 generative-model true interaction:
    Δ_int = (c_EZ·q_EZ − c_EZ·q_gate) − (c_EM·q_EM − c_EM·q_gate).
    Exact null at q_EZ = q_gate + c_EM(q_EM − q_gate)/c_EZ ≈ 0.1044 (appendix b)."""
    return (c_ez * q_ez - c_ez * q_gate) - (c_em * q_em - c_em * q_gate)


# ---------------------------------------------------------------------------
# Within-family gate contrasts and cell rules (§5.1)
# ---------------------------------------------------------------------------

def gate_contrast(rows: list[dict], *, family: str,
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """GATE-EZ / GATE-EM: paired ITT contrast CW − CG within one extraction
    family, per benchmark. D_i = Y_i^CW − Y_i^CG ∈ {−1..1}.

    GATE-EZ: cluster-bootstrap CI lower bound > 0 is the go bar.
    GATE-EM: reported with CI, NO go bar (§6: power 0.4–0.8 only; pilot showed
    the effect is extractor-limited) → pass is None.

    ``rows``: {"cluster", "y_cw", "y_cg", "provenance"} per injection.
    """
    fam = family.lower()
    if fam not in ("ez", "em"):
        raise ValueError(f"family must be 'ez' or 'em'; got {family!r}")
    d = []
    for r in rows:
        if r["y_cw"] not in (0, 1) or r["y_cg"] not in (0, 1):
            raise ValueError("y_cw / y_cg must be 0/1")
        d.append(r["y_cw"] - r["y_cg"])
    boot = cluster_bootstrap_mean(d, [r["cluster"] for r in rows],
                                  n_boot=n_boot, seed=seed, bounds=(-1, 1))
    if fam == "ez":
        rule_id, passed = "GATE-EZ", boot["lo"] > 0
        rule = "paired ITT contrast CW−CG within E-Z: cluster-bootstrap CI lower bound > 0"
    else:
        rule_id, passed = "GATE-EM", None
        rule = ("paired ITT contrast CW−CG within E-M: reported with CI; "
                "NO go bar (§5.1 GATE-EM)")
    strata = {}
    for prov in ("injected", "natural"):
        sub = [i for i, r in enumerate(rows) if r.get("provenance") == prov]
        if sub:
            b = cluster_bootstrap_mean([d[i] for i in sub],
                                       [rows[i]["cluster"] for i in sub],
                                       n_boot=n_boot, seed=seed, bounds=(-1, 1))
            strata[prov] = {"point": b["point"], "ci": [b["lo"], b["hi"]], "n": b["n"]}
        else:
            strata[prov] = {"point": None, "ci": None, "n": 0}
    return _result(rule_id, rule, boot["point"], (boot["lo"], boot["hi"]), passed,
                   n=boot["n"], n_clusters=boot["n_clusters"],
                   ci_kind="cluster_bootstrap", strata_V2P9=strata)


def cell_ezcw_rule(successes: Sequence[int], clusters: Sequence,
                   *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """CELL-EZCW: E-Z/C-W ITT speculation contamination, LME —
    point ≥ 10% AND cluster lower ≥ 5% (the v1 Test-2 measurability bar)."""
    boot = stats.cluster_bootstrap_rate(list(successes), list(clusters),
                                        n_boot=n_boot, seed=seed)
    passed = boot["point"] >= 0.10 and boot["lo"] >= 0.05
    return _result("CELL-EZCW",
                   "point >= 0.10 AND cluster-bootstrap lower >= 0.05 (§5.1 CELL-EZCW)",
                   boot["point"], (boot["lo"], boot["hi"]), passed,
                   x=int(sum(successes)), n=boot["n"],
                   n_clusters=boot["n_clusters"], ci_kind="cluster_bootstrap")


# ---------------------------------------------------------------------------
# Wilson-rule endpoints (§5.1)
# ---------------------------------------------------------------------------

def manip_rule(x: int, n: int) -> dict:
    """MANIP (validity gate, per benchmark): E-Z speculation canonicalization —
    point ≥ 0.80 AND Wilson lower ≥ 0.70. Fail ⇒ the interaction is
    uninterpretable for that benchmark (C1-iv, FM-18), cells reported
    descriptively."""
    lo, _ = stats.wilson_ci(x, n) if n else (None, None)
    passed = n > 0 and x / n >= 0.80 and lo >= 0.70
    return _wilson_result(
        "MANIP", "point >= 0.80 AND Wilson lower >= 0.70 (§5.1 MANIP; FM-18 on fail)",
        x, n, passed,
        on_fail="interaction uninterpretable for this benchmark (FM-18); "
                "cells reported descriptively")


def canon_report(x: int, n: int, *, arm: str, item_class: str,
                 benchmark: str) -> dict:
    """CANON (secondary, C1-i): canonicalization rate per class per extraction
    arm per benchmark — the 'fifth policy'. Reported with CIs; no bar
    (E-M expected ≈ 0.42, pilot anchor)."""
    return _wilson_result(
        "CANON", "reported with CIs; no go bar (§5.1 CANON; pilot anchor E-M ≈ 0.42)",
        x, n, None, arm=arm, item_class=item_class, benchmark=benchmark)


def corr_recall_rule(x: int, n: int) -> dict:
    """CORR-RECALL: correction commit recall, gated cells, per benchmark —
    point ≥ 0.9 AND Wilson lower ≥ 0.8 (tolerates 12 misses at n=120; first
    fails at 13 — §5.1, computed with stats.wilson_ci)."""
    lo, _ = stats.wilson_ci(x, n) if n else (None, None)
    passed = n > 0 and x / n >= 0.9 and lo >= 0.8
    return _wilson_result(
        "CORR-RECALL", "point >= 0.9 AND Wilson lower >= 0.8 (§5.1 CORR-RECALL)",
        x, n, passed)


def corr_residue_rule(rows: list[dict], *, n_boot: int = N_BOOT,
                      seed: int = SEED) -> dict:
    """CORR-RESIDUE: paired (gated − ungated) stale-old difference —
    cluster-bootstrap UPPER bound ≤ +0.05 ('the gate must not be worse').
    ``rows``: {"cluster", "stale_gated" 0/1, "stale_ungated" 0/1} per correction."""
    d = [r["stale_gated"] - r["stale_ungated"] for r in rows]
    boot = cluster_bootstrap_mean(d, [r["cluster"] for r in rows],
                                  n_boot=n_boot, seed=seed, bounds=(-1, 1))
    passed = boot["hi"] <= 0.05
    return _result("CORR-RESIDUE",
                   "paired (gated − ungated) stale-old difference: "
                   "cluster-bootstrap upper <= +0.05 (§5.1 CORR-RESIDUE)",
                   boot["point"], (boot["lo"], boot["hi"]), passed,
                   n=boot["n"], n_clusters=boot["n_clusters"],
                   ci_kind="cluster_bootstrap")


def spec_lic_rule(x: int, n: int) -> dict:
    """SPEC-LIC: classifier speculation licensing per extraction arm —
    Wilson upper ≤ 10% (≤ 5% aspirational), carried from v1."""
    _, hi = stats.wilson_ci(x, n) if n else (None, None)
    passed = n > 0 and hi <= 0.10
    out = _wilson_result(
        "SPEC-LIC", "Wilson upper <= 0.10 (<= 0.05 aspirational) (§5.1 SPEC-LIC)",
        x, n, passed)
    out["pass_5pct_aspirational"] = n > 0 and hi <= 0.05
    return out


# ---------------------------------------------------------------------------
# Natural-resolution endpoints (§5.1 NAT-RES / NAT-CENS; FM-19)
# ---------------------------------------------------------------------------

NAT_RES_DENOMINATOR = 35  # frozen at freeze = the 35-pair artifact (V2-F3)
NAT_RES_VOID_BELOW = 25   # FM-19: voided -> report-only if < 25 pairs survive


def natres_rule(x: int, n: int = NAT_RES_DENOMINATOR,
                *, survived: Optional[int] = None) -> dict:
    """NAT-RES: gated promotion on the resolution turn (sensitivity window
    +2 turns) — point ≥ 0.70 AND Wilson lower ≥ 0.55 (26/35 = 0.743 lower
    0.579 passes; 25/35 fails on the lower bound). Denominator frozen at 35;
    the 42ec0761/c6853660 arcs are a supplementary stratum, NEVER in the
    denominator. Voided → report-only if < 25 pairs survive labeling (FM-19).
    """
    voided = survived is not None and survived < NAT_RES_VOID_BELOW
    lo, _ = stats.wilson_ci(x, n) if n else (None, None)
    passed: Optional[bool]
    if voided or n == 0:
        passed = None
    else:
        passed = x / n >= 0.70 and lo >= 0.55
    out = _wilson_result(
        "NAT-RES",
        "point >= 0.70 AND Wilson lower >= 0.55 on the frozen 35-pair "
        "denominator (§5.1 NAT-RES); voided to report-only below 25 survivors (FM-19)",
        x, n, passed,
        denominator_frozen=NAT_RES_DENOMINATOR, survived=survived,
        status="voided_report_only_FM19" if voided else "scored",
        pooled_across_benchmarks="by necessity, flagged (FM-22 exception); strata reported")
    return out


def natcens_report(x: int, n: int = 4) -> dict:
    """NAT-CENS: false-promote rate on censored natural arcs — denominator
    frozen at n=4 (LoCoMo censored arcs) unless an LME hedge list was frozen
    by changelog before §10.2 step 3 (MINOR-5). Report-only, no bar
    (0/4 → Wilson upper 0.490)."""
    return _wilson_result(
        "NAT-CENS", "false-promote rate with Wilson CI; report-only, no bar "
                    "(§5.1 NAT-CENS — n too small)",
        x, n, None)


# ---------------------------------------------------------------------------
# SYN-RES (v1 Test-3 rules verbatim), TTC (v1 §3.2 verbatim)
# ---------------------------------------------------------------------------

def synres_promotion_rule(successes: Sequence[int], clusters: Sequence,
                          *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """SYN-RES promotion (per benchmark): point ≥ 0.8 AND cluster lower ≥ 0.65
    (v1 verbatim; clusters = cases)."""
    boot = stats.cluster_bootstrap_rate(list(successes), list(clusters),
                                        n_boot=n_boot, seed=seed)
    passed = boot["point"] >= 0.8 and boot["lo"] >= 0.65
    return _result("SYN-RES/promotion",
                   "point >= 0.8 AND cluster-bootstrap-by-case lower >= 0.65 (v1 verbatim)",
                   boot["point"], (boot["lo"], boot["hi"]), passed,
                   x=int(sum(successes)), n=boot["n"],
                   n_clusters=boot["n_clusters"], ci_kind="cluster_bootstrap")


def synres_false_fire_rule(successes: Sequence[int], clusters: Sequence,
                           *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """SYN-RES false-fire (per benchmark, n=100 distractor probes):
    point ≤ 5% AND cluster upper ≤ 7.5% (v1 verbatim)."""
    boot = stats.cluster_bootstrap_rate(list(successes), list(clusters),
                                        n_boot=n_boot, seed=seed)
    passed = boot["point"] <= 0.05 and boot["hi"] <= 0.075
    return _result("SYN-RES/false_fire",
                   "point <= 0.05 AND cluster-bootstrap-by-case upper <= 0.075 (v1 verbatim)",
                   boot["point"], (boot["lo"], boot["hi"]), passed,
                   x=int(sum(successes)), n=boot["n"],
                   n_clusters=boot["n_clusters"], ci_kind="cluster_bootstrap")


def ttc_rule(excess_turns: Sequence[float], censored_x: int, censored_n: int) -> dict:
    """TTC (staleness, all genuine updates; v1 §3.2 verbatim): gated median TTC
    excess = 0 turns AND staged censoring ≤ 1 − 0.8 = 0.2."""
    med = float(np.median(excess_turns)) if len(excess_turns) else None
    cens_frac = censored_x / censored_n if censored_n else None
    passed = (med is not None and med == 0
              and cens_frac is not None and cens_frac <= 0.2)
    return _result("TTC",
                   "gated median TTC excess = 0 turns AND staged censoring <= 0.2 "
                   "(v1 §3.2 verbatim)",
                   med, None, passed,
                   n=len(excess_turns),
                   censoring={"x": censored_x, "n": censored_n, "fraction": cens_frac,
                              "bound": 0.2})


# ---------------------------------------------------------------------------
# Descriptive probe endpoints (§4 / §5.1 — no preregistered go bars)
# ---------------------------------------------------------------------------

def misattr_report(x: int, n: int, *, stratum: str, cell: str) -> dict:
    """MISATTR (descriptive per cell): key-ABSENCE rate over the frozen seeded
    cat-5 sample; primary stratum = the 75 speaker_swap rows,
    focal_fact_distortion (25) exploratory (V2-F4). NO preregistered go bar;
    the §5.1 feasibility reading (point ≥ 0.9 AND lower ≥ 0.8, tolerates 7
    misses at n=75) is reported as a reading, never a verdict."""
    if stratum not in ("speaker_swap", "focal_fact_distortion"):
        raise ValueError(f"unknown MISATTR stratum {stratum!r}")
    out = _wilson_result(
        "MISATTR", "reported per cell; descriptive — no go bar (§5.1 MISATTR)",
        x, n, None, stratum=stratum, cell=cell,
        stratum_status="primary" if stratum == "speaker_swap" else "exploratory")
    if n:
        lo, _ = stats.wilson_ci(x, n)
        out["feasibility_reading_not_a_bar"] = x / n >= 0.9 and lo >= 0.8
    return out


def time_idx_report(x: int, n: int, *, cell: str) -> dict:
    """TIME-IDX (secondary, descriptive): P-TIME accuracy per cell over the
    n=19 scoreable items (V2-F4). No bar is preregistered (17/19 = 0.895,
    CI half-width ≈ 14 pp); R-NAT can never assign it a headline (n < 40)."""
    return _wilson_result(
        "TIME-IDX", "reported per cell (history diagnostic); no bar (§5.1 TIME-IDX)",
        x, n, None, cell=cell, r_nat_eligible=False)


def qa_nat_report(x: int, n: int, *, cell: str) -> dict:
    """QA-NAT (secondary): P-NATQA accuracy per cell, natural stratum
    (n = frozen coverage count, FM-21). Reported; no bar. For cluster CIs
    pass per-item successes to qa_nat_report_items."""
    return _wilson_result(
        "QA-NAT", "reported per cell, natural stratum (§5.1 QA-NAT); no bar",
        x, n, None, cell=cell)


def qa_nat_report_items(successes: Sequence[int], clusters: Sequence, *, cell: str,
                        n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """QA-NAT with per-item successes and cluster ids (conversation-level)."""
    boot = stats.cluster_bootstrap_rate(list(successes), list(clusters),
                                        n_boot=n_boot, seed=seed)
    return _result("QA-NAT",
                   "reported per cell, natural stratum (§5.1 QA-NAT); no bar",
                   boot["point"], (boot["lo"], boot["hi"]), None,
                   x=int(sum(successes)), n=boot["n"],
                   n_clusters=boot["n_clusters"], ci_kind="cluster_bootstrap",
                   cell=cell)


# ---------------------------------------------------------------------------
# C1 decomposition + natural-first reporting machinery (§1.4, §8; V2-P9)
# ---------------------------------------------------------------------------

def itt_decomposition(canon_x: int, canon_n: int,
                      cond_x: int, cond_n: int) -> dict:
    """C1-iii: every contamination table prints ITT = canonicalization ×
    conditional contamination, with BOTH bases (conditional rates are never
    compared across extraction arms without them)."""
    canon = canon_x / canon_n if canon_n else None
    cond = cond_x / cond_n if cond_n else None
    itt = canon * cond if canon is not None and cond is not None else None
    return {
        "canonicalization": {"x": canon_x, "n": canon_n, "point": canon,
                             "ci": list(stats.wilson_ci(canon_x, canon_n)) if canon_n else None},
        "conditional_contamination": {"x": cond_x, "n": cond_n, "point": cond,
                                      "ci": list(stats.wilson_ci(cond_x, cond_n)) if cond_n else None},
        "itt": itt,
        "identity": "ITT = canonicalization × conditional contamination (C1-iii)",
    }


def r_nat_assignment(natural_n: int, natural_ci: Optional[Sequence[float]]) -> dict:
    """§8 rule 2 (R-NAT), mechanical: the natural stratum carries the headline
    iff natural n ≥ 40 AND its cluster-bootstrap CI half-width ≤ 15 pp;
    otherwise the injected stratum carries it (natural printed beside)."""
    half_width = (None if natural_ci is None
                  else (float(natural_ci[1]) - float(natural_ci[0])) / 2)
    natural_carries = (natural_n >= 40 and half_width is not None
                       and half_width <= 0.15)
    return {
        "rule": "natural carries iff n >= 40 AND CI half-width <= 0.15 (§8 rule 2)",
        "natural_n": natural_n,
        "ci_half_width": half_width,
        "carrier": "natural" if natural_carries else "injected",
    }


def stratify_by_provenance(rows: list[dict],
                           endpoint_fn: Callable[[list[dict]], dict]) -> dict:
    """Run an endpoint on all rows and per provenance stratum (V2-P9 / §8
    rule 1: every headline table reports natural and injected separately),
    then apply R-NAT to name the carrier."""
    out = {"all": endpoint_fn(rows) if rows else None}
    for prov in ("natural", "injected"):
        sub = [r for r in rows if r.get("provenance") == prov]
        out[prov] = endpoint_fn(sub) if sub else None
    nat = out["natural"]
    out["r_nat"] = r_nat_assignment(
        nat["n"] if nat else 0, nat["ci"] if nat else None)
    return out


def natural_consistency_check(injected_result: dict, natural_result: dict) -> dict:
    """§8 rule 5: Δ̂_int on the natural speculation stratum printed next to
    H-INT. If its sign OPPOSES H-INT and its cluster CI EXCLUDES the injected
    point estimate, the headline claim is downgraded to injected-scope."""
    ip, np_ = injected_result.get("point"), natural_result.get("point")
    nci = natural_result.get("ci")
    if ip is None or np_ is None or not nci:
        return {"applicable": False, "downgrade_to_injected_scope": False,
                "note": "one stratum empty; check not applicable"}
    sign_opposes = (ip > 0 > np_) or (ip < 0 < np_)
    ci_excludes_injected_point = not (nci[0] <= ip <= nci[1])
    downgrade = sign_opposes and ci_excludes_injected_point
    return {
        "applicable": True,
        "sign_opposes": sign_opposes,
        "natural_ci_excludes_injected_point": ci_excludes_injected_point,
        "downgrade_to_injected_scope": downgrade,
        "rule": "§8 rule 5 (FM-1 discipline)",
    }

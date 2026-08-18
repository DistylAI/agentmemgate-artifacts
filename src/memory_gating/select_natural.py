"""Natural-item greedy min-replay-cost selection (owner: W5).

Executes protocol_spec_v2.md §2.3 (V2-P1; ordering and executability fixed by
V2-F1): SELECT-then-LABEL. Selection consumes ONLY frozen pool fields (`kind`
as the PROVISIONAL class tag, `quality`, `question_id`, anchors) — no labels
exist at selection time. Deterministic: no RNG is consumed (seed 20260702
recorded, not consumed) and no LLM is called.

Cost model (§2.3, executable rule):
  - cost(q) = number of USER turns in q's s_cleaned haystack, charged once per
    extractor family lacking a cached replay (default families_charged = 2,
    the preregistered worst case; pilot-cache hits reduce a family's charge to
    0 at run time — the §7 row-10 cap applies to REALIZED calls).
  - LoCoMo-source rows cost 0 (their conversations are replayed in full as the
    natural pass). The 70 NAT-RES pair rows cost 0 (STORELESS, §4/V2-F3).
    NO other cost-0 carve-out exists (V2-F1 — in particular, items on v2 LME
    substrates are NOT cost-0).

Algorithm (ascending cost, fill floors, stop): iterate distinct source
questions in ascending (cost, question_id) order; buy a question iff it
contains ≥ 1 row of a class whose floor is unmet AND the running replay total
stays ≤ 10,000 (§7 row 10, FM-20); every row on a bought question is selected
(marginal cost 0); stop when all floors are met or no affordable question
remains. Shortfall after the cap ⇒ injected top-up per v1 §4.2 (logged per
class). Post-labeling floor breaks extend by the SAME rule (`greedy_select`
with `initial_counts` + `exclude_questions` + `initial_spend`), logged in the
changelog before use (§2.3).

Preregistered worst-case fill from the frozen pool at ×2 charge (§2.3, the
correctness oracle for tests/test_select.py): cost-0 rows supply
48 spec / 89 comp / 1 corr / 2 other / 3 aband; the greedy buys 22 questions
for 9,714 of the 10,000-call cap and ends at spec 64 / comp 96 / corr 7 /
other 7 ⇒ injected top-up 16 spec / 33 corr / 43 other.

CLI (from pilot/):  .venv/bin/python select_natural.py
  -> ../results/v2/natural_selection.json  (per-item replay-cost accounting;
     asserts the ≤10k-call cap)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from paths import RESEARCH_DIR, RESULTS_DIR

POOL_PATH = RESEARCH_DIR / "natural_candidates_full.jsonl"
SELECTION_PATH = RESULTS_DIR / "v2" / "natural_selection.json"

SEED = 20260702  # recorded, NOT consumed — the rule is fully deterministic
REPLAY_CAP = 10_000          # §7 row 10 hard cap (FM-20)
FAMILIES_CHARGED = 2         # worst case: both extractor families uncached

# §2.3 natural class floors (turn-level, v2). Abandonment/invalidation is
# report-only and NEVER triggers a purchase.
FLOORS_V2 = {
    "speculative": 80,
    "completed_event": 70,
    "explicit_correction": 40,
    "other": 50,
}
HEDGED_RECALL_MIN = 15  # mandatory stratum inside `other` (injected top-up fills it)

# Pool `kind` (provisional tag, never gold — P8) -> provisional class.
KIND_TO_CLASS = {
    "speculation": "speculative",
    "completed_event": "completed_event",
    "correction": "explicit_correction",
    "other_hedged_recall": "other",
    "abandonment_invalidation": "abandonment",  # report-only
}
HEDGED_KINDS = {"other_hedged_recall"}

# v1 per-arm Test-1 floors the injected top-up ultimately restores (§2.3).
V1_TOPUP_FLOORS = {"explicit_correction": 60, "speculative": 80,
                   "completed_event": 70, "other": 50}


# ---------------------------------------------------------------------------
# Pool loading and cost computation
# ---------------------------------------------------------------------------

def load_pool(path: Path = POOL_PATH) -> list[dict]:
    """The merged canonical natural pool (§10.1 item 7): one row per
    (session-hash, turn); pair rows authoritative. Validates the kind map."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        if r["kind"] not in KIND_TO_CLASS:
            raise ValueError(f"unknown pool kind {r['kind']!r} "
                             f"(question {r['question_id']})")
    return rows


def provisional_class(row: dict) -> str:
    return KIND_TO_CLASS[row["kind"]]


def is_cost0(row: dict) -> bool:
    """§2.3: LoCoMo-source rows and the 70 NAT-RES pair rows cost 0.
    No other carve-out exists (V2-F1)."""
    return row["source"] == "locomo" or bool(row.get("pair_id"))


def cost_basis(row: dict) -> str:
    if row.get("pair_id"):
        return "cost0_pair_storeless"
    if row["source"] == "locomo":
        return "cost0_locomo_instream"
    return "replay_question"


def question_user_turn_costs(pool: list[dict]) -> dict[str, int]:
    """cost(q) = user turns in q's s_cleaned haystack, for every distinct
    positive-cost source question. Reuses the pipeline's s_cleaned loader
    (test1_eval.s_cleaned_substrate — the v1 §2.1-B3 information set)."""
    import eval_classifier as test1_eval  # runner module; lazy so pure logic stays importable

    costs: dict[str, int] = {}
    for row in pool:
        if is_cost0(row):
            continue
        qid = row["question_id"]
        if qid in costs:
            continue
        substrate = test1_eval.s_cleaned_substrate(qid)
        costs[qid] = sum(1 for session in substrate.sessions
                         for turn in session if turn["role"] == "user")
    return costs


# ---------------------------------------------------------------------------
# The executable greedy rule (§2.3)
# ---------------------------------------------------------------------------

def _count_classes(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        cls = provisional_class(r)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def greedy_select(
    pool: list[dict],
    costs: dict[str, int],
    *,
    floors: dict[str, int] = FLOORS_V2,
    cap: int = REPLAY_CAP,
    families_charged: int = FAMILIES_CHARGED,
    initial_counts: Optional[dict[str, int]] = None,
    exclude_questions: frozenset[str] = frozenset(),
    initial_spend: int = 0,
) -> dict:
    """Run the §2.3 greedy. Deterministic.

    Primary round: ``initial_counts=None`` seeds the class counts from the
    cost-0 rows. Extension round (post-labeling floor re-check, §2.3): pass
    the post-labeling class counts as ``initial_counts``, the already-bought
    questions as ``exclude_questions``, and the realized prior spend as
    ``initial_spend`` — the same rule continues under the same cap.
    """
    cost0_rows = [r for r in pool if is_cost0(r)]
    positive_rows = [r for r in pool if not is_cost0(r)]

    by_question: dict[str, list[dict]] = {}
    for r in positive_rows:
        by_question.setdefault(r["question_id"], []).append(r)
    missing = sorted(set(by_question) - set(costs))
    if missing:
        raise ValueError(f"no replay cost for pool questions: {missing[:5]} ...")

    counts = dict(initial_counts) if initial_counts is not None \
        else _count_classes(cost0_rows)
    hedged = (0 if initial_counts is not None
              else sum(1 for r in cost0_rows if r["kind"] in HEDGED_KINDS))

    def unmet() -> list[str]:
        return sorted(c for c, f in floors.items() if counts.get(c, 0) < f)

    order = sorted(by_question, key=lambda q: (costs[q], q))
    total = initial_spend
    bought: list[dict] = []
    for qid in order:
        if not unmet():
            break  # all floors met
        if qid in exclude_questions:
            continue
        rows = by_question[qid]
        # buy iff >= 1 row of a class whose floor is unmet (abandonment and
        # already-met classes never trigger a purchase)
        triggers = sorted({provisional_class(r) for r in rows}
                          & set(unmet()))
        if not triggers:
            continue
        charge = families_charged * costs[qid]
        if total + charge > cap:
            continue  # unaffordable; keep scanning (§2.3: "no affordable question remains")
        total += charge
        for r in rows:
            cls = provisional_class(r)
            counts[cls] = counts.get(cls, 0) + 1
            if r["kind"] in HEDGED_KINDS:
                hedged += 1
        bought.append({
            "question_id": qid,
            "cost_user_turns": costs[qid],
            "families_charged": families_charged,
            "charge": charge,
            "running_total": total,
            "unmet_classes_at_purchase": triggers,
            "rows_by_class": _count_classes(rows),
            "n_rows": len(rows),
        })

    bought_qids = {b["question_id"] for b in bought}
    selected = ([] if initial_counts is not None else list(cost0_rows)) + [
        r for r in positive_rows if r["question_id"] in bought_qids]
    shortfalls = {cls: max(0, floor - counts.get(cls, 0))
                  for cls, floor in floors.items()}
    assert total <= cap, f"replay-cost cap violated: {total} > {cap} (FM-20)"
    return {
        "counts": counts,
        "hedged_recall_count": hedged,
        "bought": bought,
        "selected_rows": selected,
        "total_charge": total,
        "cap": cap,
        "families_charged": families_charged,
        "shortfalls_injected_topup": shortfalls,
        "floors": dict(floors),
        "floors_met": not any(shortfalls.values()),
        "n_questions_scanned": len(order),
        "excluded_questions": sorted(exclude_questions),
    }


# ---------------------------------------------------------------------------
# Report: per-item replay-cost accounting (§2.3 / §7 row 10)
# ---------------------------------------------------------------------------

def _item_row(row: dict, bought_qids: set[str]) -> dict:
    basis = cost_basis(row)
    return {
        "question_id": row["question_id"],
        "session_idx": row["session_idx"],
        "turn_idx": row["turn_idx"],
        "source": row["source"],
        "kind": row["kind"],
        "provisional_class": provisional_class(row),
        "quality": row.get("quality"),
        "pair_id": row.get("pair_id"),
        "pair_role": row.get("pair_role"),
        "cost_basis": basis,
        # per-item marginal cost is 0 everywhere: cost-0 rows by rule; rows on
        # a bought question because the question charge is paid once (§2.3)
        "marginal_replay_calls": 0,
        "charged_via_question": (row["question_id"]
                                 if basis == "replay_question"
                                 and row["question_id"] in bought_qids else None),
    }


def build_report(pool: list[dict], costs: dict[str, int],
                 selection: dict) -> dict:
    cost0_rows = [r for r in pool if is_cost0(r)]
    bought_qids = {b["question_id"] for b in selection["bought"]}
    cost_values = sorted(costs.values())
    n = len(cost_values)
    median = (cost_values[n // 2] if n % 2 else
              (cost_values[n // 2 - 1] + cost_values[n // 2]) / 2)
    abandonment_rows = [r for r in selection["selected_rows"]
                        if provisional_class(r) == "abandonment"]
    report = {
        "generated_by": "select_natural.py (protocol_spec_v2.md §2.3, V2-F1)",
        "seed": {"value": SEED, "consumed": False,
                 "note": "the greedy is deterministic; no RNG is consumed"},
        "ordering_V2F1": "selection precedes ALL labeling; inputs are frozen "
                         "pool fields only (kind is a provisional tag, never gold)",
        "pool": {"path": str(POOL_PATH), "n_rows": len(pool),
                 "n_cost0_rows": len(cost0_rows),
                 "cost0_class_counts": _count_classes(cost0_rows),
                 "n_positive_cost_questions": len(costs)},
        "cost_model": {
            "definition": "cost(q) = user turns in q's s_cleaned haystack, "
                          "charged once per extractor family lacking a cached "
                          "replay; LoCoMo rows and the 70 pair rows cost 0; "
                          "no other carve-out (V2-F1)",
            "families_charged": selection["families_charged"],
            "distribution": {"min": cost_values[0], "median": median,
                             "mean": round(sum(cost_values) / n, 1),
                             "max": cost_values[-1]},
        },
        "floors": {"v2_natural": selection["floors"],
                   "hedged_recall_stratum_min": HEDGED_RECALL_MIN,
                   "v1_per_arm_topup": V1_TOPUP_FLOORS,
                   "abandonment": "report-only; never triggers a purchase (§2.3)"},
        "replay_budget": {
            "cap": selection["cap"],
            "total_charge": selection["total_charge"],
            "within_cap": selection["total_charge"] <= selection["cap"],
            "note": "worst-case charge (×{f} families); pilot-cache hits only "
                    "lower realized calls (§7 row 10 applies to realized "
                    "calls)".format(f=selection["families_charged"]),
        },
        "bought_questions": selection["bought"],
        "n_bought_questions": len(selection["bought"]),
        "final_class_counts": selection["counts"],
        "hedged_recall_stratum": {
            "n": selection["hedged_recall_count"],
            "min_required": HEDGED_RECALL_MIN,
            "met": selection["hedged_recall_count"] >= HEDGED_RECALL_MIN,
            "note": "shortfall filled by v1 §4.2 hedged injections (§2.3)",
        },
        "shortfalls_injected_topup": selection["shortfalls_injected_topup"],
        "floors_met_without_topup": selection["floors_met"],
        "abandonment_report_only": {
            "n_realized": len(abandonment_rows),
            "definition": "all cost-0 abandonment rows plus those landing on "
                          "bought questions (§2.3)",
        },
        "n_selected_items": len(selection["selected_rows"]),
        "selected_items": [_item_row(r, bought_qids)
                           for r in selection["selected_rows"]],
        "post_labeling_recheck": "labeling may reclassify; broken floors "
                                 "extend by the same rule (greedy_select with "
                                 "initial_counts/exclude_questions/initial_spend), "
                                 "logged in the changelog before use (§2.3)",
    }
    assert report["replay_budget"]["within_cap"], \
        f"<=10k-call assertion failed: {selection['total_charge']} > {selection['cap']}"
    return report


def run_selection(pool_path: Path = POOL_PATH,
                  families_charged: int = FAMILIES_CHARGED,
                  cap: int = REPLAY_CAP) -> dict:
    pool = load_pool(pool_path)
    costs = question_user_turn_costs(pool)
    selection = greedy_select(pool, costs, cap=cap,
                              families_charged=families_charged)
    return build_report(pool, costs, selection)


def main() -> None:
    report = run_selection()
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {SELECTION_PATH}")
    print(f"bought {report['n_bought_questions']} questions for "
          f"{report['replay_budget']['total_charge']}/{report['replay_budget']['cap']} "
          f"worst-case replay calls (within cap: {report['replay_budget']['within_cap']})")
    print(f"final class counts: {report['final_class_counts']}")
    print(f"injected top-up per class: {report['shortfalls_injected_topup']}")


if __name__ == "__main__":
    main()

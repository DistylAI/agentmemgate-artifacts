"""Extractor call wrapper — protocol_spec.md §3.1, extractor call contract B2.

The extractor is invoked ONCE per USER turn. Its input is all prior turns of the
substrate in chronological order (both roles, with session date markers) plus the
current user turn, clearly marked; emitted tuples are attributed to the current
turn and `confidence` is scored as of that turn. No future turns ever enter a
call. This is the single extraction pathway for Tests 1 and 2.

The system prompt is the FROZEN `prompts/extractor.txt`, used verbatim (freeze
table: prompts/README.md; it is never edited here).

`prior_turns` contract (this module's input convention, consumed by store.py):
each element is `{"role": str, "content": str, "date": str}` where `date` is the
host session's date string. A session date marker is rendered whenever the date
changes between consecutive turns.

CLI (R2-B-2 producer):
  .venv/bin/python extract.py produce-tuples
writes `../results/test1/extractor_tuples.jsonl` (row schema documented in
`label.load_extractor_tuples`) via `store.extract_substrate_tuples` — the same
extraction pathway as the Test 2 replay, so the llm.py cache guarantees a
single extraction pathway (protocol §3.1 B2).
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from llm import call_llm

from paths import PROMPTS_DIR, RESULTS_DIR

EXTRACTOR_PROMPT = (PROMPTS_DIR / "extractor.txt").read_text()

_CURRENT_TURN_MARKER = (
    "=== CURRENT USER MESSAGE (extract fact tuples from this message only; "
    "everything above is context) ==="
)


@dataclass(frozen=True)
class Tuple:
    """One extracted statement tuple (protocol_spec §1.0 unit of classification)."""

    entity: str
    attribute: str
    value: str
    statement_span: str
    confidence: int  # verbalized numeric 0-100, protocol_spec §3.3 (s_conf baseline)


def render_extraction_prompt(
    prior_turns: list[dict], current_turn: str, current_date: str | None
) -> str:
    """Render the B2 user prompt: prior turns (both roles, chronological, session
    date markers) + the current turn clearly marked."""
    lines: list[str] = []
    last_date: str | None = None
    if prior_turns:
        lines.append("=== PRIOR CONVERSATION (context only) ===")
        for turn in prior_turns:
            if turn["date"] != last_date:
                lines.append(f"[session of {turn['date']}]")
                last_date = turn["date"]
            lines.append(f"{turn['role'].upper()}: {turn['content']}")
    lines.append(_CURRENT_TURN_MARKER)
    if current_date is not None and current_date != last_date:
        lines.append(f"[session of {current_date}]")
    lines.append(f"USER: {current_turn}")
    return "\n".join(lines)


def parse_tuples(payload: dict) -> list[Tuple]:
    """Parse the extractor's JSON output into Tuple objects.

    `confidence` must be an integer 0-100 (protocol_spec §3.3); out-of-range
    values are a contract violation and raise.
    """
    parsed: list[Tuple] = []
    for row in payload["tuples"]:
        confidence = int(round(float(row["confidence"])))
        if not 0 <= confidence <= 100:
            raise ValueError(f"confidence out of range 0-100: {row['confidence']!r}")
        parsed.append(
            Tuple(
                entity=str(row["entity"]).strip(),
                attribute=str(row["attribute"]).strip(),
                value=str(row["value"]).strip(),
                statement_span=str(row["statement_span"]),
                confidence=confidence,
            )
        )
    return parsed


async def extract_turn(
    prior_turns: list[dict],
    current_turn: str,
    trace_tag: str,
    *,
    current_date: str | None = None,
) -> list[Tuple]:
    """One extractor call for one user turn (contract B2). Tuples are attributed
    to `current_turn`. `current_date` is the current turn's session date; it is
    rendered as a date marker when it differs from the last prior turn's session
    (e.g. the first turn of a new session)."""
    user_prompt = render_extraction_prompt(prior_turns, current_turn, current_date)
    output = await call_llm(EXTRACTOR_PROMPT, user_prompt, trace_tag=trace_tag)
    return parse_tuples(output)


# ---------------------------------------------------------------------------
# Extractor-tuples producer (R2-B-2): ../results/test1/extractor_tuples.jsonl
# ---------------------------------------------------------------------------

EXTRACTOR_TUPLES_PATH = RESULTS_DIR / "test1" / "extractor_tuples.jsonl"


def producer_jobs(schema_qids: list[str], candidates: list[dict]) -> list[tuple[str, str, bool]]:
    """Deterministic (source_file, question_id, perturbed) extraction jobs.

    - Every schema question (the 44 key_schema substrates) extracts from its
      PERTURBED substrate — all non-T3 manifest rows spliced, the identical
      splice the Test 2 replay sees (R2-B-2).
    - Every natural-candidate question not already covered extracts from its
      UNPERTURBED source substrate (natural-only source questions). An
      s_cleaned candidate sharing a question_id with a schema substrate is a
      DIFFERENT substrate (the s_cleaned haystack; manifest placements address
      oracle indices) and is extracted unperturbed under source_file
      "s_cleaned".
    """
    jobs: list[tuple[str, str, bool]] = [("oracle", qid, True) for qid in sorted(schema_qids)]
    seen = {("oracle", qid) for qid in schema_qids}
    for cand in candidates:
        key = (cand["source_file"], cand["question_id"])
        if key in seen:
            continue
        seen.add(key)
        jobs.append((cand["source_file"], cand["question_id"], False))
    return jobs


async def produce_extractor_tuples(out_path: Path = EXTRACTOR_TUPLES_PATH) -> int:
    """Write one row per (substrate, user turn) to extractor_tuples.jsonl.

    Row schema (documented in label.load_extractor_tuples): question_id,
    source_file, session_idx (ORIGINAL haystack index), turn_idx, turn_text
    (the extraction-time turn text, consumed by the R2-M-d packaging assert),
    tuples. Routing through store.extract_substrate_tuples makes these calls
    byte-identical to the Test 2 replay's phase-1 calls (single extraction
    pathway, B2), so reruns and the Test 2 run share the llm.py cache.
    """
    # Runner-level glue (review R2-B-2): store/data are used through their
    # INTERFACES-named functions; test1_eval supplies the natural-candidate
    # pool loader and substrate resolution so extraction sees the SAME
    # substrates the Test 0/1 packaging sees (the R2-M-d assert relies on it).
    import data
    import store
    import eval_classifier as test1_eval

    manifest = test1_eval.load_manifest()
    for row in manifest:
        if row["inj_type"] not in ("t3_staging", "t3_variant") and not row.get("text"):
            sys.exit("manifest has non-T3 rows without rendered text; "
                     "run `inject.py render` first")
    schema_qids = sorted(data.load_key_schema())
    candidates = test1_eval.load_natural_candidates()
    jobs = producer_jobs(schema_qids, candidates)

    async def run_job(source_file: str, qid: str, perturbed: bool) -> list[dict]:
        if perturbed:
            substrate = test1_eval.perturbed_substrate(
                test1_eval.non_t3_rows(manifest, qid), qid)
        else:
            substrate = test1_eval.substrate_for(source_file, qid)
        turn_rows = await store.extract_substrate_tuples(substrate)
        return [
            {
                "question_id": qid,
                "source_file": source_file,
                "session_idx": r["session_idx"],
                "turn_idx": r["turn_idx"],
                "turn_text": r["turn_text"],
                "tuples": [asdict(t) for t in r["tuples"]],
            }
            for r in turn_rows
        ]

    nested = await asyncio.gather(*(run_job(*job) for job in jobs))
    rows = [row for chunk in nested for row in chunk]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("produce-tuples",
                   help="write ../results/test1/extractor_tuples.jsonl (R2-B-2)")
    args = parser.parse_args()
    if args.cmd == "produce-tuples":
        n = asyncio.run(produce_extractor_tuples())
        print(f"wrote {EXTRACTOR_TUPLES_PATH} ({n} rows)")


if __name__ == "__main__":
    main()

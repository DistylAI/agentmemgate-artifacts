"""runner_products.py — Mem0 + Graphiti BLACK-BOX product baselines
(protocol/CHANGELOG.md 2026-08-10 "PREREGISTRATION: Mem0 + Zep/Graphiti
black-box product baselines"; that entry is the spec for this runner).

Systems (feasibility-smoked configs, reproduced verbatim):
  * Mem0 OSS 2.0.17, embedded file-backed Qdrant, pinned model via the
    documented ``is_reasoning_model`` config, text-embedding-3-small.
    OSS ``Memory.add`` has no ingestion timestamp -> the session date rides
    in metadata AND as a date line prepended to every message (the
    preregistered, disclosed fidelity caveat).
  * graphiti-core 0.29.3 on embedded Kuzu, pinned model for model AND
    small_model, per-MESSAGE episodes with reference_time = session date,
    one fresh Kuzu DB per conversation, with the two documented shims:
    (1) ``driver._database = <group_id>`` (KuzuDriver never sets _database
    and has no clone(); presetting skips the broken clone branch), and
    (2) manual FTS index creation (KuzuDriver.build_indices_and_constraints
    is a no-op but search paths query FTS indexes). The preregistered
    FalkorDB fallback is NOT wired here; it activates only if Kuzu degrades
    on the full run, in which case its use is logged (a Kuzu failure
    surfaces as a per-conversation error, never a silent fallback).

Substrates (each conversation run ONCE; resumable per conversation):
  * internal — the 60 conversation-benchmark arcs, results/v2.4/m4/arcs.jsonl.
  * external_d1 — the 147 frozen HOLDOUT_SPLIT dataset1 TEST conversations,
    session-grouped by adapter_d1.sessions_for (IMPORTED, never copied).
    These ids were de-novo-consumed by the 2026-07-31 one-shot run; the
    baselines prereg entry authorizes exactly this reuse ("reported as the
    same comparison set, NOT as a fresh held-out claim"). The loader
    fail-closed-checks every id against the frozen split via
    holdout_guard.slice_of; holdout_guard.assert_ingest_allowed (the
    EXTERNAL_TEST_RUN_AUTHORIZED one-shot flag) governs the frozen-harness
    systems only and is deliberately not reused here.

LABEL HYGIENE (hard rule): this runner records RAW ANSWERS ONLY — answering
and scoring are separate phases. No labels/answer-keys file is ever opened:
the internal loader touches only arc_id / key.probe_question / conversation;
the external loader touches only the conversations file, the model-visible
questions file, and the split — never adapter_d1.adapt / field_registry_row
/ the labels seam (those read the held-out label rows for attribute wiring).

Per conversation x product: fresh isolated store, sessions ingested strictly
chronologically, and after EACH session the conversation's probe question is
asked: product retrieval (mem0.search / graphiti.search, top RETRIEVAL_K),
then ONE pinned-model answering call over the frozen ANSWER_* template
(answer only from the provided memories; exactly "unknown" when absent).
Records: results/baselines/{mem0,graphiti}/{internal,external_d1}/<id>.json
(skip if the record file already exists). Concurrency: process pool across
conversations, hard cap 8; within a conversation strictly sequential.

Usage accounting: baseline product calls run OUTSIDE the SharedCallBudget
harness (product-internal calls are not interceptable); instead the OpenAI
SDK resource methods are wrapped per worker process (chat.completions /
responses / embeddings, sync + async) counting calls and tokens per record;
aggregate_usage() sums all records into results/baselines/usage.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "false")

_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_EVAL = _ROOT / "experiments" / "external_eval"
if str(_EXTERNAL_EVAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_EVAL))
import _bootstrap  # noqa: F401,E402  (R1 layout: puts src/ + experiments/ on sys.path)

from llm import MODEL as PINNED_MODEL  # noqa: E402  (the repo-wide pinned model)
from paths import RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
ANSWER_REASONING_EFFORT = "medium"
#: Retrieval depth for both products (mem0 top_k / graphiti num_results).
RETRIEVAL_K = 10

#: Pinned product versions (asserted at build time; the feasibility smoke and
#: the prereg entry name exactly these).
MEM0_VERSION = "2.0.17"
GRAPHITI_VERSION = "0.29.3"

PRODUCTS = ("mem0", "graphiti")
SUBSTRATES = ("internal", "external_d1")

ARCS_PATH = RESULTS_DIR / "v2.4" / "m4" / "arcs.jsonl"
DEFAULT_RESULTS_ROOT = RESULTS_DIR / "baselines"

#: FROZEN answering template (the ONE pinned-model call per probe). Contains
#: no substrate text and no label text — only the retrieved memories, the
#: as-of session date, and the probe question are ever formatted in.
ANSWER_SYSTEM_PROMPT = (
    "You are answering a question about a user, using ONLY the memories "
    "retrieved from a memory system. Answer only from the provided "
    "memories; do not use outside knowledge or guess. If the memories do "
    "not contain the answer, reply exactly: unknown. Otherwise reply with "
    "a short, direct answer and nothing else."
)
ANSWER_USER_TEMPLATE = (
    "Retrieved memories (collected from sessions up to {asof_date}):\n"
    "{memories}\n"
    "\n"
    "Question: {question}"
)
ANSWER_PROMPT_SHA = hashlib.sha256(
    (ANSWER_SYSTEM_PROMPT + "\n---\n" + ANSWER_USER_TEMPLATE).encode()
).hexdigest()

#: The record contract (tests pin it; scoring code reads these fields).
RECORD_KEYS = (
    "record_version", "product", "substrate", "conversation_id",
    "model", "embed_model", "retrieval_k", "answer_prompt_sha",
    "product_version", "kuzu_fallback", "n_sessions", "session_probes",
    "final_answer", "final_retrieved", "usage", "wall_clock_s",
    "started_at", "finished_at",
)

#: Corpus-wide session-date format (m4 arcs AND adapter_d1 both emit it).
DATE_FMT = "%Y/%m/%d (%a) %H:%M"


def parse_corpus_date(date_str: str) -> datetime:
    """'2023/02/21 (Tue) 18:25' -> tz-aware UTC datetime (graphiti
    reference_time requires tz-aware)."""
    return datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Substrate loaders (answers/labels NEVER read — see module docstring)
# ---------------------------------------------------------------------------

def load_internal(arcs_path: Path = ARCS_PATH) -> list[dict]:
    """The 60 internal conversation-benchmark conversations. Reads ONLY
    arc_id, key.probe_question and the conversation sessions from each arc
    record; every other field (values, resolution status, ...) is scoring
    material and is never touched."""
    convs = []
    for line in arcs_path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        sessions = []
        for sess in rec["conversation"]:
            sessions.append({
                "session_idx": sess["session_idx"],
                "date": sess["date"],
                "turns": [{"role": t["role"], "text": t["text"]}
                          for t in sess["turns"]],
            })
        stamps = [parse_corpus_date(s["date"]) for s in sessions]
        assert all(a <= b for a, b in zip(stamps, stamps[1:])), (
            f"{rec['arc_id']}: non-chronological sessions")
        convs.append({
            "conversation_id": rec["arc_id"],
            "substrate": "internal",
            "probe_question": rec["key"]["probe_question"],
            "sessions": sessions,
        })
    assert len(convs) == 60, f"expected 60 internal arcs, got {len(convs)}"
    return convs


def load_external_d1() -> list[dict]:
    """The 147 frozen dataset1 TEST conversations, grouped into calendar-day
    sessions by adapter_d1.sessions_for (imported — the single grouping
    implementation). Probe questions come from the MODEL-VISIBLE questions
    file (adapter_d1.questions). The labels seam is never touched."""
    import adapter_d1
    import holdout_guard

    ids = holdout_guard.test_ids("dataset1")
    assert len(ids) == 147, f"frozen dataset1 test slice must be 147, got {len(ids)}"
    convs = []
    for cid in ids:
        # Fail-closed membership check against the frozen split (read-only;
        # the one-shot ingest flag is NOT reused here — see module docstring).
        assert holdout_guard.slice_of("dataset1", cid) == "test"
        raw = adapter_d1.conversations()[cid]
        sessions = adapter_d1.sessions_for(raw)
        qids = raw["test_question_ids"]
        assert len(qids) == 1, f"{cid}: expected exactly 1 question, got {qids}"
        q = adapter_d1.questions()[qids[0]]
        assert q["record_id"] == cid, (
            f"{cid}: question {qids[0]} wired to {q['record_id']}")
        convs.append({
            "conversation_id": cid,
            "substrate": "external_d1",
            "probe_question": q["question_text"],
            "sessions": [{"session_idx": s["session_idx"], "date": s["date"],
                          "turns": [{"role": t["role"], "text": t["text"]}
                                    for t in s["turns"]]}
                         for s in sessions],
        })
    assert len(convs) == 147
    return convs


def load_substrate(substrate: str) -> list[dict]:
    if substrate == "internal":
        return load_internal()
    if substrate == "external_d1":
        return load_external_d1()
    raise ValueError(f"unknown substrate {substrate!r}")


# ---------------------------------------------------------------------------
# Usage accounting (per worker process; products' calls are not otherwise
# interceptable — prereg: accounted via wrapped OpenAI clients, outside the
# SharedCallBudget harness)
# ---------------------------------------------------------------------------

class UsageCounter:
    """Counts OpenAI API calls + tokens by endpoint kind."""

    def __init__(self) -> None:
        self.calls: Counter = Counter()
        self.tokens: Counter = Counter()

    def record(self, kind: str, usage) -> None:
        self.calls[kind] += 1
        if usage is None:
            return
        inp = getattr(usage, "input_tokens", None)
        if inp is None:
            inp = getattr(usage, "prompt_tokens", None)
        out = getattr(usage, "output_tokens", None)
        if out is None:
            out = getattr(usage, "completion_tokens", None)
        if isinstance(inp, int):
            self.tokens[f"{kind}_input"] += inp
        if isinstance(out, int):
            self.tokens[f"{kind}_output"] += out

    def snapshot(self) -> dict:
        return {"calls": dict(self.calls), "tokens": dict(self.tokens)}


def usage_delta(after: dict, before: dict) -> dict:
    """after - before, per counter family; also totals for convenience."""
    out = {}
    for fam in ("calls", "tokens"):
        d = {k: after[fam].get(k, 0) - before[fam].get(k, 0)
             for k in set(after[fam]) | set(before[fam])}
        out[fam] = {k: v for k, v in sorted(d.items()) if v}
    out["total_calls"] = sum(out["calls"].values())
    return out


_USAGE: UsageCounter | None = None


def install_usage_hooks() -> UsageCounter:
    """Wrap the OpenAI SDK resource methods (sync + async) that mem0,
    graphiti and our answering client use, so every API call in this process
    is counted. Idempotent per process."""
    global _USAGE
    if _USAGE is not None:
        return _USAGE
    _USAGE = UsageCounter()

    import openai.resources.chat.completions.completions as _cc
    import openai.resources.embeddings as _emb
    import openai.resources.responses.responses as _rr

    def _wrap_sync(cls, name, kind):
        fn = getattr(cls, name)

        def wrapped(self, *a, **k):
            r = fn(self, *a, **k)
            _USAGE.record(kind, getattr(r, "usage", None))
            return r

        setattr(cls, name, wrapped)

    def _wrap_async(cls, name, kind):
        fn = getattr(cls, name)

        async def wrapped(self, *a, **k):
            r = await fn(self, *a, **k)
            _USAGE.record(kind, getattr(r, "usage", None))
            return r

        setattr(cls, name, wrapped)

    _wrap_sync(_cc.Completions, "create", "chat")          # mem0 LLM
    _wrap_async(_cc.AsyncCompletions, "create", "chat")    # graphiti fallback path
    _wrap_sync(_emb.Embeddings, "create", "embeddings")    # mem0 embedder
    _wrap_async(_emb.AsyncEmbeddings, "create", "embeddings")  # graphiti embedder
    _wrap_sync(_rr.Responses, "create", "responses")       # our answering call
    _wrap_sync(_rr.Responses, "parse", "responses")
    _wrap_async(_rr.AsyncResponses, "create", "responses")
    _wrap_async(_rr.AsyncResponses, "parse", "responses")  # graphiti LLM
    return _USAGE


# ---------------------------------------------------------------------------
# Answering call (the ONE pinned-model call per probe; frozen template)
# ---------------------------------------------------------------------------

def build_answer_prompt(memory_lines: list[str], question: str,
                        asof_date: str) -> tuple[str, str]:
    memories = "\n".join(memory_lines) if memory_lines else "(no memories retrieved)"
    return (ANSWER_SYSTEM_PROMPT,
            ANSWER_USER_TEMPLATE.format(memories=memories, question=question,
                                        asof_date=asof_date))


def _answer_call(client, memory_lines: list[str], question: str,
                 asof_date: str) -> str:
    system_text, user_text = build_answer_prompt(memory_lines, question, asof_date)
    resp = client.responses.create(
        model=PINNED_MODEL,
        reasoning={"effort": ANSWER_REASONING_EFFORT},
        input=[
            {"role": "system",
             "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user",
             "content": [{"type": "input_text", "text": user_text}]},
        ],
    )
    return resp.output_text.strip()


# ---------------------------------------------------------------------------
# Mem0 (feasibility-smoked config)
# ---------------------------------------------------------------------------

def mem0_config(conversation_id: str, store_dir: Path) -> dict:
    """Per-conversation Mem0 OSS config: pinned model (is_reasoning_model is
    REQUIRED — mem0's name heuristic would otherwise send temperature=0.1,
    which the pinned model rejects), fresh file-backed qdrant path."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": PINNED_MODEL,
                "is_reasoning_model": True,
                "reasoning_effort": ANSWER_REASONING_EFFORT,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": EMBED_MODEL},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_baseline",
                "path": str(store_dir / "qdrant"),
                "on_disk": True,
            },
        },
        "history_db_path": str(store_dir / "mem0_history.db"),
    }


def _build_mem0(conversation_id: str, store_dir: Path):
    import mem0
    from mem0 import Memory
    assert mem0.__version__ == MEM0_VERSION, (
        f"mem0 {mem0.__version__} != pinned {MEM0_VERSION}")
    return Memory.from_config(mem0_config(conversation_id, store_dir))


def mem0_memory_lines(hits: list[dict]) -> list[str]:
    lines = []
    for h in hits:
        meta = h.get("metadata") or {}
        lines.append(f"- {h['memory']} (session_date: {meta.get('session_date')})")
    return lines


def _mem0_hit_snapshot(hits: list[dict]) -> list[dict]:
    return [{"memory": h.get("memory"),
             "score": h.get("score"),
             "metadata": h.get("metadata"),
             "created_at": h.get("created_at"),
             "updated_at": h.get("updated_at")} for h in hits]


def _run_mem0(conv: dict, store_dir: Path, counter: UsageCounter) -> list[dict]:
    """Sessions ingested chronologically, one add per session (session date
    in metadata + a date line on every message); probe after each session."""
    from openai import OpenAI

    memory = _build_mem0(conv["conversation_id"], store_dir)
    client = OpenAI()
    question = conv["probe_question"]
    probes = []
    for sess in conv["sessions"]:
        t0 = time.monotonic()
        before = counter.snapshot()
        date_str = sess["date"]
        day = parse_corpus_date(date_str).date().isoformat()
        messages = [{"role": t["role"], "content": f"[{date_str}] {t['text']}"}
                    for t in sess["turns"]]
        memory.add(messages,
                   user_id=conv["conversation_id"],
                   run_id=f"session_{sess['session_idx']}",
                   metadata={"session_date": day})
        hits = memory.search(question,
                             filters={"user_id": conv["conversation_id"]},
                             top_k=RETRIEVAL_K)["results"]
        answer = _answer_call(client, mem0_memory_lines(hits), question, date_str)
        probes.append({
            "session_idx": sess["session_idx"],
            "session_date": date_str,
            "answer": answer,
            "retrieved": _mem0_hit_snapshot(hits),
            "usage": usage_delta(counter.snapshot(), before),
            "wall_clock_s": round(time.monotonic() - t0, 3),
        })
    return probes


# ---------------------------------------------------------------------------
# Graphiti on embedded Kuzu (feasibility-smoked config + the two shims)
# ---------------------------------------------------------------------------

async def _build_graphiti(conversation_id: str, db_path: Path):
    import graphiti_core
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient

    # graphiti_core exposes no __version__ attribute; use dist metadata,
    # with a module-attr fallback (used by the SDK-mocked tests).
    installed = getattr(graphiti_core, "__version__", None)
    if installed is None:
        from importlib.metadata import version
        installed = version("graphiti-core")
    assert installed == GRAPHITI_VERSION, (
        f"graphiti-core {installed} != pinned {GRAPHITI_VERSION}")

    driver = KuzuDriver(db=str(db_path))
    # Shim 1 (graphiti-core 0.29.3 on Kuzu): add_episode(group_id=...) reads
    # driver._database, which KuzuDriver never sets (and it has no clone()).
    # Presetting it to our group_id skips the broken clone branch; Kuzu
    # stores group_id as a node property, so scoping still works.
    driver._database = conversation_id
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=OpenAIClient(
            config=LLMConfig(model=PINNED_MODEL, small_model=PINNED_MODEL)),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(embedding_model=EMBED_MODEL)),
        # cross_encoder left default; plain search() uses RRF (no reranker calls)
    )
    await graphiti.build_indices_and_constraints()
    # Shim 2: KuzuDriver.build_indices_and_constraints() is a no-op, but the
    # search paths query FTS indexes (e.g. edge_name_and_fact) and crash
    # without them. Create them explicitly on the fresh DB.
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.graph_queries import get_fulltext_indices
    existing, _, _ = await driver.execute_query("CALL SHOW_INDEXES() RETURN *")
    if not existing:
        for query in get_fulltext_indices(GraphProvider.KUZU):
            await driver.execute_query(query)
    return graphiti


def graphiti_memory_lines(edges) -> list[str]:
    return [f"- {e.fact} (valid_at={e.valid_at}, invalid_at={e.invalid_at})"
            for e in edges]


def _graphiti_edge_snapshot(edges) -> list[dict]:
    return [{"fact": e.fact,
             "name": getattr(e, "name", None),
             "uuid": str(getattr(e, "uuid", None)),
             "valid_at": str(e.valid_at) if e.valid_at else None,
             "invalid_at": str(e.invalid_at) if e.invalid_at else None}
            for e in edges]


async def _run_graphiti(conv: dict, store_dir: Path,
                        counter: UsageCounter) -> list[dict]:
    """Per-message episodes with reference_time = session date (the
    product-as-intended granularity); probe after each session. Strictly
    sequential within the conversation."""
    from graphiti_core.nodes import EpisodeType
    from openai import OpenAI

    cid = conv["conversation_id"]
    graphiti = await _build_graphiti(cid, store_dir / "kuzu.db")
    client = OpenAI()
    question = conv["probe_question"]
    probes = []
    try:
        for sess in conv["sessions"]:
            t0 = time.monotonic()
            before = counter.snapshot()
            date_str = sess["date"]
            ref_time = parse_corpus_date(date_str)
            si = sess["session_idx"]
            for ti, turn in enumerate(sess["turns"]):
                await graphiti.add_episode(
                    name=f"{cid}_s{si}_t{ti}",
                    episode_body=f"{turn['role']}: {turn['text']}",
                    source=EpisodeType.message,
                    source_description=f"chat session {si} ({date_str})",
                    reference_time=ref_time,
                    group_id=cid,
                )
            edges = await graphiti.search(question, group_ids=[cid],
                                          num_results=RETRIEVAL_K)
            answer = _answer_call(client, graphiti_memory_lines(edges),
                                  question, date_str)
            probes.append({
                "session_idx": si,
                "session_date": date_str,
                "answer": answer,
                "retrieved": _graphiti_edge_snapshot(edges),
                "usage": usage_delta(counter.snapshot(), before),
                "wall_clock_s": round(time.monotonic() - t0, 3),
            })
    finally:
        await graphiti.close()
    return probes


# ---------------------------------------------------------------------------
# Per-conversation record (resumable)
# ---------------------------------------------------------------------------

def record_path(results_root: Path, product: str, substrate: str,
                conversation_id: str) -> Path:
    return results_root / product / substrate / f"{conversation_id}.json"


def make_record(*, product: str, substrate: str, conversation_id: str,
                product_version: str, session_probes: list[dict],
                usage: dict, wall_clock_s: float, started_at: str,
                finished_at: str, kuzu_fallback: bool = False) -> dict:
    record = {
        "record_version": 1,
        "product": product,
        "substrate": substrate,
        "conversation_id": conversation_id,
        "model": PINNED_MODEL,
        "embed_model": EMBED_MODEL,
        "retrieval_k": RETRIEVAL_K,
        "answer_prompt_sha": ANSWER_PROMPT_SHA,
        "product_version": product_version,
        "kuzu_fallback": kuzu_fallback,
        "n_sessions": len(session_probes),
        "session_probes": session_probes,
        "final_answer": session_probes[-1]["answer"] if session_probes else None,
        "final_retrieved": session_probes[-1]["retrieved"] if session_probes else [],
        "usage": usage,
        "wall_clock_s": round(wall_clock_s, 3),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    assert tuple(record) == RECORD_KEYS
    return record


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def run_conversation(product: str, substrate: str, conv: dict,
                     results_root: Path, store_root: Path) -> dict:
    """Run ONE conversation through ONE product end-to-end and write its
    record. Resumable: an existing record file short-circuits before any
    store/SDK work."""
    cid = conv["conversation_id"]
    rec_path = record_path(results_root, product, substrate, cid)
    if rec_path.exists():
        return {"product": product, "substrate": substrate,
                "conversation_id": cid, "skipped": True}

    counter = install_usage_hooks()
    before = counter.snapshot()
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    # Fresh isolated store: wipe any partial dir from a crashed attempt.
    store_dir = store_root / product / substrate / cid
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)

    if product == "mem0":
        product_version = MEM0_VERSION
        session_probes = _run_mem0(conv, store_dir, counter)
    elif product == "graphiti":
        product_version = GRAPHITI_VERSION
        session_probes = asyncio.run(_run_graphiti(conv, store_dir, counter))
    else:
        raise ValueError(f"unknown product {product!r}")

    record = make_record(
        product=product, substrate=substrate, conversation_id=cid,
        product_version=product_version, session_probes=session_probes,
        usage=usage_delta(counter.snapshot(), before),
        wall_clock_s=time.monotonic() - t0,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_write_json(rec_path, record)
    return {"product": product, "substrate": substrate, "conversation_id": cid,
            "skipped": False, "final_answer": record["final_answer"],
            "total_calls": record["usage"]["total_calls"],
            "wall_clock_s": record["wall_clock_s"]}


# ---------------------------------------------------------------------------
# Usage aggregation -> results/baselines/usage.json
# ---------------------------------------------------------------------------

def aggregate_usage(results_root: Path = DEFAULT_RESULTS_ROOT) -> dict:
    """Sum per-record usage into results/baselines/usage.json (per product x
    substrate cell + grand total). Baseline calls run outside the
    SharedCallBudget harness per the prereg entry; this file is their
    separate accounting."""
    per_cell: dict = {}
    total_calls: Counter = Counter()
    total_tokens: Counter = Counter()
    n_records = 0
    for product in PRODUCTS:
        for substrate in SUBSTRATES:
            cell_dir = results_root / product / substrate
            if not cell_dir.is_dir():
                continue
            calls: Counter = Counter()
            tokens: Counter = Counter()
            wall = 0.0
            n = 0
            for path in sorted(cell_dir.glob("*.json")):
                rec = json.loads(path.read_text())
                calls.update(rec["usage"]["calls"])
                tokens.update(rec["usage"]["tokens"])
                wall += rec["wall_clock_s"]
                n += 1
            if not n:
                continue
            per_cell[f"{product}/{substrate}"] = {
                "n_records": n,
                "calls": dict(calls),
                "total_calls": sum(calls.values()),
                "tokens": dict(tokens),
                "wall_clock_s": round(wall, 1),
            }
            total_calls.update(calls)
            total_tokens.update(tokens)
            n_records += n
    usage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": PINNED_MODEL,
        "embed_model": EMBED_MODEL,
        "accounting_note": ("black-box product baselines; calls counted by "
                            "wrapped OpenAI SDK methods, OUTSIDE the "
                            "SharedCallBudget harness (prereg 2026-08-10)"),
        "n_records": n_records,
        "per_cell": per_cell,
        "total": {"calls": dict(total_calls),
                  "total_calls": sum(total_calls.values()),
                  "tokens": dict(total_tokens)},
    }
    _atomic_write_json(results_root / "usage.json", usage)
    return usage


# ---------------------------------------------------------------------------
# CLI (process pool across conversations, hard cap 8)
# ---------------------------------------------------------------------------

def _worker(task: tuple) -> dict:
    product, substrate, conv, results_root, store_root = task
    try:
        return run_conversation(product, substrate, conv,
                                Path(results_root), Path(store_root))
    except Exception as exc:  # keep the pool alive; conversation is retryable
        return {"product": product, "substrate": substrate,
                "conversation_id": conv["conversation_id"],
                "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--products", nargs="+", choices=PRODUCTS,
                        default=list(PRODUCTS))
    parser.add_argument("--substrates", nargs="+", choices=SUBSTRATES,
                        default=list(SUBSTRATES))
    parser.add_argument("--limit", type=int, default=None,
                        help="first N conversations per substrate (smoke)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="process-pool width across conversations (cap 8)")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--store-root", default=os.environ.get(
        "BASELINE_STORE_ROOT", ""),
        help="scratch dir for per-conversation product stores "
             "(default: fresh temp dir)")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    store_root = Path(args.store_root) if args.store_root else Path(
        tempfile.mkdtemp(prefix="baseline_stores_"))
    workers = max(1, min(args.concurrency, 8))  # preregistered hard cap

    tasks = []
    for substrate in args.substrates:
        convs = load_substrate(substrate)
        if args.limit is not None:
            convs = convs[:args.limit]
        for product in args.products:
            for conv in convs:
                tasks.append((product, substrate, conv,
                              str(results_root), str(store_root)))
    print(f"{len(tasks)} conversation-runs queued "
          f"(workers={workers}, stores={store_root})", flush=True)

    done = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            tag = f"{res['product']}/{res['substrate']}/{res['conversation_id']}"
            if res.get("skipped"):
                skipped += 1
                print(f"SKIP  {tag} (record exists)", flush=True)
            elif "error" in res:
                failed += 1
                print(f"FAIL  {tag}: {res['error']}", flush=True)
            else:
                done += 1
                print(f"OK    {tag} calls={res['total_calls']} "
                      f"wall={res['wall_clock_s']}s "
                      f"final={res['final_answer']!r}", flush=True)

    usage = aggregate_usage(results_root)
    print(f"done={done} skipped={skipped} failed={failed}; "
          f"usage.json total_calls={usage['total']['total_calls']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

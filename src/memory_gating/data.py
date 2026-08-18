"""Dataset access for the memory-gating pilot (role I1).

Parses the LongMemEval oracle file and the frozen key schema, and exposes
per-question ``Substrate`` objects whose sessions keep their ORIGINAL
haystack indices (key_schema and natural_candidates address original
session_idx) while ``session_order`` gives the chronological iteration
order. Haystack order is NOT guaranteed chronological (34/500 oracle
questions are out of order, e.g. gpt4_2655b836); code must never assume
it is (pinned decision P12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

from paths import DATA_DIR, GOLD_DIR

ORACLE_PATH = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"
KEY_SCHEMA_PATH = GOLD_DIR / "key_schema.json"

# e.g. "2023/05/25 (Thu) 20:21"
HAYSTACK_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


@dataclass
class Substrate:
    """One oracle question's sessions, addressable by ORIGINAL haystack index.

    ``sessions`` and ``dates`` keep the original haystack_sessions order, so
    ``sessions[i]`` corresponds to key_schema / natural_candidates
    ``session_idx == i``. ``session_order`` lists those original indices
    sorted chronologically by haystack_dates; iterate it to replay the
    substrate in time order. Turn dicts are exactly ``{role, content}``
    (the oracle's ``has_answer`` gold anchor is stripped so downstream
    replay/labeling code cannot see it).
    """

    question_id: str
    sessions: list[list[dict]]
    dates: list[str]
    session_order: list[int]


def parse_haystack_date(date: str) -> datetime:
    """Parse a LongMemEval haystack date like ``2023/05/25 (Thu) 20:21``."""
    return datetime.strptime(date, HAYSTACK_DATE_FORMAT)


def chronological_order(dates: list[str]) -> list[int]:
    """Indices of ``dates`` sorted chronologically (stable on exact ties)."""
    return sorted(range(len(dates)), key=lambda i: parse_haystack_date(dates[i]))


@cache
def load_oracle() -> list[dict]:
    """Parse ``../data/longmemeval/longmemeval_oracle.json`` (500 questions).

    The parsed list is cached and shared; callers must not mutate it
    (``get_substrate`` hands out fresh copies).
    """
    return json.loads(ORACLE_PATH.read_text())


@cache
def _oracle_by_id() -> dict[str, dict]:
    return {question["question_id"]: question for question in load_oracle()}


@cache
def load_key_schema() -> dict[str, dict]:
    """question_id -> key_schema substrate entry (all 44: 25 core + 19 alternate)."""
    schema = json.loads(KEY_SCHEMA_PATH.read_text())
    return {entry["question_id"]: entry for entry in schema["substrates"]}


def get_substrate(question_id: str) -> Substrate:
    """Build a ``Substrate`` for any oracle question (fresh copies each call)."""
    question = _oracle_by_id()[question_id]
    sessions = [
        [{"role": turn["role"], "content": turn["content"]} for turn in session]
        for session in question["haystack_sessions"]
    ]
    dates = list(question["haystack_dates"])
    return Substrate(
        question_id=question_id,
        sessions=sessions,
        dates=dates,
        session_order=chronological_order(dates),
    )


def user_turns(substrate: Substrate) -> list[tuple[int, int, str]]:
    """(orig_session_idx, turn_idx, text) for every user turn, in chronological order."""
    turns = []
    for orig_idx in substrate.session_order:
        for turn_idx, turn in enumerate(substrate.sessions[orig_idx]):
            if turn["role"] == "user":
                turns.append((orig_idx, turn_idx, turn["content"]))
    return turns

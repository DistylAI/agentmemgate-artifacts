"""ZepStore — history-preserving supersession store (protocol_spec_v2 §1.2 item 5;
zep_arm_design.md §4; freeze item §10.1/10).

The shared store primitive for all four v2 cells: the ungated column (C-W)
writes through it directly; the gated column's (C-G) commit / supersede /
retro-invalidate operations use the same primitive. The pilot's Test-2 dict
(last-writer-wins, `store.py`) is the degenerate case of this store with
history dropped, so `lookup()` is provably last-writer-wins-equivalent
(asserted by tests/test_zepstore.py before any v2 run, per §10.1 item 10).

Entry (zep_arm_design §4): (value, provenance, valid_from, invalid_from,
created_seq, expired_seq) where
  - valid_from    = host session date (V timeline; tie-broken by write order),
  - invalid_from  = None until superseded (closed at the superseder's
                    valid_from — or at the entry's OWN valid_from when a
                    correction-class commit retro-invalidates it),
  - created_seq / expired_seq = ingestion sequence numbers (the T'
    transactional timeline, RWG §2 `t'_created`/`t'_expired`).

Write algorithm (per canonicalized tuple, AFTER the shared key-mapper call —
this module is LLM-free by design):
  1. sentinel canonicalization (protocol_spec_v2 §1.2 item 10): an emitted
     value matching the frozen SENTINEL_RE is stored as `(unknown)`;
  2. refresh (Z7): if a current entry exists and the new value is
     normalize_value-equal to it, log a duplicate event, no history change;
  3. otherwise supersede: close the current entry (invalid_from, expired_seq)
     and append the new entry as current. The old entry is retained (Z4:
     supersede, never delete).

Retro-invalidation hook (gated column ONLY — the caller-side policy; the
ungated column always writes with correction=False per zep_arm_design §4
"no correction/update distinction"): a correction-class commit
(`write(..., correction=True)`) marks the superseded interval
invalid-from-origin (`old.invalid_from = old.valid_from`, an empty validity
interval: the record was never right — protocol_spec v1 §1.3). This is a
history/provenance difference only, never a `lookup()` difference.

Lookups:
  - lookup(key)          current value — pilot last-writer-wins-equivalent;
  - present(key)         presence probe: the `(unknown)` sentinel counts as
                         ABSENT (P-MISATTR, protocol_spec_v2 §4);
  - as_of(key, date)     P-TIME: the value whose [valid_from, invalid_from)
                         validity interval contains the probe date;
  - as_of_seq(key, seq)  transaction-timeline lookup: store state as of a
                         past ingestion sequence number (probe points need no
                         snapshot copies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from store import normalize_value

# ---------------------------------------------------------------------------
# Negation-without-replacement sentinel (protocol_spec_v2 §1.2 item 10)
# ---------------------------------------------------------------------------

#: The frozen sentinel value written when a committed tuple asserts
#: departure/cessation with no replacement value (v1 rule 1.5f
#: "supersede-with-unknown-new-value"). A key whose current value is the
#: sentinel counts as ABSENT for presence probes and as NON-EQUIVALENT to any
#: concrete value for equivalence scoring.
UNKNOWN_SENTINEL = "(unknown)"

# Frozen sentinel regex (freeze-set artifact, §10.1 item 10). Deliberately
# conservative: whole-string no-value markers only — never substring matches,
# so concrete values like "unknown artist" or "none of the above" never
# trigger it. Neither extractor prompt changes; this canonicalization is the
# shared write path's, identical in all four cells.
_SENTINEL_FORMS = r"""
    unknown
  | none
  | n/a | n\.a\.?
  | null | nil
  | not \s+ applicable
  | no \s+ value
  | not \s+ known
  | unspecified
  | no \s+ longer \s+ (?: applicable | valid | true | holds | current )
  | does \s+ not \s+ apply (?: \s+ any \s* more )?
"""
SENTINEL_RE = re.compile(
    rf"^\s*[\(\[\"']?\s*(?:{_SENTINEL_FORMS})\s*[\)\]\"']?\s*$",
    re.IGNORECASE | re.VERBOSE,
)


def is_sentinel(value: str) -> bool:
    """True iff the emitted value matches the frozen sentinel regex (or is
    already the canonical sentinel)."""
    return value == UNKNOWN_SENTINEL or bool(SENTINEL_RE.match(str(value)))


def apply_sentinel(value: str) -> str:
    """The shared write path's value canonicalization: sentinel-matching
    emitted values are stored as the frozen `(unknown)` sentinel; every other
    value is stored verbatim."""
    return UNKNOWN_SENTINEL if is_sentinel(value) else value


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class ZepEntry:
    """One store entry on both timelines (zep_arm_design §4).

    Validity (V) timeline: [valid_from, invalid_from) — half-open; None
    invalid_from = still valid. A retro-invalidated entry has
    invalid_from == valid_from (an empty interval: never valid).
    Transactional (T') timeline: [created_seq, expired_seq) — half-open over
    ingestion sequence numbers; None expired_seq = still current.
    """

    value: str  # sentinel-canonicalized stored value
    provenance: Any  # opaque; pilot shape: (session_idx, turn_idx, span, confidence)
    valid_from: str  # host session date (injected turns inherit it, P12)
    created_seq: int
    invalid_from: str | None = None
    expired_seq: int | None = None
    retro_invalidated: bool = False  # gated-column correction hook

    @property
    def is_current(self) -> bool:
        return self.expired_seq is None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ZepStore:
    """History-preserving supersession store; one instance per store unit
    (per substrate in LME, per speaker-store in LoCoMo)."""

    def __init__(self) -> None:
        self._history: dict[tuple, list[ZepEntry]] = {}
        self._seq: int = 0
        self.events: list[dict] = []  # provenance audit trail, one row per write

    # -- writing ------------------------------------------------------------

    @property
    def seq(self) -> int:
        """The last consumed ingestion sequence number (0 before any write)."""
        return self._seq

    def write(
        self,
        key: tuple,
        value: str,
        *,
        valid_from: str,
        provenance: Any = None,
        correction: bool = False,
    ) -> dict:
        """Apply one canonicalized (post key-mapper) tuple write.

        `correction=True` is the gated column's retro-invalidation hook
        (correction-class commits only): the superseded interval is marked
        invalid-from-origin. The ungated column NEVER passes it (caller
        policy, zep_arm_design §4). Within-turn ordering is the caller's:
        tuples apply in emitted list order (D5).

        Returns the logged event dict (action: commit | refresh | supersede).
        """
        key = tuple(key)
        self._seq += 1
        seq_now = self._seq
        stored_value = apply_sentinel(value)
        sentinel_applied = stored_value != value

        entries = self._history.setdefault(key, [])
        cur = entries[-1] if entries and entries[-1].is_current else None

        event = {
            "seq": seq_now,
            "key": key,
            "value": stored_value,
            "valid_from": valid_from,
            "correction": bool(correction),
            "sentinel_applied": sentinel_applied,
            "prev_value": cur.value if cur is not None else None,
        }

        if cur is not None and normalize_value(stored_value) == normalize_value(cur.value):
            # Z7 refresh: duplicate assertion — logged, no history change.
            event["action"] = "refresh"
            event["retro_invalidated"] = False
            self.events.append(event)
            return event

        retro = False
        if cur is not None:
            # Supersede, never delete (Z4). Correction-class commits mark the
            # superseded interval invalid-from-origin (retro-invalidation).
            retro = bool(correction)
            cur.invalid_from = cur.valid_from if retro else valid_from
            cur.expired_seq = seq_now
            cur.retro_invalidated = retro
            event["action"] = "supersede"
        else:
            event["action"] = "commit"
        event["retro_invalidated"] = retro

        entries.append(
            ZepEntry(
                value=stored_value,
                provenance=provenance,
                valid_from=valid_from,
                created_seq=seq_now,
            )
        )
        self.events.append(event)
        return event

    # -- lookups ------------------------------------------------------------

    def current_entry(self, key: tuple) -> ZepEntry | None:
        entries = self._history.get(tuple(key), [])
        if entries and entries[-1].is_current:
            return entries[-1]
        return None

    def lookup(self, key: tuple) -> str | None:
        """Current (non-invalidated) entry's value — behaviorally identical to
        the pilot's last-writer-wins dict at every probe point (zep_arm_design
        §4; equivalence asserted in tests/test_zepstore.py). Returns the
        `(unknown)` sentinel verbatim when that is the current value: presence
        semantics belong to `present()`."""
        entry = self.current_entry(key)
        return entry.value if entry is not None else None

    def present(self, key: tuple) -> bool:
        """Presence probe (P-MISATTR, protocol_spec_v2 §4): True iff the key
        has a current value that is not the `(unknown)` sentinel."""
        value = self.lookup(key)
        return value is not None and value != UNKNOWN_SENTINEL

    def as_of(
        self,
        key: tuple,
        probe_date: str,
        *,
        date_key: Callable[[str], Any] | None = None,
    ) -> str | None:
        """P-TIME validity-timeline lookup (protocol_spec_v2 §4): the value
        whose [valid_from, invalid_from) interval contains the probe date,
        read from supersession history.

        `date_key` converts stored/probe date strings to comparable values;
        the default compares the raw strings (correct for lexicographically
        ordered formats; LoCoMo callers pass a `%I:%M %p on %d %B, %Y`
        parser). Retro-invalidated entries have empty intervals and never
        match. If several intervals contain the date (impossible under
        chronological replay, defensive otherwise), the latest-created entry
        wins. Returns the `(unknown)` sentinel verbatim when the containing
        interval holds it (scoring treats it as absent / non-equivalent)."""
        keyfn = date_key if date_key is not None else (lambda d: d)
        probe = keyfn(probe_date)
        best: ZepEntry | None = None
        for entry in self._history.get(tuple(key), []):
            if keyfn(entry.valid_from) > probe:
                continue
            if entry.invalid_from is not None and keyfn(entry.invalid_from) <= probe:
                continue
            if best is None or entry.created_seq > best.created_seq:
                best = entry
        return best.value if best is not None else None

    def as_of_seq(self, key: tuple, seq: int) -> str | None:
        """Transactional-timeline lookup: the value current as of ingestion
        sequence `seq` (state after the seq-th write; retro-invalidated
        entries are still visible before their correction arrived — T'
        records what the store believed, V records what was true)."""
        best: ZepEntry | None = None
        for entry in self._history.get(tuple(key), []):
            if entry.created_seq > seq:
                continue
            if entry.expired_seq is not None and entry.expired_seq <= seq:
                continue
            if best is None or entry.created_seq > best.created_seq:
                best = entry
        return best.value if best is not None else None

    # -- history / audit ----------------------------------------------------

    def keys(self) -> list[tuple]:
        return list(self._history.keys())

    def history(self, key: tuple) -> list[ZepEntry]:
        """Full supersession history of the key, oldest first (Z4: nothing is
        ever deleted)."""
        return list(self._history.get(tuple(key), []))

    def history_depth(self, key: tuple) -> int:
        return len(self._history.get(tuple(key), []))

    def snapshot(self) -> dict[tuple, str]:
        """Current values for all keys — the pilot last-writer-wins dict view."""
        return {
            key: entries[-1].value
            for key, entries in self._history.items()
            if entries and entries[-1].is_current
        }

    def snapshot_at_seq(self, seq: int) -> dict[tuple, str]:
        """The `snapshot()` view as of a past ingestion sequence number."""
        snap: dict[tuple, str] = {}
        for key in self._history:
            value = self.as_of_seq(key, seq)
            if value is not None:
                snap[key] = value
        return snap

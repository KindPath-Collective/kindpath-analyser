"""
KindPath Analyser :: Seedbank CDC — Change Data Capture

An append-only event log for every mutation to the seedbank.

Every deposit, fork, recompute, stale-mark, and confluence is written as a
single JSON line to seedbank/events.jsonl. Lines are never deleted; that would
compromise the integrity of the provenance record.

This log is the authoritative audit trail for the seedbank. provenance.py
reads it to reconstruct the full history of any record or the corpus as a whole.

Hash chain:
    Each event carries a chain_hash — SHA-256 of (prior_chain_hash + this_event_json).
    A broken chain signals tampering. The chain can be verified at any point
    via verify_chain(). This is lightweight Merkle-style tamper evidence:
    not a full distributed ledger, but sufficient to detect any post-write mutation
    to any line, including deletions of lines from the middle.

On the 2-key future:
    The CDC log's hash chain is the structural skeleton that the planned dual-key
    encryption will lock around. Key 1 (k-key, derived from the constants baseline)
    seals the shared context. Key 2 (Δ-key, derived from the record's individual delta)
    seals the individual record. Neither key alone decrypts the full record —
    a design that maps directly to the community data ownership model: the hub
    holds the k-key; the individual holds the Δ-key. This is not yet implemented
    here but the hash chain is the prerequisite structure it requires.

Event types:
    deposit            — a new record was deposited
    fork               — a record's reading was forked (recompute produced a new universe)
    confluence         — a confluent reading was produced from multiple prior universes
    historical_marked  — a record was marked historical due to a tag revision
    historical_cleared — a historical flag was cleared (by recompute or manual correction)
    verified           — a record's provenance context was verified by a human
    warning            — runtime/system warning captured as an archival confluence point

Legacy aliases are accepted for compatibility:
    stale_marked  -> historical_marked
    stale_cleared -> historical_cleared
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

_CDC_FILE = os.path.join(os.path.dirname(__file__), "events.jsonl")

_EVENT_TYPE_ALIASES = {
    "stale_marked": "historical_marked",
    "stale_cleared": "historical_cleared",
}


# ── Write ─────────────────────────────────────────────────────────────────────

def append_event(
    event_type: str,
    record_id: str,
    data: dict,
    *,
    _file: str = _CDC_FILE,
) -> dict:
    """
    Append one event to the CDC log.

    event_type: "deposit" | "fork" | "confluence" | "historical_marked" |
                "historical_cleared" | "verified" | "warning"
    record_id:  the seedbank record UUID this event concerns
    data:       event-specific payload (arbitrary dict, must be JSON-serialisable)

    The chain_hash field is computed automatically — do not pass it in data.

    Returns the event dict as written (including chain_hash).
    """
    canonical_type = _EVENT_TYPE_ALIASES.get(event_type, event_type)
    event_data = dict(data or {})
    if canonical_type != event_type:
        # Preserve the caller's legacy vocabulary while storing one canonical type.
        event_data.setdefault("legacy_event_type", event_type)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": canonical_type,
        "record_id": record_id,
        "data": event_data,
    }

    # Build the hash chain link.
    # The chain input is: prior_chain_hash + canonical JSON of this event (without chain_hash).
    # Computing chain_hash after writing would allow undetected insertion of events
    # before the final line — so we compute it before writing.
    prior_hash = _read_last_hash(_file)
    event_json_for_chain = json.dumps(event, separators=(",", ":"), sort_keys=True)
    chain_input = (prior_hash + event_json_for_chain).encode("utf-8")
    event["chain_hash"] = hashlib.sha256(chain_input).hexdigest()

    with open(_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")

    return event


def _read_last_hash(_file: str) -> str:
    """Return the chain_hash of the last line in the log, or 'genesis' if empty/missing."""
    if not os.path.exists(_file):
        return "genesis"
    try:
        with open(_file, "rb") as fh:
            fh.seek(0, 2)  # seek to end
            size = fh.tell()
            if size == 0:
                return "genesis"
            # Walk backwards to find the last non-empty line
            pos = size - 1
            while pos > 0:
                fh.seek(pos)
                char = fh.read(1)
                if char == b"\n" and pos < size - 1:
                    break
                pos -= 1
            if pos == 0:
                fh.seek(0)
            last_line = fh.read().strip()
            if not last_line:
                return "genesis"
            last_event = json.loads(last_line)
            return last_event.get("chain_hash", "genesis")
    except Exception:
        return "genesis"


# ── Read ──────────────────────────────────────────────────────────────────────

def read_events(
    event_type: Optional[str] = None,
    record_id: Optional[str] = None,
    *,
    _file: str = _CDC_FILE,
) -> list:
    """
    Read the CDC log, optionally filtered by event_type and/or record_id.
    Returns events in chronological order (oldest first).
    """
    if not os.path.exists(_file):
        return []

    canonical_filter = _EVENT_TYPE_ALIASES.get(event_type, event_type) if event_type else None

    results = []
    with open(_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if canonical_filter is not None and event.get("event_type") != canonical_filter:
                continue
            if record_id is not None and event.get("record_id") != record_id:
                continue
            results.append(event)

    return results


def get_events_for_record(record_id: str, *, _file: str = _CDC_FILE) -> list:
    """Return all events for a specific record, chronologically."""
    return read_events(record_id=record_id, _file=_file)


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_chain(*, _file: str = _CDC_FILE) -> dict:
    """
    Verify the hash chain integrity of the CDC log.

    Recomputes each event's chain_hash from scratch and compares it to what
    was stored. Any mismatch — including a deletion, insertion, or edit of a
    prior line — will break the chain at the point of mutation.

    Returns:
        ok:                  bool  — True if chain is unbroken
        events_verified:     int   — number of events checked
        broken_at_line:      int|None — 1-based line number of first broken link
        error:               str|None — description of break
    """
    if not os.path.exists(_file):
        return {"ok": True, "events_verified": 0, "broken_at_line": None, "error": None}

    prior_hash = "genesis"
    line_num = 0

    with open(_file, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line_num += 1

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "events_verified": line_num - 1,
                    "broken_at_line": line_num,
                    "error": f"JSON parse error: {exc}",
                }

            stored_hash = event.get("chain_hash", "")

            # Reconstruct the pre-chain-hash event (as it was when chain_hash was computed)
            event_for_chain = {k: v for k, v in event.items() if k != "chain_hash"}
            event_json = json.dumps(event_for_chain, separators=(",", ":"), sort_keys=True)
            chain_input = (prior_hash + event_json).encode("utf-8")
            expected_hash = hashlib.sha256(chain_input).hexdigest()

            if stored_hash != expected_hash:
                return {
                    "ok": False,
                    "events_verified": line_num - 1,
                    "broken_at_line": line_num,
                    "error": (
                        f"Chain break: record={event.get('record_id', '?')} "
                        f"type={event.get('event_type', '?')} "
                        f"expected={expected_hash[:12]}… got={stored_hash[:12]}…"
                    ),
                }

            prior_hash = stored_hash

    return {"ok": True, "events_verified": line_num, "broken_at_line": None, "error": None}


# ── Convenience helpers ───────────────────────────────────────────────────────

def cdc_stats(*, _file: str = _CDC_FILE) -> dict:
    """
    Summary statistics about the CDC log.
    """
    events = read_events(_file=_file)
    by_type: dict = {}
    for e in events:
        t = e.get("event_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total_events": len(events),
        "by_type": by_type,
        "first_event_at": events[0]["timestamp"] if events else None,
        "last_event_at": events[-1]["timestamp"] if events else None,
        "file": _file,
    }

"""
KindPath Analyser :: Seedbank Provenance

Reconstructs, verifies, and exports the full provenance of any seedbank record.

Provenance is built from two sources:
    1. The record's reading_history in the index — the current state
    2. The events.jsonl CDC log — the append-only audit trail

Together they form a complete picture: what was deposited, when, under which
constants, forked how many times, whether reading_input_hash is present, and
whether the log chain remains unbroken.

Provenance bundles are self-describing JSON files that contain everything
needed to independently verify a record's authenticity and computational
lineage — without access to the original audio.

Sealing:
    Bundles can be sealed via transport.seal_record_bundle() to produce an
    AES-256-GCM encrypted, tamper-evident archive. The seal function is an
    optional dependency — provenance.py works without it. When sealed, the
    bundle becomes the IP protection artefact: a time-stamped, content-hashed,
    encrypted record of exactly what existed at the point of sealing.

On paper-strawed IP:
    The primary defence against intentional under-production or misattribution
    is publication timing. The CDC chain hash + a sealed provenance bundle
    committed to a public Git repo provides a timestamped witness: the work
    existed in this exact form at this date. Opening the bundle proves it.
    Encryption is secondary to this — the hash is the proof, the encryption
    is the protection of sensitive context around it.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import seedbank.index as _idx
from seedbank.cdc import get_events_for_record, verify_chain, read_events


# ── Record provenance ─────────────────────────────────────────────────────────

def record_provenance(record_id: str) -> dict:
    """
    Reconstruct the full provenance of a seedbank record.

    Returns a dict (the provenance bundle) containing:
        record_id:                  the record UUID
        filename:                   original filename
        deposited_at:               UTC ISO timestamp of initial deposit
        reading_count:              total readings (universes) to date
        reading_history:            full fork/confluence history from index
        cdc_events:                 all CDC log entries for this record
        cdc_chain_ok:               whether the full log chain is intact
        cdc_chain_events_total:     total events in the full log (not just this record)
        integrity_issues:           list of any cross-check failures
        bundle_generated_at:        when this bundle was assembled
        provenance_hash:            SHA-256 of the bundle content (for sealing)

    Raises ValueError if record not found.
    """
    index = _idx._load_index()
    record = next((r for r in index["records"] if r.get("id") == record_id), None)
    if record is None:
        raise ValueError(f"Record not found: {record_id}")

    cdc_events = get_events_for_record(record_id)
    chain_result = verify_chain()
    integrity = _check_integrity(record, cdc_events, chain_result)
    extraction_events = [
        e for e in cdc_events
        if e.get("event_type") == "confluence"
        and e.get("data", {}).get("kind") == "kindpress_extraction_state"
    ]

    bundle: dict = {
        "record_id": record_id,
        "filename": record.get("filename", ""),
        "deposited_at": record.get("deposited_at", ""),
        "reading_count": len(record.get("reading_history", [])),
        "reading_history": record.get("reading_history", []),
        "cdc_events": cdc_events,
        "kindpress_extraction_events": extraction_events,
        "kindpress_extraction_event_count": len(extraction_events),
        "cdc_chain_ok": chain_result["ok"],
        "cdc_chain_events_total": chain_result["events_verified"],
        "integrity_issues": integrity["issues"],
        "bundle_generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Deterministic hash of the bundle content (excluding the hash field itself)
    bundle_json = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    bundle["provenance_hash"] = hashlib.sha256(bundle_json.encode("utf-8")).hexdigest()

    return bundle


def export_provenance_bundle(
    record_id: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Export the provenance of a record as a JSON file.

    If output_path is given, writes there. Otherwise writes to
    seedbank/records/{record_id}_provenance.json.

    Returns the output path.
    """
    bundle = record_provenance(record_id)

    if output_path is None:
        records_dir = _idx.RECORDS_DIR
        output_path = os.path.join(records_dir, f"{record_id}_provenance.json")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)

    issues = bundle.get("integrity_issues", [])
    if issues:
        print(f"[PROVENANCE] Exported with {len(issues)} integrity issue(s): {output_path}")
        for issue in issues:
            print(f"  ⚠  {issue}")
    else:
        print(f"[PROVENANCE] Exported (clean): {output_path}")

    return output_path


# ── Integrity verification ────────────────────────────────────────────────────

def verify_record_integrity(record_id: str) -> dict:
    """
    Verify that a record's reading_history is consistent with the CDC log.

    Cross-checks:
    - Does the CDC have a deposit event for this record?
    - Does the count of fork/confluence events match reading_history length?
    - Is reading_input_hash present for all readings that should have one?
    - Is the full log chain intact?

    Returns {"ok": bool, "issues": list[str]}.
    """
    index = _idx._load_index()
    record = next((r for r in index["records"] if r.get("id") == record_id), None)
    if record is None:
        return {"ok": False, "issues": [f"Record not found: {record_id}"]}

    cdc_events = get_events_for_record(record_id)
    chain_result = verify_chain()
    return _check_integrity(record, cdc_events, chain_result)


def _check_integrity(record: dict, cdc_events: list, chain_result: dict) -> dict:
    """Internal: cross-check record dict + CDC events + chain result."""
    issues = []

    # Does a deposit event exist for this record?
    deposit_events = [e for e in cdc_events if e["event_type"] == "deposit"]
    if not deposit_events:
        issues.append(
            "No CDC deposit event found — record may predate the CDC log. "
            "Consider backfilling with backfill_cdc_for_existing_records()."
        )

    # Are fork/confluence counts consistent with reading_history length?
    mutation_events = [e for e in cdc_events if e["event_type"] in ("fork", "confluence")]
    reading_count = len(record.get("reading_history", []))
    expected_mutations = max(0, reading_count - 1)  # first reading is the deposit itself
    if len(mutation_events) != expected_mutations:
        issues.append(
            f"CDC has {len(mutation_events)} fork/confluence event(s) but "
            f"reading_history has {reading_count} reading(s) "
            f"(expected {expected_mutations} mutation event(s))"
        )

    # Does every reading have a reading_input_hash?
    for i, reading in enumerate(record.get("reading_history", [])):
        if "reading_input_hash" not in reading:
            issues.append(
                f"Reading {i} (computed_at={reading.get('computed_at', '?')}) "
                f"is missing reading_input_hash — predates provenance system. "
                f"Non-critical: future readings will carry this field."
            )

    # Is the hash chain intact?
    if not chain_result["ok"]:
        issues.append(
            f"CDC chain broken at line {chain_result['broken_at_line']}: "
            f"{chain_result['error']}"
        )

    return {"ok": len(issues) == 0, "issues": issues}


# ── Corpus-level provenance ───────────────────────────────────────────────────

def corpus_timeline(limit: Optional[int] = None) -> list:
    """
    Return all CDC events for all records, chronologically.
    Optionally limited to the most recent N events.
    """
    events = read_events()
    if limit is not None:
        events = events[-limit:]
    return events


def corpus_integrity_report() -> dict:
    """
    Run integrity checks across all records in the index.

    Returns a summary dict:
        total_records: int
        clean: int
        with_issues: int
        issues_by_record: dict[record_id → list[str]]
        cdc_chain_ok: bool
        cdc_events_total: int
    """
    index = _idx._load_index()
    chain_result = verify_chain()

    clean = 0
    with_issues = 0
    issues_by_record: dict = {}

    for rec in index.get("records", []):
        record_id = rec.get("id", "")
        cdc_events = get_events_for_record(record_id)
        result = _check_integrity(rec, cdc_events, chain_result)
        if result["ok"]:
            clean += 1
        else:
            with_issues += 1
            issues_by_record[record_id] = result["issues"]

    return {
        "total_records": len(index.get("records", [])),
        "clean": clean,
        "with_issues": with_issues,
        "issues_by_record": issues_by_record,
        "cdc_chain_ok": chain_result["ok"],
        "cdc_events_total": chain_result["events_verified"],
    }


def backfill_cdc_for_existing_records() -> int:
    """
    Backfill CDC deposit events for records that existed before the CDC log.

    Reads all records from the index and appends a synthetic deposit event
    for any record_id that has no existing deposit event in the CDC log.

    This is a one-time migration operation. After running, verify_record_integrity()
    will no longer report missing deposit events for pre-CDC records.

    Returns the count of events written.
    """
    from seedbank.cdc import append_event, read_events as _read_events

    existing_deposit_ids = {
        e["record_id"]
        for e in _read_events(event_type="deposit")
    }

    index = _idx._load_index()
    written = 0
    for rec in index.get("records", []):
        record_id = rec.get("id", "")
        if not record_id or record_id in existing_deposit_ids:
            continue
        append_event(
            "deposit",
            record_id,
            {
                "filename": rec.get("filename", ""),
                "deposited_at": rec.get("deposited_at", ""),
                "baseline_version": rec.get("baseline_version", ""),
                "tags": rec.get("tags", []),
                "backfilled": True,
                "note": "Synthetic CDC event — record predates CDC log",
            },
        )
        written += 1

    print(f"[PROVENANCE] Backfilled {written} deposit event(s) to CDC log.")
    return written

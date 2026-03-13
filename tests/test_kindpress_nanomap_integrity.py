import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kindpress.press import encode
from seedbank.nanomap import (
    append_warning_event,
    validate_kindpress_chain_order,
)


def _make_warning_fingerprint(tmp_path):
    record = {
        "id": "warn-1",
        "deposited_at": "2026-03-13T00:00:00+00:00",
        "filename": "warning.json",
        "context": "runtime_warning_capture",
        "baseline_version": "kindpath:warning:v1",
        "lsii_score": 0.0,
        "lsii_flag_level": "none",
        "authentic_emission_score": 0.0,
        "manufacturing_score": 0.0,
        "creative_residue": 0.0,
        "era_fingerprint": "runtime",
        "key_estimate": "unknown",
        "tempo_bpm": 0.0,
        "genre_estimate": "runtime-warning",
        "tags": ["warning"],
        "verified": False,
    }

    raw_path = tmp_path / "warning_raw.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    packet = encode(record)
    packet_path = tmp_path / "warning_packet.json"
    with open(packet_path, "w", encoding="utf-8") as fh:
        json.dump(packet.to_dict(), fh, indent=2)

    import hashlib

    def sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    return {
        "raw_path": str(raw_path),
        "packet_path": str(packet_path),
        "raw_sha256": sha256_file(raw_path),
        "packet_sha256": sha256_file(packet_path),
        "packet_hash": packet.packet_hash,
    }


def test_warning_fingerprint_tamper_rejected(tmp_path):
    fp = _make_warning_fingerprint(tmp_path)

    # Tamper after fingerprinting to force mismatch.
    with open(fp["raw_path"], "a", encoding="utf-8") as fh:
        fh.write("\nTAMPER\n")

    with pytest.raises(ValueError, match="fingerprint rejected"):
        append_warning_event(
            run_id="run_tamper",
            message="tamper test",
            category="ComplexWarning",
            filename="fingerprints.py",
            lineno=381,
            module="fingerprints",
            fingerprint=fp,
            cdc_file=str(tmp_path / "events.jsonl"),
        )


def test_kindpress_chain_order_valid_case():
    events = [
        {"event_type": "confluence", "data": {"kind": "analysis_started"}},
        {"event_type": "confluence", "data": {"kind": "kindpress_clone"}},
        {"event_type": "warning", "data": {"message": "w"}},
        {"event_type": "confluence", "data": {"kind": "kindpress_extraction_state"}},
        {"event_type": "confluence", "data": {"artifact_name": "analysis_profile"}},
    ]
    ok, issues = validate_kindpress_chain_order(events)
    assert ok is True
    assert issues == []


def test_kindpress_chain_order_detects_profile_before_extract():
    events = [
        {"event_type": "confluence", "data": {"kind": "analysis_started"}},
        {"event_type": "confluence", "data": {"artifact_name": "analysis_profile"}},
        {"event_type": "confluence", "data": {"kind": "kindpress_clone"}},
        {"event_type": "confluence", "data": {"kind": "kindpress_extraction_state"}},
    ]
    ok, issues = validate_kindpress_chain_order(events)
    assert ok is False
    assert any("analysis_profile logged before kindpress_clone" in i or "analysis_profile logged before kindpress_extraction_state" in i for i in issues)

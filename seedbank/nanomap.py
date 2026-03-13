"""
KindPath Analyser :: Seedbank NanoMap

NanoMap treats each warning, artifact, and transformation as a first-class
lineage point. This module appends cryptographic evidence for:

1. Runtime warnings (captured verbatim)
2. Uncompressed analysis artifacts
3. KindPress-compressed clones of the same artifacts
4. Cloud replication intents (queue for external replication workers)

This keeps local truth append-only and tamper-evident while making remote
multi-cloud replication explicit and auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from kindpress.press import KindPressPacket, encode, verify_integrity
from seedbank.cdc import append_event


_QUEUE_FILE = os.path.join(os.path.dirname(__file__), "cloud_replication_queue.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def append_warning_event(
    run_id: str,
    message: str,
    category: str,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
    module: Optional[str] = None,
    fingerprint: Optional[dict] = None,
    cdc_file: Optional[str] = None,
) -> dict:
    """Write one warning as a CDC confluence point with a deterministic hash."""
    if fingerprint:
        valid, reason = validate_warning_fingerprint(fingerprint)
        if not valid:
            raise ValueError(f"warning fingerprint rejected: {reason}")

    payload = {
        "recorded_at": _now_iso(),
        "category": category,
        "message": message,
        "filename": filename,
        "lineno": lineno,
        "module": module,
    }
    if fingerprint:
        payload["fingerprint"] = fingerprint
    payload["warning_hash"] = sha256_bytes(canonical_json_bytes(payload))
    if cdc_file:
        return append_event("warning", run_id, payload, _file=cdc_file)
    return append_event("warning", run_id, payload)


def write_warning_packet(run_id: str, warning_payload: dict, index: int) -> dict:
    """
    Extract warning record into dedicated warning archive before CDC write.

    The archive stores both raw warning JSON and a KindPress packet version,
    allowing warning lineage to be replayed as its own compressed corpus.
    """
    warning_dir = os.path.join(os.path.dirname(__file__), "warning_archive")
    os.makedirs(warning_dir, exist_ok=True)

    message = warning_payload.get("message", "")
    category = warning_payload.get("category", "Warning")
    filename = warning_payload.get("filename")
    lineno = warning_payload.get("lineno")
    module = warning_payload.get("module")

    raw_record = {
        "id": f"{run_id}:warning:{index}",
        "deposited_at": _now_iso(),
        "filename": os.path.basename(filename) if filename else "runtime_warning",
        "context": "runtime_warning_capture",
        "baseline_version": "kindpath:warning:v1",
        "release_circumstances": "analysis_runtime",
        "creator_statement": None,
        "duration_seconds": 0.0,
        "lsii_score": 0.0,
        "lsii_flag_level": "none",
        "authentic_emission_score": 0.0,
        "manufacturing_score": 0.0,
        "creative_residue": 0.0,
        "era_fingerprint": "runtime",
        "key_estimate": "unknown",
        "tempo_bpm": 0.0,
        "genre_estimate": "runtime-warning",
        "tags": ["warning", "nanomap", category.lower()],
        "verified": False,
        "warning_message": message,
        "warning_category": category,
        "warning_source_file": filename,
        "warning_source_line": lineno,
        "warning_module": module,
    }

    base_name = f"{run_id}_warning_{index:03d}"
    raw_path = os.path.join(warning_dir, f"{base_name}_raw.json")
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw_record, fh, indent=2, ensure_ascii=False)

    packet = encode(raw_record)
    packet_path = os.path.join(warning_dir, f"{base_name}_kindpress_packet.json")
    with open(packet_path, "w", encoding="utf-8") as fh:
        json.dump(packet.to_dict(), fh, indent=2, ensure_ascii=False)

    raw_sha = sha256_file(raw_path)
    packet_sha = sha256_file(packet_path)

    return {
        "index": index,
        "raw_path": raw_path,
        "packet_path": packet_path,
        "raw_sha256": raw_sha,
        "packet_sha256": packet_sha,
        "packet_hash": packet.packet_hash,
        "packet_k_version": packet.k_version,
    }


def append_artifact_event(
    run_id: str,
    artifact_name: str,
    *,
    file_path: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    """
    Log a cryptographic artifact record. Accepts either a file path or dict payload.
    """
    if file_path:
        digest = sha256_file(file_path)
        size = os.path.getsize(file_path)
        source = {
            "artifact_name": artifact_name,
            "path": file_path,
            "sha256": digest,
            "size_bytes": size,
            "kind": "file",
        }
    elif payload is not None:
        raw = canonical_json_bytes(payload)
        digest = sha256_bytes(raw)
        source = {
            "artifact_name": artifact_name,
            "sha256": digest,
            "size_bytes": len(raw),
            "kind": "json",
        }
    else:
        raise ValueError("append_artifact_event requires file_path or payload")

    source["recorded_at"] = _now_iso()
    source["storage_scope"] = "local_verified"
    return append_event("confluence", run_id, source)


def queue_cloud_replication(
    run_id: str,
    artifact_name: str,
    artifact_sha256: str,
    artifact_ref: str,
    cloud_targets: Optional[list[str]] = None,
) -> dict:
    """
    Queue replication intent for external workers (e.g. GCS/S3/B2).
    This module does not perform network writes directly.
    """
    targets = cloud_targets or ["gcs", "s3", "b2"]
    entry = {
        "queued_at": _now_iso(),
        "run_id": run_id,
        "artifact_name": artifact_name,
        "artifact_sha256": artifact_sha256,
        "artifact_ref": artifact_ref,
        "targets": targets,
        "status": "pending",
    }
    os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
    with open(_QUEUE_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


def write_kindpress_clone(run_id: str, profile: dict) -> dict:
    """
    Persist uncompressed profile summary + KindPress packet side-by-side,
    then log both cryptographically.
    """
    clone_dir = os.path.join(os.path.dirname(__file__), "kindpress_clones")
    os.makedirs(clone_dir, exist_ok=True)

    metadata = profile.get("metadata", {})
    lsii = profile.get("lsii", {})
    fp = profile.get("fingerprints", {})
    psy = profile.get("psychosomatic", {})

    # Construct a deterministic record shape for KindPress compression.
    source_record = {
        "id": run_id,
        "deposited_at": _now_iso(),
        "filename": metadata.get("filename", "unknown"),
        "context": "runtime_analysis",
        "baseline_version": "kindpath:runtime:v1",
        "release_circumstances": "runtime_capture",
        "creator_statement": None,
        "duration_seconds": metadata.get("duration_seconds"),
        "lsii_score": lsii.get("score"),
        "lsii_flag_level": lsii.get("flag_level"),
        "authentic_emission_score": psy.get("authentic_emission_score"),
        "manufacturing_score": psy.get("manufacturing_score"),
        "creative_residue": psy.get("creative_residue"),
        "era_fingerprint": (fp.get("era_matches") or [{}])[0].get("name", "unknown"),
        "key_estimate": "unknown",
        "tempo_bpm": None,
        "genre_estimate": "unknown",
        "tags": ["runtime", "nanomap"],
        "verified": False,
        "profile_sha256": sha256_bytes(canonical_json_bytes(profile)),
    }

    raw_path = os.path.join(clone_dir, f"{run_id}_uncompressed.json")
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(source_record, fh, indent=2, ensure_ascii=False)

    packet = encode(source_record)
    packet_path = os.path.join(clone_dir, f"{run_id}_kindpress_packet.json")
    with open(packet_path, "w", encoding="utf-8") as fh:
        json.dump(packet.to_dict(), fh, indent=2, ensure_ascii=False)

    raw_event = append_artifact_event(run_id, "uncompressed_record", file_path=raw_path)
    packet_event = append_artifact_event(run_id, "kindpress_packet", file_path=packet_path)

    queue_cloud_replication(run_id, "uncompressed_record", raw_event["data"]["sha256"], raw_path)
    queue_cloud_replication(run_id, "kindpress_packet", packet_event["data"]["sha256"], packet_path)

    append_event(
        "confluence",
        run_id,
        {
            "kind": "kindpress_clone",
            "raw_path": raw_path,
            "packet_path": packet_path,
            "raw_sha256": raw_event["data"]["sha256"],
            "packet_sha256": packet_event["data"]["sha256"],
        },
    )

    return {
        "raw_path": raw_path,
        "packet_path": packet_path,
        "raw_sha256": raw_event["data"]["sha256"],
        "packet_sha256": packet_event["data"]["sha256"],
        "packet_hash": packet.packet_hash,
        "packet_k_version": packet.k_version,
    }


def build_kindpress_extraction_state(
    run_id: str,
    profile: dict,
    clone_meta: dict,
    warning_fingerprints: Optional[list[dict]] = None,
) -> dict:
    """
    Build explicit extraction-state metadata for whole and fragmented datasets.

    whole_dataset: profile-level canonical hash + primary KindPress packet details.
    fragmented_datasets: per-section hashes and warning packet fragments.
    """
    warning_fingerprints = warning_fingerprints or []
    profile_hash = sha256_bytes(canonical_json_bytes(profile))

    fragmented = []
    for section_key in (
        "lsii",
        "trajectory",
        "fingerprints",
        "psychosomatic",
        "influence_chain",
    ):
        section_value = profile.get(section_key)
        if section_value is None:
            continue
        section_hash = sha256_bytes(canonical_json_bytes(section_value))
        fragmented.append(
            {
                "name": section_key,
                "kind": "profile_section",
                "sha256": section_hash,
                "size_bytes": len(canonical_json_bytes(section_value)),
            }
        )

    for wf in warning_fingerprints:
        fragmented.append(
            {
                "name": f"warning_{wf.get('index', 0):03d}",
                "kind": "warning_packet",
                "raw_sha256": wf.get("raw_sha256"),
                "packet_sha256": wf.get("packet_sha256"),
                "packet_hash": wf.get("packet_hash"),
                "packet_k_version": wf.get("packet_k_version"),
            }
        )

    return {
        "run_id": run_id,
        "recorded_at": _now_iso(),
        "whole_dataset": {
            "profile_sha256": profile_hash,
            "kindpress_packet_sha256": clone_meta.get("packet_sha256"),
            "kindpress_packet_hash": clone_meta.get("packet_hash"),
            "kindpress_k_version": clone_meta.get("packet_k_version"),
            "kindpress_packet_path": clone_meta.get("packet_path"),
        },
        "fragmented_datasets": fragmented,
        "fragment_count": len(fragmented),
        "decryption_pointer": {
            "k_version": clone_meta.get("packet_k_version"),
            "packet_hash": clone_meta.get("packet_hash"),
            "note": "KindPress packet hash + k_version identify required reconstruction context.",
        },
    }


def append_kindpress_extraction_state(run_id: str, extraction_state: dict) -> dict:
    """Persist extraction state as a dedicated confluence event for provenance."""
    return append_event(
        "confluence",
        run_id,
        {
            "kind": "kindpress_extraction_state",
            "state": extraction_state,
        },
    )


def validate_warning_fingerprint(fingerprint: dict) -> tuple[bool, str]:
    """
    Validate warning fingerprint integrity before event acceptance.

    Checks:
    - required fields present
    - raw and packet files exist
    - file SHA-256 matches declared SHA-256
    - KindPress packet hash verifies
    """
    required = ["raw_path", "packet_path", "raw_sha256", "packet_sha256", "packet_hash"]
    missing = [k for k in required if not fingerprint.get(k)]
    if missing:
        return False, f"missing fields: {', '.join(missing)}"

    raw_path = fingerprint["raw_path"]
    packet_path = fingerprint["packet_path"]
    if not os.path.exists(raw_path):
        return False, f"raw file not found: {raw_path}"
    if not os.path.exists(packet_path):
        return False, f"packet file not found: {packet_path}"

    raw_sha = sha256_file(raw_path)
    packet_sha = sha256_file(packet_path)
    if raw_sha != fingerprint["raw_sha256"]:
        return False, "raw_sha256 mismatch"
    if packet_sha != fingerprint["packet_sha256"]:
        return False, "packet_sha256 mismatch"

    try:
        with open(packet_path, encoding="utf-8") as fh:
            packet_dict = json.load(fh)
        packet = KindPressPacket.from_dict(packet_dict)
    except Exception as exc:
        return False, f"invalid packet json: {exc}"

    if packet.packet_hash != fingerprint["packet_hash"]:
        return False, "packet_hash mismatch"
    if not verify_integrity(packet):
        return False, "kindpress integrity check failed"

    return True, "ok"


def validate_kindpress_chain_order(events: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate extract-first ordering for one run's event list.

    Required ordering constraints:
    1) kindpress_clone event before warning events
    2) kindpress_clone event before analysis_profile event
    3) kindpress_extraction_state event before analysis_profile event
    """
    issues: list[str] = []
    idx_clone = None
    idx_extract_state = None
    idx_profile = None
    idx_warnings: list[int] = []

    for i, e in enumerate(events):
        et = e.get("event_type")
        data = e.get("data", {})
        kind = data.get("kind")

        if et == "warning":
            idx_warnings.append(i)
        if et == "confluence" and kind == "kindpress_clone":
            idx_clone = i
        if et == "confluence" and kind == "kindpress_extraction_state":
            idx_extract_state = i
        if et == "confluence" and data.get("artifact_name") == "analysis_profile":
            idx_profile = i

    if idx_clone is None:
        issues.append("missing kindpress_clone event")
    if idx_extract_state is None:
        issues.append("missing kindpress_extraction_state event")
    if idx_profile is None:
        issues.append("missing analysis_profile artifact event")

    if idx_clone is not None:
        for w in idx_warnings:
            if idx_clone > w:
                issues.append("warning logged before kindpress_clone")
                break
        if idx_profile is not None and idx_clone > idx_profile:
            issues.append("analysis_profile logged before kindpress_clone")

    if idx_extract_state is not None and idx_profile is not None and idx_extract_state > idx_profile:
        issues.append("analysis_profile logged before kindpress_extraction_state")

    return (len(issues) == 0, issues)

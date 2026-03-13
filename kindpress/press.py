"""
KindPress :: press.py

Semantic encode/decode for KindPath records.

A record compressed by KindPress is split into two parts:

    k_version  — a pointer to the shared constants bank (the tag baseline
                  snapshot in use when the record was deposited or last recomputed)
    delta      — the fields that carry per-record signal: what this record IS,
                  beyond what k already knows

The full picture is always recoverable:
    record = k_defaults(k_version) + delta

Transmission economy:
    If both sender and receiver hold the same k-version, only delta needs to
    travel. The receiver reconstructs the full record locally.
    k-version mismatch is detected before decode — the receiver knows exactly
    which constants have drifted and can flag the divergence rather than
    silently producing a miscalibrated picture.

This is the same operation the human brain performs:
    Episodic memory (Δ) is encoded against semantic memory (k).
    When semantic memory is updated, old episodic memories re-express —
    the same event carries new meaning under the revised context.
    The delta between old and new expression IS the learning signal.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

# Fields that belong to k (shared context / structure) — not carried in Δ.
# These are either the k_version pointer itself, or record identity fields
# that have no analytical content.
_K_STRUCTURAL_FIELDS = frozenset({
    "id",
    "deposited_at",
    "compressed_at",
    "filename",        # Identity: which piece — not analytical signal
    "context",         # Creator-provided context — not derivable from k
    "baseline_version",        # k pointer itself
    "reconstruction_protocol", # How Δ was derived — methodology, not signal
    "reading_history",         # Full history — too large; carried separately
    "stale_tags",              # Operational flag — not signal
    "creative_residue_stale",  # Operational flag — not signal
})

# Δ fields: the per-record analytical fingerprint.
# These are the values a reasoner works with.
_DELTA_FIELDS = (
    "lsii_score",
    "lsii_flag_level",
    "authentic_emission_score",
    "manufacturing_score",
    "creative_residue",
    "era_fingerprint",
    "key_estimate",
    "tempo_bpm",
    "genre_estimate",
    "tags",
    "verified",
    "release_circumstances",
    "creator_statement",
    "duration_seconds",
)


@dataclass
class KindPressPacket:
    """
    A meaning-compressed record: k-version pointer + Δ.

    This is the unit of transmission in the KindPress protocol.
    Δ is the fingerprint — what is distinct about this record.
    k_version is the shared baseline required to decompress it.

    Both endpoints must be k-aligned for decode() to produce a
    semantically equivalent result to the original. If k has drifted
    between compress and decode, k_alignment_check() will surface the gap.

    packet_hash: SHA-256 of (k_version + sorted Δ JSON) — integrity check.
    Any corruption of Δ OR k-version mismatch produces a different hash.
    """
    record_id: str
    k_version: str               # baseline_version string — the constants pointer
    delta: dict                  # Per-record signal: what k doesn't already know
    compressed_at: str           # ISO 8601 timestamp of compression
    packet_hash: str             # Integrity: SHA-256(k_version + delta)
    source_filename: str = ""    # For human legibility only — not used in decode

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KindPressPacket":
        return cls(**d)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_hash(k_version: str, delta: dict) -> str:
    """Deterministic SHA-256 of k_version + canonically serialised delta."""
    canonical = k_version + json.dumps(delta, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encode(record: dict) -> KindPressPacket:
    """
    Compress a seedbank record into a KindPress packet.

    Extracts only the Δ fields — the analytical fingerprint — and pairs
    them with the k_version pointer (baseline_version). The full record
    is recoverable by anyone holding the matching k.

    Unrecognised fields (not in _DELTA_FIELDS and not in _K_STRUCTURAL_FIELDS)
    are preserved in delta under a 'extra' key — nothing is silently dropped.
    """
    k_version = record.get("baseline_version", "")

    delta: dict = {}
    for field_name in _DELTA_FIELDS:
        val = record.get(field_name)
        if val is not None:
            delta[field_name] = val

    # Preserve any extra fields that aren't in our known sets
    known = _K_STRUCTURAL_FIELDS | set(_DELTA_FIELDS)
    extra = {k: v for k, v in record.items() if k not in known and v is not None}
    if extra:
        delta["__extra__"] = extra

    return KindPressPacket(
        record_id=record.get("id", ""),
        k_version=k_version,
        delta=delta,
        compressed_at=_now_iso(),
        packet_hash=_compute_hash(k_version, delta),
        source_filename=record.get("filename", ""),
    )


def decode(packet: KindPressPacket, k_defaults: Optional[dict] = None) -> dict:
    """
    Reconstruct a full record from a KindPress packet.

    k_defaults: the shared baseline fields for the packet's k_version.
    These are the fields k contributes — Δ then overrides and extends.

    If k_defaults is None, the reconstructed record is Δ-only — it is
    analytically valid but missing the k-layer context. The caller should
    treat it as a partial record and flag k_defaults as required for full
    reconstruction.

    The packet_hash is NOT re-verified here — call verify_integrity()
    separately if provenance checking is needed.
    """
    base = dict(k_defaults) if k_defaults else {}

    # Merge: k provides defaults, Δ provides per-record values.
    # Δ always wins where both exist — it is the record's specific fingerprint.
    extra = packet.delta.pop("__extra__", {}) if "__extra__" in packet.delta else {}
    base.update(packet.delta)
    base.update(extra)
    base["id"] = packet.record_id
    base["baseline_version"] = packet.k_version

    # Re-insert __extra__ into packet.delta (non-destructive)
    if extra:
        packet.delta["__extra__"] = extra

    return base


def verify_integrity(packet: KindPressPacket) -> bool:
    """
    Verify the packet's delta and k_version haven't been altered since compression.
    Returns True if the hash matches, False if corruption or tampering is detected.
    """
    expected = _compute_hash(packet.k_version, packet.delta)
    return packet.packet_hash == expected


def k_alignment_check(
    packet_k_version: str,
    available_k_version: str,
) -> dict:
    """
    Check whether a receiver's k aligns with the k a packet was compressed against.

    Returns a report dict:
        aligned: bool
        drifted_tags: list of tag names where versions differ
        packet_tags: dict of tag_name -> version from packet k
        available_tags: dict of tag_name -> version from available k
        recommendation: str

    Aligned = both k-versions encode identical tag-version pairs.
    Drifted = one or more tags have been revised between the two k-versions.

    Drifted packets can still be decoded — the decoded record will reflect
    the k-version it was compressed under, not the receiver's current k.
    Callers should decide whether to recompute or accept the semantic drift.
    """
    def _parse_k(k_str: str) -> dict:
        """Parse 'tag:v2,other:v1' → {'tag': 2, 'other': 1}"""
        result = {}
        if not k_str:
            return result
        for part in k_str.split(","):
            part = part.strip()
            if ":v" in part:
                name, ver = part.rsplit(":v", 1)
                try:
                    result[name] = int(ver)
                except ValueError:
                    result[name] = 0
        return result

    packet_tags = _parse_k(packet_k_version)
    available_tags = _parse_k(available_k_version)

    all_tag_names = set(packet_tags) | set(available_tags)
    drifted = []
    for tag in sorted(all_tag_names):
        pv = packet_tags.get(tag, 0)
        av = available_tags.get(tag, 0)
        if pv != av:
            drifted.append(tag)

    aligned = len(drifted) == 0
    if aligned:
        recommendation = "k-aligned. Decode will produce a semantically equivalent record."
    else:
        recommendation = (
            f"{len(drifted)} tag(s) have drifted: {', '.join(drifted)}. "
            "Decode will produce a record encoded under the packet's k, not the "
            "receiver's current k. Consider recomputing stale records before using "
            "the decoded values as current-universe readings."
        )

    return {
        "aligned": aligned,
        "drifted_tags": drifted,
        "packet_tags": packet_tags,
        "available_tags": available_tags,
        "recommendation": recommendation,
    }


def delta_size_bytes(packet: KindPressPacket) -> int:
    """Return the byte size of the Δ payload (JSON-serialised)."""
    return len(json.dumps(packet.delta, ensure_ascii=False).encode("utf-8"))


def compression_ratio(record: dict, packet: KindPressPacket) -> float:
    """
    Ratio of delta size to full record size (excluding reading_history).
    A well-calibrated k produces a small ratio — most meaning is shared.
    A miscalibrated k produces a large ratio — k captures little, Δ carries most.

    Ratio < 0.5: k is capturing more than half the record's information.
    Ratio > 0.8: k is capturing very little — consider revising.
    """
    full_without_history = {k: v for k, v in record.items() if k != "reading_history"}
    full_size = len(json.dumps(full_without_history, ensure_ascii=False).encode("utf-8"))
    d_size = delta_size_bytes(packet)
    if full_size == 0:
        return 1.0
    return round(d_size / full_size, 4)

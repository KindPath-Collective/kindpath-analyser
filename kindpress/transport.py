"""
KindPress :: transport.py

Transport-oriented encoding profile.

When a seedbank record needs to travel across a wire (or between systems),
the full JSON payload is unnecessarily large. Most of the information in
a Δ vector is sub-noise-floor variance — storage-grade fidelity with
fractions-of-a-percent precision is not needed in transit.

This module provides a lossy-but-legible transport encoding:
- Drops Δ values below the noise floor (configurable threshold)
- Quantises remaining values to integer steps (default: 15-bit signed)
- Selectively includes/excludes zero-delta fields
- Benchmarks compression ratio vs. reconstruction fidelity

The transport layer is intentionally separate from press.py's full
encode/decode, which is the lossless storage API. This is the postcard
format: smaller, sufficient, clearly marked as transport-grade.

Transport payloads are NOT a replacement for seedbank deposits. They are
point-in-time snapshots for inter-system synchronisation and streaming
data feeds. Always store the full record; use transport for transmission.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Transport profile ─────────────────────────────────────────────────────────

@dataclass
class TransportProfile:
    """
    Configuration for a transport-grade encoding.

    noise_floor: Absolute Δ values below this are treated as zero and dropped
                 (unless include_zero_deltas=True). Default 0.002 ≈ 0.2% of scale.

    quantise_bits: Number of bits for integer quantisation.
                   15 bits → 32767 steps → worst-case quantisation error ≈ 0.003%.
                   8 bits → 127 steps → worst-case quantisation error ≈ 0.4%.

    include_zero_deltas: Whether to include sub-noise-floor fields in the payload.
                         False = smaller payload; True = fully reconstructable to baseline.

    compress_keys: Shorten JSON keys to 2-letter codes on the wire (see _KEY_SHORT).
                   Saves 30–50% on key overhead for small payloads.
    """
    noise_floor: float = 0.002
    quantise_bits: int = 15         # Signed: range ±(2^(bits-1) - 1)
    include_zero_deltas: bool = False
    compress_keys: bool = False


# Three ready-made profiles covering the common cases
DEFAULT_PROFILE = TransportProfile()
HIGH_FIDELITY_PROFILE = TransportProfile(noise_floor=0.0001, quantise_bits=16)
MINIMAL_PROFILE = TransportProfile(noise_floor=0.01, quantise_bits=8, compress_keys=True)


# ── Key compression map ───────────────────────────────────────────────────────
# Maps long psychosomatic/lsii field names → 2-letter wire tokens.
# Expand this list as new top-level features are added to the JSON report.

_KEY_SHORT: dict[str, str] = {
    "creative_residue":          "cr",
    "authentic_emission_score":  "ae",
    "manufacturing_score":       "ms",
    "score":                     "li",   # lsii.score
    "valence":                   "va",
    "arousal":                   "ar",
    "coherence":                 "co",
    "complexity":                "cx",
    "tension_resolution_ratio":  "tr",
    "identity_capture_risk":     "ir",
    "stage3_tag_risk":           "st",
    "authenticity_index":        "ai",
}
_KEY_LONG: dict[str, str] = {v: k for k, v in _KEY_SHORT.items()}


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_for_transport(
    record_dict: dict,
    baseline_dict: dict,
    profile: TransportProfile = DEFAULT_PROFILE,
) -> dict:
    """
    Encode a record's delta values for transport.

    record_dict:   a JSON report dict (from generate_json_report())
    baseline_dict: the baseline means for the relevant tag cluster,
                   keyed by feature name (matches psychosomatic + lsii sub-dicts)
    profile:       TransportProfile controlling fidelity/size tradeoff

    Returns a transport payload: fully JSON-serialisable, substantially smaller
    than the full record. The payload contains only the delta from baseline, not
    the full record — the receiver reconstructs absolute values via decode_from_transport().

    The _quantise_bits and _noise_floor fields in the payload are the keys needed
    to reconstruct correctly; do not strip them.
    """
    steps = (2 ** (profile.quantise_bits - 1)) - 1   # e.g. 15-bit → 16383

    psych = record_dict.get("psychosomatic", {})
    lsii_doc = record_dict.get("lsii", {})

    # Gather raw deltas from all numeric baseline keys
    raw_deltas: dict[str, float] = {}
    for key, baseline_val in baseline_dict.items():
        if key.startswith("_"):
            continue
        record_val = psych.get(key)
        if record_val is None:
            record_val = lsii_doc.get(key)
        if record_val is None:
            continue
        try:
            raw_deltas[key] = float(record_val) - float(baseline_val)
        except (TypeError, ValueError):
            pass

    # Quantise and apply noise floor
    payload_deltas: dict[str, int] = {}
    for key, delta in raw_deltas.items():
        if not profile.include_zero_deltas and abs(delta) < profile.noise_floor:
            continue
        quantised = int(round(delta * steps))
        quantised = max(-steps, min(steps, quantised))   # clamp to representable range
        wire_key = _KEY_SHORT.get(key, key) if profile.compress_keys else key
        payload_deltas[wire_key] = quantised

    return {
        "_transport_version": 1,
        "_quantise_bits": profile.quantise_bits,
        "_noise_floor": profile.noise_floor,
        "_include_zero_deltas": profile.include_zero_deltas,
        "_compressed_keys": profile.compress_keys,
        "source_id": record_dict.get("source", {}).get("filepath", ""),
        "generated_at": record_dict.get("generated_at", ""),
        "tag_baseline_version": baseline_dict.get("_tag_version", "unknown"),
        "deltas": payload_deltas,
    }


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_from_transport(
    payload: dict,
    baseline_dict: dict,
) -> dict:
    """
    Reconstruct approximate record values from a transport payload.

    Returns a flat dict of feature_name → reconstructed_value, covering all
    features present in baseline_dict. Features absent from the payload
    (sub-noise-floor during encoding) are restored as the baseline value exactly.

    The reconstructed values are approximate; maximum error is bounded by
    the quantisation step size defined by _quantise_bits in the payload.
    """
    bits = payload.get("_quantise_bits", 15)
    steps = (2 ** (bits - 1)) - 1
    compressed = payload.get("_compressed_keys", False)

    # Decompress wire keys if needed
    deltas: dict[str, int] = {}
    for k, v in payload.get("deltas", {}).items():
        long_key = _KEY_LONG.get(k, k) if compressed else k
        deltas[long_key] = v

    reconstructed: dict[str, float] = {}
    for key, baseline_val in baseline_dict.items():
        if key.startswith("_"):
            continue
        try:
            bval = float(baseline_val)
        except (TypeError, ValueError):
            continue

        if key in deltas:
            reconstructed[key] = bval + (deltas[key] / steps)
        else:
            # Field was below noise floor at encoding time — restore exactly as baseline
            reconstructed[key] = bval

    return reconstructed


# ── Benchmarking ──────────────────────────────────────────────────────────────

def benchmark_transport(
    records: list[dict],
    baseline_dict: dict,
    profiles: Optional[list[TransportProfile]] = None,
) -> dict:
    """
    Benchmark transport profiles across a set of records.

    records:      list of JSON report dicts (generate_json_report() output)
    baseline_dict: baseline for the tag cluster
    profiles:     list of TransportProfile to compare;
                  defaults to [DEFAULT_PROFILE, HIGH_FIDELITY_PROFILE, MINIMAL_PROFILE]

    Returns a benchmark summary dict keyed by a profile label, each entry containing:
        payload_bytes_mean / _min / _max — payload wire size in bytes
        decode_latency_ms_mean          — mean round-trip decode time
        reconstruction_error_mean       — mean absolute difference from true value
        vs_json_ratio                   — compression ratio vs full JSON serialisation
    """
    if profiles is None:
        profiles = [DEFAULT_PROFILE, HIGH_FIDELITY_PROFILE, MINIMAL_PROFILE]

    results: dict = {}

    for profile in profiles:
        label = (
            f"{profile.quantise_bits}bit"
            f"_nf{profile.noise_floor}"
            + ("_compressed" if profile.compress_keys else "")
        )

        payload_sizes: list[int] = []
        latencies: list[float] = []
        errors: list[float] = []

        for record in records:
            payload = encode_for_transport(record, baseline_dict, profile)
            payload_json = json.dumps(payload)
            payload_sizes.append(len(payload_json.encode("utf-8")))

            t0 = time.perf_counter()
            decoded = decode_from_transport(payload, baseline_dict)
            latencies.append((time.perf_counter() - t0) * 1000)

            psych = record.get("psychosomatic", {})
            lsii_doc = record.get("lsii", {})
            for key, decoded_val in decoded.items():
                orig_val = psych.get(key)
                if orig_val is None:
                    orig_val = lsii_doc.get(key)
                if orig_val is not None:
                    try:
                        errors.append(abs(float(orig_val) - decoded_val))
                    except (TypeError, ValueError):
                        pass

        full_sizes = [len(json.dumps(r).encode("utf-8")) for r in records]
        mean_full = sum(full_sizes) / len(full_sizes) if full_sizes else 1
        mean_payload = sum(payload_sizes) / len(payload_sizes) if payload_sizes else 0

        results[label] = {
            "payload_bytes_mean": round(mean_payload),
            "payload_bytes_min": min(payload_sizes) if payload_sizes else 0,
            "payload_bytes_max": max(payload_sizes) if payload_sizes else 0,
            "decode_latency_ms_mean": round(sum(latencies) / len(latencies), 4) if latencies else 0,
            "reconstruction_error_mean": round(sum(errors) / len(errors), 6) if errors else 0,
            "vs_json_ratio": round(mean_payload / mean_full, 3) if mean_full else 1.0,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# ── File Sealing — AES-256-GCM encryption for arbitrary payloads ─────────
# ════════════════════════════════════════════════════════════════════════════
#
# This section is the IP protection layer. It seals arbitrary byte payloads
# (files, records, provenance bundles) into tamper-evident, encrypted archives.
#
# Architecture:
#   SealedBundle = {metadata, content_hash, nonce, ciphertext}
#   The content_hash (SHA-256 of plaintext) is computed before encryption and
#   verified after decryption. A failed verification means either the wrong
#   key was used, or the ciphertext was tampered with (AES-GCM also authenticates,
#   so any bit-flip in the ciphertext will raise an error before we even reach
#   the hash check).
#
# On the two-key architecture (long-term, community data ownership hubs):
#   KindPress already models shared knowledge as k (constants) and individual
#   signal as Δ (delta). The natural encryption counterpart is:
#       Key 1 (k-key): derived from the shared constants baseline snapshot.
#           Held by anyone with the same baseline — the community/hub key.
#       Key 2 (Δ-key): derived from the record's individual delta vector.
#           Unique per record — the individual sovereignty key.
#       Combined key = HKDF(k-key || Δ-key)
#       Decryption requires both. Neither alone is sufficient.
#   This is not yet deployed infrastructure, but derive_dual_key() below
#   is a working prototype of the derivation step. The community data ownership
#   hubs in the pilots will hold k-keys. Individuals will retain Δ-keys.
#   The merge point, and KindPress itself, is the interface between them.
#
# Dependency:
#   `cryptography` package (pip install cryptography>=41.0.0)
#   Optional — sealing functions raise ImportError with install instructions
#   if cryptography is not available. All non-sealing transport functions work
#   without it.

import base64
import hashlib
import os as _os
from dataclasses import dataclass as _dataclass, field as _field
from typing import Optional as _Optional


@_dataclass
class SealedBundle:
    """
    An encrypted, tamper-evident archive of an arbitrary plaintext payload.

    Fields:
        bundle_version:  format version (currently 1)
        sealed_at:       ISO 8601 UTC timestamp of sealing
        content_type:    "file" | "record" | "provenance_bundle" | "arbitrary"
        content_hash:    SHA-256 hex of the plaintext (before encryption)
        nonce_b64:       Base64-encoded 12-byte AES-GCM nonce
        ciphertext_b64:  Base64-encoded AES-256-GCM ciphertext (includes auth tag)
        metadata:        arbitrary dict (filename, record_id, etc.) — stored plaintext

    The bundle is self-describing: a receiver can verify the content_type and
    metadata before attempting decryption, and verify content_hash after.

    JSON serialisation: to_json() / from_json() handle the base64 fields.
    """
    bundle_version: int
    sealed_at: str
    content_type: str
    content_hash: str
    nonce_b64: str
    ciphertext_b64: str
    metadata: dict = _field(default_factory=dict)

    def to_json(self) -> str:
        import json as _json
        return _json.dumps(
            {
                "bundle_version": self.bundle_version,
                "sealed_at": self.sealed_at,
                "content_type": self.content_type,
                "content_hash": self.content_hash,
                "nonce_b64": self.nonce_b64,
                "ciphertext_b64": self.ciphertext_b64,
                "metadata": self.metadata,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "SealedBundle":
        import json as _json
        d = _json.loads(s)
        return cls(
            bundle_version=d["bundle_version"],
            sealed_at=d["sealed_at"],
            content_type=d["content_type"],
            content_hash=d["content_hash"],
            nonce_b64=d["nonce_b64"],
            ciphertext_b64=d["ciphertext_b64"],
            metadata=d.get("metadata", {}),
        )


def _require_cryptography():
    """Raise a helpful ImportError if the cryptography package is not installed."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'cryptography' package is required for sealing operations. "
            "Install it with: pip install cryptography>=41.0.0"
        )


# ── Key generation and derivation ────────────────────────────────────────────

def generate_key() -> bytes:
    """
    Generate a cryptographically random 32-byte AES-256 key.
    Store this securely — it is the only way to unseal a bundle.
    """
    return _os.urandom(32)


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte AES-256 key from a passphrase using PBKDF2-HMAC-SHA256.

    salt: must be random (16+ bytes) and stored alongside the sealed bundle.
          Use os.urandom(16) to generate. The same passphrase + salt always
          produces the same key.

    Iteration count: 600_000 (NIST 2023 recommendation for PBKDF2-SHA256).
    """
    import hashlib as _hashlib
    return _hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=600_000,
        dklen=32,
    )


def derive_dual_key(k_baseline: dict, delta: dict) -> tuple:
    """
    Prototype: derive the dual-key pair from KindPress k and Δ structures.

    This is the cryptographic counterpart to KindPress's semantic model:
        k  = shared constants (what the community knows collectively)
        Δ  = individual delta (what is irreducibly unique to this record)

    Key 1 (k-key): SHA-256 of canonical JSON of the baseline constants.
    Key 2 (Δ-key): SHA-256 of canonical JSON of the individual delta vector.

    Combined key (for sealing): HKDF over (k-key XOR Δ-key).
    Neither key alone produces the combined key — both are required.

    Returns (k_key: bytes, delta_key: bytes, combined_key: bytes).

    In the community ownership hub model:
        The hub distributes k-keys to members who hold the shared constants.
        Individuals retain Δ-keys for their own records.
        Sealing requires the combined key; unsealing requires knowing both halves.
        Neither the hub nor the individual can unseal the other's records alone.
    """
    import json as _json

    k_key = hashlib.sha256(
        _json.dumps(k_baseline, sort_keys=True).encode("utf-8")
    ).digest()

    delta_key = hashlib.sha256(
        _json.dumps(delta, sort_keys=True).encode("utf-8")
    ).digest()

    # XOR the two 32-byte keys to produce the combined key.
    # HKDF would be more robust; XOR is used here for prototyping transparency.
    # Replace with HKDF from cryptography.hazmat.primitives.kdf.hkdf for production.
    combined_key = bytes(a ^ b for a, b in zip(k_key, delta_key))

    return k_key, delta_key, combined_key


# ── Core seal / unseal ────────────────────────────────────────────────────────

def seal(
    payload: bytes,
    key: bytes,
    content_type: str = "arbitrary",
    metadata: _Optional[dict] = None,
) -> SealedBundle:
    """
    Seal an arbitrary byte payload with AES-256-GCM.

    key:          32-byte AES key (from generate_key() or derive_key())
    content_type: human-readable descriptor ("file", "record", "provenance_bundle", etc.)
    metadata:     arbitrary plaintext metadata stored alongside the ciphertext

    AES-GCM provides both confidentiality and authenticity — any tampering with
    the ciphertext will cause unseal() to raise an error before the plaintext
    is returned. The content_hash is an additional layer: post-decryption
    verification that the correct key was used and no silent truncation occurred.

    Raises ImportError if the cryptography package is not installed.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import time as _time

    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes for AES-256; got {len(key)} bytes.")

    nonce = _os.urandom(12)  # 96-bit nonce — GCM standard

    content_hash = hashlib.sha256(payload).hexdigest()
    ciphertext = AESGCM(key).encrypt(nonce, payload, None)

    return SealedBundle(
        bundle_version=1,
        sealed_at=_import_datetime_utc(),
        content_type=content_type,
        content_hash=content_hash,
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
        metadata=metadata or {},
    )


def unseal(bundle: SealedBundle, key: bytes) -> bytes:
    """
    Decrypt a SealedBundle and verify its content hash.

    Returns the plaintext bytes.

    Raises:
        ImportError   — if cryptography is not installed
        ValueError    — if the content_hash does not match after decryption
                        (wrong key, truncation, or silent corruption)
        cryptography.exceptions.InvalidTag — if AES-GCM authentication fails
                        (ciphertext tampered with)
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes for AES-256; got {len(key)} bytes.")

    nonce = base64.b64decode(bundle.nonce_b64)
    ciphertext = base64.b64decode(bundle.ciphertext_b64)

    # AES-GCM decryption: raises InvalidTag if ciphertext was altered
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)

    # Verify content hash as a second layer of integrity assurance
    actual_hash = hashlib.sha256(plaintext).hexdigest()
    if actual_hash != bundle.content_hash:
        raise ValueError(
            f"Content hash mismatch after decryption: "
            f"expected {bundle.content_hash[:12]}… "
            f"got {actual_hash[:12]}…"
        )

    return plaintext


# ── Convenience wrappers ──────────────────────────────────────────────────────

def seal_file(
    filepath: str,
    key: bytes,
    output_path: _Optional[str] = None,
) -> SealedBundle:
    """
    Seal an arbitrary file.

    Reads filepath, encrypts, writes the SealedBundle JSON to output_path
    (default: filepath + '.sealed'). Returns the SealedBundle.

    The original file is not modified or deleted.
    """
    with open(filepath, "rb") as fh:
        payload = fh.read()

    filename = _os.path.basename(filepath)
    bundle = seal(
        payload,
        key,
        content_type="file",
        metadata={"filename": filename, "original_size_bytes": len(payload)},
    )

    dest = output_path or (filepath + ".sealed")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(bundle.to_json())

    print(f"[SEAL] {filename} → {dest}  ({len(payload):,} bytes → {len(base64.b64decode(bundle.ciphertext_b64)):,} ciphertext bytes)")
    return bundle


def unseal_file(
    sealed_path: str,
    key: bytes,
    output_path: _Optional[str] = None,
) -> bytes:
    """
    Unseal a file previously sealed by seal_file().

    Reads the .sealed JSON, decrypts, writes plaintext to output_path
    (default: sealed_path with '.sealed' stripped, or + '.decrypted').
    Returns the plaintext bytes.
    """
    with open(sealed_path, encoding="utf-8") as fh:
        bundle = SealedBundle.from_json(fh.read())

    plaintext = unseal(bundle, key)

    if output_path is None:
        if sealed_path.endswith(".sealed"):
            output_path = sealed_path[: -len(".sealed")]
        else:
            output_path = sealed_path + ".decrypted"

    with open(output_path, "wb") as fh:
        fh.write(plaintext)

    print(f"[UNSEAL] {sealed_path} → {output_path}")
    return plaintext


def seal_record(
    record_dict: dict,
    key: bytes,
    output_path: _Optional[str] = None,
) -> SealedBundle:
    """
    Seal a seedbank record dict (from generate_json_report() or record_provenance()).

    Serialises to canonical JSON, encrypts, optionally writes to output_path.
    Returns the SealedBundle.
    """
    import json as _json

    payload = _json.dumps(record_dict, separators=(",", ":"), sort_keys=True).encode("utf-8")
    record_id = record_dict.get("record_id") or record_dict.get("id", "")

    bundle = seal(
        payload,
        key,
        content_type="record",
        metadata={
            "record_id": record_id,
            "filename": record_dict.get("filename", ""),
        },
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(bundle.to_json())
        print(f"[SEAL] record {record_id[:8]}… → {output_path}")

    return bundle


def seal_provenance_bundle(
    record_id: str,
    key: bytes,
    output_path: _Optional[str] = None,
) -> SealedBundle:
    """
    Build and seal the provenance bundle for a seedbank record.

    This is the primary IP protection artefact: a time-stamped, content-hashed,
    encrypted archive of the record's full provenance — reading history, CDC events,
    integrity check results, and provenance_hash.

    Committing the .sealed output file to a public Git repo creates a
    timestamped witness: this provenance existed in this exact form at this date.
    If attribution is ever challenged, unsealing the bundle proves it.

    output_path: defaults to seedbank/records/{record_id}_provenance.sealed
    """
    from seedbank.provenance import record_provenance
    import seedbank.index as _idx_seal

    prov = record_provenance(record_id)

    if output_path is None:
        records_dir = _idx_seal.RECORDS_DIR
        output_path = _os.path.join(records_dir, f"{record_id}_provenance.sealed")

    bundle = seal_record(prov, key, output_path=output_path)
    print(f"[SEAL] Provenance bundle for {record_id[:8]}… → {output_path}")
    return bundle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_datetime_utc() -> str:
    """Return current UTC time as ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# KindPath Analyser — Schema Versions

This document is the authoritative record of each schema version for the
seedbank index and record format. Every field addition, type change, or
structural revision must be recorded here before merging.

The schema version is not a number carried in the index file — it is a
timestamp-anchored description of the state of the data model at a point
in time. Changes are additive wherever possible (new optional fields do
not break existing readers). Breaking changes are flagged explicitly.

---

## Schema 1.0 — Initial (2026-07 sprint)

**Released:** 2026 (retroactive documentation)

### Index format (`seedbank/index.json`)

```json
{
  "version": 1,
  "last_updated": "<ISO 8601>",
  "total": <int>,
  "records": [ <SeedbankRecord>, ... ]
}
```

### SeedbankRecord fields

| Field                    | Type          | Notes                                                         |
|--------------------------|---------------|---------------------------------------------------------------|
| `id`                     | string (UUID) | Record identifier                                             |
| `deposited_at`           | string        | ISO 8601 UTC timestamp                                        |
| `filename`               | string        | Original audio filename                                       |
| `context`                | string        | Depositor-provided description                                |
| `release_circumstances`  | string\|null  | How this was released                                         |
| `creator_statement`      | string\|null  | Direct creator voice                                          |
| `duration_seconds`       | float         |                                                               |
| `lsii_score`             | float         | 0–1                                                           |
| `lsii_flag_level`        | string        | none / low / moderate / high / extreme                        |
| `authentic_emission_score` | float       | 0–1                                                           |
| `manufacturing_score`    | float         | 0–1                                                           |
| `creative_residue`       | float         | 0–1                                                           |
| `era_fingerprint`        | string        | Top era match name                                            |
| `key_estimate`           | string        |                                                               |
| `tempo_bpm`              | float         |                                                               |
| `full_profile_path`      | string        | Relative path to full JSON in `records/`                      |
| `tags`                   | list[string]  |                                                               |
| `genre_estimate`         | string        |                                                               |
| `verified`               | bool          |                                                               |
| `verification_notes`     | string        |                                                               |
| `baseline_version`       | string        | SHA-based snapshot of tag constants at deposit time           |
| `reconstruction_protocol` | string       | Human-readable description of how `creative_residue` was derived |
| `reading_history`        | list[Reading] | Fork-and-retain history (see below)                           |

### Reading fields (inside `reading_history`)

| Field                    | Type          | Notes                                            |
|--------------------------|---------------|--------------------------------------------------|
| `creative_residue`       | float         |                                                  |
| `authentic_emission_score` | float       |                                                  |
| `manufacturing_score`    | float         |                                                  |
| `baseline_version`       | string        | The k-context this reading was computed in       |
| `reconstruction_protocol` | string       |                                                  |
| `computed_at`            | string        | ISO 8601 UTC                                     |
| `is_current`             | bool          | False for all readings except the latest         |
| `computation_source`     | string        | deposit / recompute / confluent                  |
| `residue_delta_from_prior` | float\|null | None for first reading                           |
| `parent_reading_ids`     | list\|null    | Timestamps of parent readings for confluent      |

---

## Schema 1.1 — Provenance addition (2026-03-13)

**Released:** 2026-03-13

### What changed

**New field in every Reading entry:**

| Field                  | Type   | Notes                                                                      |
|------------------------|--------|----------------------------------------------------------------------------|
| `reading_input_hash`   | string | SHA-256 hex of `{"authentic_emission_score", "baseline_version", "creative_residue", "manufacturing_score"}` (sorted keys). Proves the exact values present when this reading was computed. |

**Backward compatibility:** Pre-1.1 readings lack this field. They remain
valid — `verify_record_integrity()` reports the absence as an informational
note, not an error. All new deposits and recomputations (from 2026-03-13)
carry `reading_input_hash`.

**New file: `seedbank/events.jsonl`**

Append-only Change Data Capture log. One JSON line per mutation:
```json
{"timestamp": "...", "event_type": "deposit|fork|confluence|stale_marked|stale_cleared|verified", "record_id": "...", "data": {...}, "chain_hash": "..."}
```

The `chain_hash` field is SHA-256 of `prior_chain_hash + this_event_json`.
Tampering with any line breaks the chain from that point forward.
Verified by `seedbank.cdc.verify_chain()`.

**New modules:**

- `seedbank/cdc.py` — CDC event writer, reader, and chain verifier
- `seedbank/provenance.py` — Record provenance reconstruction and export

**No breaking changes.** Existing readers ignoring unknown fields
continue to work without modification.

---

## Schema 1.2 — Sealing (2026-03-13)

**Released:** 2026-03-13

### What changed

**New capabilities in `kindpress/transport.py`:**

- `SealedBundle` dataclass — AES-256-GCM encrypted, tamper-evident archive
- `seal(payload, key)` — seal arbitrary bytes
- `unseal(bundle, key)` — decrypt and verify
- `seal_file(filepath, key)` — seal an arbitrary file to `.sealed` JSON
- `unseal_file(sealed_path, key)` — reverse
- `seal_record(record_dict, key)` — seal a seedbank record
- `seal_provenance_bundle(record_id, key)` — build + seal provenance bundle
- `generate_key()` — 32-byte random AES key
- `derive_key(passphrase, salt)` — PBKDF2-HMAC-SHA256 key derivation (600k rounds)
- `derive_dual_key(k_baseline, delta)` — prototype dual-key derivation
  (k-key from shared constants + Δ-key from individual delta;
  intended for community data ownership hubs in pilots)

**New dependency:** `cryptography>=41.0.0` (optional — sealing functions
raise `ImportError` with install instructions if absent; all prior transport
functions work without it).

**No breaking changes.**

---

## Upgrade path for existing deployments

On first run after upgrading to Schema 1.1+:

1. **Backfill CDC events** for pre-CDC records:
   ```python
   from seedbank.provenance import backfill_cdc_for_existing_records
   backfill_cdc_for_existing_records()
   ```

2. **Verify corpus integrity**:
   ```python
   from seedbank.provenance import corpus_integrity_report
   print(corpus_integrity_report())
   ```

3. **Install sealing dependency** if you want IP protection bundles:
   ```bash
   pip install cryptography>=41.0.0
   ```

Pre-1.1 readings are not retroactively modified. The `reading_input_hash`
field will be absent on old readings — this is expected and noted by the
integrity checker.

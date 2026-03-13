# Error Correction Landscape — KindPath Analyser

**Scope:** How adjacent fields handle errors, drift, and versioned mutation in
persistent data stores. What can we adopt, adapt, or reject for the KindPath
analyser seedbank and KindPress encoding pipeline?

**Date:** 2025-07  
**Author:** KindPath Engineering

---

## The Problem We Are Solving

The analyser's seedbank is a living corpus. Constants (k) evolve. Feature
extractors improve. Fingerprint baselines shift as more records accumulate.
When any of these change, every existing record's Δ values potentially
become stale — they were valid under the old k, but may not reflect what
the new k would produce.

We need a model for:
1. **Detecting** when a record's stored values have drifted from what a
   current extraction would produce
2. **Preserving** the original reading (fork-and-retain, not overwrite)
3. **Communicating** schema evolution between systems (CLI, workspace UI, future APIs)
4. **Validating** proposed revisions before they're committed (HMoE gate)

The question is: which of these four concerns is our own invention, and which
has a well-solved prior art we should stand on?

---

## Candidate Approaches

### 1. Forward Error Correction (FEC) / Error-Correcting Codes (ECC)

**Origin:** Telecommunications, storage systems (RAID, QR codes, DVDs)

**What it does:** Encodes redundant information alongside data so that
errors introduced in transit or storage can be detected and corrected
without retransmission.

**Adoption signal:** Hamming codes, Reed-Solomon, turbo codes, LDPC.

**Relevance to us:** Our data doesn't transit a noisy channel — it's written
and read from a reliable local store (SQLite, JSON files). ECC's primary
use case (bit-flip correction in unreliable media) doesn't apply.

However, the *principle* of encoding enough redundancy to detect corruption
is relevant: if we store a hash of the feature vector alongside the reading,
we can detect whether the stored values were accidentally corrupted vs.
intentionally forked. `seedbank/fork_log.py` already does this partially
with `prior_reading_hash`.

**Recommendation: ADAPT**
- Keep the hash-based integrity check already in fork_log.py
- Don't adopt full ECC — unnecessary complexity for reliable local storage
- The hash check is the light-weight equivalent we need

---

### 2. Change Data Capture (CDC)

**Origin:** Database replication (Debezium, PostgreSQL WAL, MySQL binlog)

**What it does:** Captures every insert/update/delete at the database level
as a stream of change events, enabling downstream consumers to stay in sync
without full scans.

**Relevance to us:** The seedbank currently uses a JSON index + individual
record files. There is no event stream — consumers (workspace UI, kindai,
future APIs) scan the index to detect changes.

As the seedbank grows past a few hundred records, scan-based discovery
becomes slow. CDC-style event streaming would let consumers react to new
deposits or k-revisions without scanning.

**Near-term implementation path:**
- A lightweight `seedbank/events.jsonl` append-only event log
- Each deposit, fork, and k-revision appends a JSON line: `{event, record_id, timestamp, version}`
- Consumers tail this file rather than scanning index.json
- This is CDC's core idea without the database overhead

**Recommendation: ADOPT (lightweight version)**
- Add `seedbank/events.jsonl` append-only event log
- Events: `record_deposited`, `reading_forked`, `tag_revised`, `baseline_updated`
- Keep the existing index.json for fast lookups; events.jsonl for change tracking
- This enables the workspace UI to poll for updates efficiently

---

### 3. Event Sourcing

**Origin:** DDD / CQRS patterns (Greg Young, Axon Framework, EventStore)

**What it does:** Stores all state as an immutable sequence of events rather
than mutable records. Current state is derived by replaying the event log.
Enables full audit trail, time-travel, and arbitrary projections.

**Relevance to us:** This is conceptually very close to what fork-and-retain
already implements for individual records — the `reading_history` array IS
an event log for that record's interpretation.

Full event sourcing at the corpus level would mean: never store computed
Δ values directly, only store the raw audio features and k-version, and
derive creative_residue etc. by replaying k-versions against the raw features.

**The gap:** We do store computed values (creative_residue, lsii_score etc.)
directly. This is deliberate — recomputing from scratch on every read would
be expensive, and the raw feature extraction is not deterministic (depends on
librosa versions, floating-point quirks, etc.).

**What we should adopt:** The event sourcing *vocabulary* and *mindset*:
- "A reading is an event, not a state" — already in fork_log.py
- "Projections are derived, not stored" — the corpus stats in reason.py
  are already computed on-the-fly, not stored
- "Never delete, always append" — already the fork-and-retain principle

**What we should not adopt:** Full event sourcing with replay-to-derive.
The raw audio features are not perfectly reproducible across extractor
versions. Storing the derived values alongside the extractors version that
produced them is the correct tradeoff.

**Recommendation: ADAPT (conceptually aligned, don't fully implement)**
- We already have the key insight: reading_history as immutable append-only event log
- Add `feature_extractor_version` to every reading so we know which extractor produced it
- This is the minimum viable event sourcing: enough to detect drift without replay overhead

---

### 4. Schema Evolution (Avro, Protobuf, Thrift)

**Origin:** Big data / streaming (Apache Kafka + Schema Registry, Apache Avro)

**What it does:** Defines explicit schema versions for serialised data.
Readers can decode data written by older or newer writers using resolution rules
(default values, type coercions, field renaming maps).

**Relevance to us:** Our JSON records don't have a formal schema today. When
we add a new field (e.g. `hmoe_framing` in this sprint), existing records don't
have it. Readers need to handle missing fields gracefully.

**Near-term need:** The JSON report format IS our schema. We should treat it as
such:
- Add a `"version": "1.0"` field to all reports (already done in `generate_json_report()`)
- Maintain a `docs/SCHEMA_VERSIONS.md` that documents each version's field set
- When adding fields, increment the minor version and document defaults for
  readers that encounter v1.0 records when expecting v1.1

**Full Avro/Protobuf:** Overkill for a local-first JSON store. The complexity
cost (binary format, separate schema registry, generated code) is not justified
until we're streaming records between systems at scale.

**Recommendation: ADOPT (lightweight)**
- Formalise version increments in `docs/SCHEMA_VERSIONS.md`
- Add migration helpers in `seedbank/migrate.py` when fields change shape
- Reserve Avro/Protobuf for future distributed deployment

---

### 5. Semantic Drift Detection

**Origin:** NLP / embeddings (word2vec temporal analysis, Garg et al. 2018,
Hamilton et al. 2016 "Diachronic Word Embeddings")

**What it does:** Tracks how the meaning of a symbol (word, concept, embedding)
changes over time by measuring the angular distance between its representation
at different timepoints.

**Relevance to us:** A tag's semantic meaning can drift even when its definition
string hasn't changed. If the corpus of records tagged with `"high_lsii"` shifts
over time (because the feature extractor improves, or because new records expand
the distribution), the tag's effective meaning drifts even though its name is fixed.

This is exactly what `kindpress/reason.py`'s DeltaDistribution analysis detects:
systematic bias in the Δ distribution for a tag cluster IS semantic drift at the
tag level.

The HMoE gate in `tags_registry.py` (revise_tag with validate_first=True) directly
implements semantic drift detection before committing a revision. The vocabulary is
different, but the mechanism is the same: "before you update this symbol's definition,
check whether the records it currently applies to are consistently moving together —
if they are, the symbol's current meaning may be misaligned with the data."

**Recommendation: ALREADY IMPLEMENTED**
- `kindpress/reason.py` is our semantic drift detector
- `kindpress/validate.py`'s 6-stage protocol is our schema evolution gating
- The connection to academic literature is worth documenting for future contributors
- Add a note to `docs/KINDPRESS_SPEC.md` referencing Hamilton et al. for context

---

### 6. Merkle Trees / Content-Addressed Storage (IPFS, Git)

**Origin:** Git, IPFS, blockchain

**What it does:** Addresses data by the hash of its content rather than
its location. Every version of a record has a unique hash that is derived
from its content. The history of changes is a Merkle tree — each node
includes the hash of its parent(s), making the entire history tamper-evident
and uniquely addressable.

**Relevance to us:** The `prior_reading_hash` in `fork_log.py` is a manual
implementation of the first layer of this: each reading records the hash of
the reading it forked from. This creates a hash chain (a degenerate Merkle
tree — one parent per node).

A full Merkle tree would also hash the tag version, the feature extractor
version, and the baseline version into the reading address. This would make
it possible to detect whether any of the inputs to a reading have changed
without running the full recompute.

**Near-term implementation path:**
- Add a `reading_input_hash` to each reading that is the hash of:
  `(audio_filepath, feature_extractor_version, tag_version, baseline_version)`
- If this hash changes → stale reading, candidate for recompute
- This is cheaper than actually recomputing, and avoids unnecessary forks

**Recommendation: ADOPT (one field)**
- Add `reading_input_hash` to the reading dict in `seedbank/recompute.py`
- Use `hashlib.sha256` over a canonical JSON string of the four inputs
- Stale detection: if any input changes, the hash will mismatch → flag for recompute
- Full Merkle trees: future work, only needed at multi-system scale

---

### 7. Provenance / Lineage Systems (W3C PROV, Apache Atlas)

**Origin:** Scientific data management, data governance

**What it does:** Records the full lineage of a data artefact: who created it,
from what inputs, using which processes, at what time. Enables reproducibility
and impact analysis ("if I change this input, which outputs are affected?").

**Relevance to us:** The seedbank already captures partial provenance:
- `deposited_at`, `context`, `release_circumstances` in each record
- `prior_reading_id` and `prior_reading_hash` in each fork
- `tag_version` and `baseline_version` embedded in readings

What we're missing: **impact analysis**. If tag `"high_lsii"` is revised,
which records are affected? We have `flag_stale_records()` in `tags_registry.py`,
but it doesn't quantify the downstream impact on k-calibration.

**Near-term implementation:**
- `seedbank/provenance.py`: `impact_analysis(tag_name) → dict` — how many records
  are tagged, what is their current stale status, what would recompute cost
- `corpus_residue_drift()` in recompute.py already provides the output-side view;
  provenance.py would provide the input-side view

**Recommendation: ADOPT (one new module)**
- Create `seedbank/provenance.py` with `impact_analysis(tag_name)` in a future sprint
- Sufficient for now: document the partial provenance fields already present
- Full W3C PROV model: overkill for current scale

---

## Priority Adoption Map

| Approach                    | Recommendation   | Effort | Sprint |
|-----------------------------|------------------|--------|--------|
| ECC / hash integrity        | ADAPT (already partial) | — | done (fork_log.py) |
| CDC event log               | **ADOPT**        | Low    | next sprint |
| Event sourcing vocabulary   | ADAPT (already aligned)  | — | done |
| Schema versioning           | **ADOPT**        | Low    | next sprint |
| Semantic drift detection    | ALREADY IMPLEMENTED | — | done (reason.py) |
| Merkle reading_input_hash   | **ADOPT**        | Very low | this sprint or next |
| Provenance impact_analysis  | ADOPT (deferred) | Medium | future sprint |
| Full Avro/Protobuf          | REJECT (for now) | High   | revisit at scale |
| Full Merkle trees           | REJECT (for now) | High   | revisit at scale |
| Full event sourcing replay  | REJECT           | Very high | never (extraction not reproducible) |

---

## Immediate Actionable Items

These three items can be done with minimal disruption to the existing codebase:

### A. reading_input_hash (this sprint)
Add to `seedbank/recompute.py`:

```python
import hashlib, json

def _reading_input_hash(filepath: str, extractor_version: str,
                         tag_version: int, baseline_version: str) -> str:
    """Stable hash of the four inputs that determine a reading's value."""
    canonical = json.dumps({
        "filepath": filepath,
        "extractor_version": extractor_version,
        "tag_version": tag_version,
        "baseline_version": baseline_version,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

### B. seedbank/events.jsonl (next sprint)
Append-only event log. One JSON line per event:

```python
{"event": "record_deposited", "record_id": "...", "tag": "...", "ts": "2025-07-01T..."}
{"event": "reading_forked", "record_id": "...", "from_tag_v": 2, "to_tag_v": 3, "ts": "..."}
{"event": "tag_revised", "tag_name": "...", "from_v": 2, "to_v": 3, "ts": "..."}
```

### C. docs/SCHEMA_VERSIONS.md (next sprint)
Document the JSON report schema version history. Start at v1.0 (current),
v1.1 when `hmoe_framing` is added (this sprint).

---

## References

- Hamilton et al. (2016). Diachronic Word Embeddings Reveal Statistical Laws of
  Semantic Change. https://arxiv.org/abs/1605.09096
- Greg Young (2010). CQRS Documents. https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
- W3C PROV-DM (2013). https://www.w3.org/TR/prov-dm/
- Kleppmann, M. (2017). Designing Data-Intensive Applications. O'Reilly. Chapter 11 (Stream Processing) and Chapter 10 (Batch Processing) are the most relevant.
- Merkle, R. (1979). A Certified Digital Signature. Crypto '89.

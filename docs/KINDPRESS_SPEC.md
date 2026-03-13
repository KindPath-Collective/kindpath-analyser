# KindPress — Module Specification

*Version 1.0 — March 2026*

---

## Purpose

KindPress is the compression and reasoning layer of the KindPath Analyser seedbank.

It solves a fundamental problem in any growing archive of fingerprinted experience:
**constants change**. As the seedbank accumulates records and refines its baselines,
the shared context under which old records were encoded becomes outdated. Without
a means of compressing, reasoning about, and validating those changes, the archive
either silently drifts into inaccuracy or grows indefinitely under the weight of
full re-analysis.

KindPress enables the archive to remain **calibrated** without requiring full
audio re-analysis every time a tag definition or era baseline is revised.

---

## Non-Goals

KindPress does **not**:
- Replace the full audio analysis pipeline. It operates on already-extracted features.
- Produce lossy approximations intended for playback or reconstruction of audio.
- Perform machine learning or neural embedding. It is an information-theoretic tool.
- Store raw audio. It operates exclusively on the numerical fingerprint vectors
  produced by the core analysis engine.

---

## Architecture

KindPress has three distinct subsystems, each with a clear boundary:

```
kindpress/
├── press.py       — Encoding + decoding. The compression layer.
├── reason.py      — HMoE calculation. The calibration measurement layer.
└── validate.py    — Tag revision validation protocol. The governance gate.
```

These subsystems share no shared mutable state. They communicate via data structures
and the seedbank index. Each can be tested and used independently.

---

## Subsystem 1: press.py — The Compression Layer

### Role

Encodes a seedbank record as a `KindPressPacket`: a compact representation of its
creative fingerprint in terms of the constants (k) and the individual delta (Δ).

This implements the core claim of [CONSCIOUS_CONTEXTUALISATION.md]:

```
Full picture = Constants(k) + Fingerprint(Δ)
```

### Public API

```python
from kindpress.press import encode, decode, verify_integrity, k_alignment_check

# Encode a seedbank record
packet: KindPressPacket = encode(record_dict, baseline_dict)

# Decode a packet back to a reconstructed record dict
record: dict = decode(packet, baseline_dict)

# Verify that a packet decodes faithfully to the source record
ok: bool = verify_integrity(packet, original_record_dict, baseline_dict)

# Check how well-aligned a record's constants are with the current baseline
score: float = k_alignment_check(record_dict, baseline_dict)

# Compression statistics
ratio: float = compression_ratio(packet, original_record_dict)
delta_bytes: int = delta_size_bytes(packet)
```

### KindPressPacket structure

```python
@dataclass
class KindPressPacket:
    record_id: str          # Source record UUID
    baseline_version: str   # Which constants this delta is relative to
    encoded_at: str         # ISO 8601 timestamp

    # The delta vector — what is irreducibly individual about this record
    delta: dict             # feature_name → residual value after k subtraction

    # Compression metadata
    original_feature_count: int
    encoded_feature_count: int    # Features where delta is non-trivial
    compression_ratio: float
```

### Design constraints

- Encoding is **lossless** within floating-point precision. `verify_integrity` must pass
  for all records where `k_alignment_check` returns > 0.5.
- Encoding is **deterministic**. The same record + same baseline always produces the
  same packet (modulo `encoded_at`).
- No allocation beyond standard Python dataclasses. No network calls.

---

## Subsystem 2: reason.py — The Calibration Measurement Layer

### Role

Measures the information density of the current constants bank (k) against the
corpus of existing records. The metric is **HMoE**: Heterogeneous Multiplicity of
Evidence.

```
HMoE = var(creative_residue) × mean(effective_n)
```

Where:
- `var(creative_residue)`: variance of individual fingerprints — how discriminating
  the current k is. If k is too broad, everything clusters near zero. If k is
  too narrow, everything is flagged as divergent. Healthy k produces a spread.
- `mean(effective_n)`: mean effective sample size across records, accounting for
  the number of independent readings (k-universes) each record has accumulated.
  A record with multiple readings under different k-versions has higher effective_n
  than a record seen only once.

High HMoE = the constants bank is well-calibrated: it produces discriminating
fingerprints, and the confidence in those fingerprints is supported by independent
readings. This is the target state.

### Public API

```python
from kindpress.reason import (
    hmoe_of_corpus,
    k_calibration_score,
    analyse_delta_distribution,
    survey_all_tags,
    DeltaDistribution,
)

# HMoE for a specific set of record IDs
hmoe: float = hmoe_of_corpus(record_ids)

# Overall calibration score (normalised HMoE) for a tag
score: float = k_calibration_score(tag_name)

# Full distribution analysis for a tag's delta values
dist: DeltaDistribution = analyse_delta_distribution(tag_name)

# Survey all tags and return ranked calibration scores
survey: list[dict] = survey_all_tags()
```

### DeltaDistribution

```python
@dataclass
class DeltaDistribution:
    tag_name: str
    record_count: int
    mean_residue: float
    variance: float
    std_dev: float
    min_residue: float
    max_residue: float
    mean_effective_n: float
    hmoe: float
    calibration_tier: str   # "well_calibrated" | "under_discriminating" | "over_discriminating"
    calibration_notes: list[str]
```

### Calibration thresholds

| HMoE | Interpretation |
|------|---------------|
| < 0.01 | Under-discriminating: k is too coarse, or corpus is too small |
| 0.01–0.05 | Developing: reasonable spread, low confluence |
| 0.05–0.20 | Well-calibrated: good discriminating power, growing effective_n |
| 0.20–0.50 | Highly calibrated: strong spread, significant multi-reading history |
| > 0.50 | High-information: may indicate k instability or genuine diversity |

---

## Subsystem 3: validate.py — The Governance Gate

### Role

Before any tag definition change propagates to the seedbank, it must pass a
staged validation protocol. This protocol ensures that the proposed revision
**improves** the information density of the constants bank — it does not simply
shift what gets labelled, hidden behind the authority of a version increment.

This is the mandatory gate. No `revise_tag()` call should proceed without one.

### The six-stage protocol

```
STAGE 1 — Stratified sample selection
    Draw records bearing the tag, spread across the full creative_residue
    distribution (quartile-stratified, not mean-biased).

STAGE 2 — Compressed HMoE baseline
    Encode the sample as KindPress packets.
    Compute HMoE = var(creative_residue) × mean(effective_n).
    This is the information density under the CURRENT k.

STAGE 3 — Simulated k revision
    Apply the proposed description/scope change as a temporary k shift.
    Identify which records would be flagged stale.
    Estimate their expected residue shift from reading history.

STAGE 4 — Projected HMoE under proposed k
    Compute HMoE using estimated post-revision residues.
    projected HMoE > current HMoE → revision increases information density.
    projected HMoE ≤ current HMoE → revision reduces or flattens the corpus.

STAGE 5 — Evolutionary optimum search
    Repeat stages 2–4 across increasing chunk sizes until HMoE is maximised.
    A revision that only improves small samples (but degrades at larger ones)
    is likely overfitting to the stratified sample.
    Valid revisions must improve HMoE at the evolutionary optimum, not just
    at minimal chunk sizes.

STAGE 6 — DB-wide implication report
    At the optimum chunk size, scale the analysis to all records bearing the tag.
    Report: stale record count, expected drift magnitude, HMoE trajectory.
```

### Recommendations

| Recommendation | Meaning | Action |
|---|---|---|
| APPROVE | Revision improves HMoE at evolutionary optimum | Proceed with `revise_tag()` |
| CAUTION | Sound at small scale, degrades at optimum | Proceed with monitoring; review after next 50 deposits |
| REJECT | Revision reduces HMoE | Rework the proposed description; re-run before committing |

### Public API

```python
from kindpress.validate import validate_tag_revision, print_report, TagRevisionReport

report: TagRevisionReport = validate_tag_revision(
    tag_name="high_lsii",
    proposed_description="...",
    proposed_scope="...",
    chunk_sizes=[10, 25, 50, 100, 200],
    max_sample=200,
    seed=42,
)

print_report(report)

# Pass report to revise_tag() to avoid re-running the protocol:
from seedbank.tags_registry import revise_tag
revise_tag(
    name="high_lsii",
    description=report.proposed_description,
    scope=report.proposed_scope,
    revision_reason=report.revision_rationale,
    _validation_report=report,  # pre-computed — skips re-run
)
```

### TagRevisionReport fields

```python
@dataclass
class TagRevisionReport:
    tag_name: str
    tag_current_version: int
    proposed_description: str
    proposed_scope: str
    validated_at: str

    recommendation: str     # APPROVE | CAUTION | REJECT
    is_sound: bool

    evolutionary_optimum_chunk: int
    hmoe_probes: list[HMoEProbe]

    db_wide_stale_count: int
    db_wide_expected_drift: float
    hmoe_at_optimum_current: float
    hmoe_at_optimum_projected: float

    calibration_notes: list[str]
    implication_summary: str
    revision_rationale: str   # Suggested text for revise_tag(revision_reason=...)
```

---

## Integration with tags_registry.py

The governance gate is enforced at the `revise_tag()` level.

```python
# Standard path — validation runs automatically
revise_tag(
    name="high_lsii",
    description="...",
    scope="...",
    revision_reason="Narrowed from generic LSII > 0.4 to strong inversion ≥ 0.6",
    validate_first=True,      # default — cannot be bypassed silently
)

# Pre-validated path — pass your report to avoid re-running
revise_tag(..., _validation_report=report)

# Exploration path — propose without committing
from seedbank.tags_registry import propose_tag_revision
report = propose_tag_revision("high_lsii", description="...", scope="...")
print(report.recommendation, report.revision_rationale)
```

---

## Transport Profile

See `kindpress/transport.py` for a secondary encoding profile optimised for
network payloads rather than reconstruction fidelity. The transport profile:
- Drops features whose delta is below a configurable noise floor
- Uses integer quantisation for delta values
- Produces JSON-serialisable payloads suitable for API transfer

---

## HMoE Mapping to Analyser Metrics

The following table shows how core analyser outputs map onto HMoE constructs.
This mapping is exposed in analysis reports to help users understand what the
numbers mean in terms of information density.

| Analyser metric | HMoE role | Interpretation |
|---|---|---|
| `creative_residue` | Primary residue signal (m) | What remains after k subtracted — individual deviation |
| `LSII` | Late-signal divergence read | High LSII = Q4 is a different k-universe than Q1-Q3 |
| `groove_deviation_ms` | Temporal residue | Human timing variation after genre quantisation subtracted |
| `authentic_emission_score` | Composite authentic signal | Proportion of fingerprint not explained by genre/era constants |
| `manufacturing_score` | Inverse residue | How much of the fingerprint *is* the genre/era constant |
| `effective_n` (seedbank) | Confidence multiplier | How many independent k-universe readings support this record |

A record with high `creative_residue`, high `LSII`, and high `authentic_emission_score`
represents high information density in HMoE terms — its fingerprint is discriminating,
and the constants bank has not yet over-indexed it into a known category.

---

## Glossary

| Term | Definition |
|---|---|
| **k (constants)** | The shared environmental substrate — era, genre, production context |
| **Δ (delta)** | Individual fingerprint after k is subtracted |
| **creative_residue** | Normalised Δ for a given record |
| **HMoE** | var(creative_residue) × mean(effective_n) — information density |
| **effective_n** | Effective sample size for a record, accounting for k-universe count |
| **k-universe** | The space of valid interpretations under one version of k |
| **lineage fork** | When k is revised, a record produces a new reading — a new branch of its interpretational history |
| **confluence** | The synthesis of readings from distinct k-universes into a new composite reading |
| **evolutionary optimum** | The chunk size at which HMoE is maximised; the sweet spot between underfitting and overfitting to sample size |
| **baseline_version** | The tag + version combination under which a record was encoded |

---

*KindPress does not mystify technique. The knowledge belongs to everyone.*

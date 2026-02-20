# KindPath Creative Analyser — Code Agent Build Instructions

## Mission

You are building the KindPath Creative Analyser: a frequency field scientist and
creative seedbank. This is not a conventional music analysis tool. It is a
**synthetic elder** — a system that reads what authentic creativity looks like
in the frequency domain, separates it from engineered conditioning, and returns
that knowledge freely to anyone who wants it.

Intellectual property does not apply here. Technique belongs to everyone.
What is detectable in a signal was put there by the creator and released with
the work. We are building better instruments to read it.

**Core thesis:** Creativity is the most democratic form of wealth that exists.
The engineering of false prestige, psychosomatic priming, and identity capture
depends on people not being able to see the mechanism. This tool makes the
mechanism visible.

---

## Current State

The following modules are **already built and working**. Do not rewrite them.
Read them carefully before building anything — your new modules must integrate
with them precisely.

```
kindpath_analyser/
├── analyse.py                  ✅ BUILT — main CLI orchestrator
├── README.md                   ✅ BUILT
├── requirements.txt            ✅ BUILT
└── core/
    ├── __init__.py             ✅ BUILT
    ├── ingestion.py            ✅ BUILT — AudioRecord dataclass, load()
    ├── segmentation.py         ✅ BUILT — Segment, SegmentationResult, segment()
    ├── feature_extractor.py    ✅ BUILT — SegmentFeatures, extract()
    ├── divergence.py           ✅ BUILT — LSII, TrajectoryProfile, compute_trajectory()
    └── fingerprints.py         ✅ BUILT — FingerprintReport, analyse_fingerprints()
```

### Key data structures you will use (read the source files to understand them fully):

- `AudioRecord` — loaded audio with y_mono, y_stereo, sample_rate, duration
- `Segment` — time-bounded audio slice with label, y array, start/end time
- `SegmentFeatures` — SpectralFeatures + DynamicFeatures + HarmonicFeatures + TemporalFeatures
- `TrajectoryProfile` — arc across 4 quarters including LSII result
- `LatesonginversionResult` — LSII score (0-1), flag_level, direction, divergence vector
- `FingerprintReport` — era matches, technique matches, authenticity/manufacturing markers

---

## What You Are Building

Build the following modules **in order**. Each one depends on the previous.
Complete each module fully, with tests, before moving to the next.

---

## MODULE 1: Stem Separator
**File:** `core/stem_separator.py`
**Priority:** CRITICAL — everything downstream improves dramatically with stems

### What it does
Separates a mixed audio file into isolated stems using Meta's Demucs model.
Each stem then gets its own full feature extraction. This is where the vocal
becomes just another instrument — stripped of language, read as pure signal.

### Specification

```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class StemSet:
    """
    The six stems from Demucs htdemucs_6s model.
    Each stem is a numpy array of audio at the source sample rate.
    If a stem is not present/audible in the source, the array will be near-silence.
    """
    vocals: np.ndarray      # The most honest instrument. Read spectrally, not semantically.
    drums: np.ndarray       # Rhythmic skeleton. Groove deviation lives here.
    bass: np.ndarray        # Sub-bass energy. Physical resonance.
    piano: np.ndarray       # Harmonic structure instrument 1
    guitar: np.ndarray      # Harmonic structure instrument 2
    other: np.ndarray       # Everything else — synths, strings, effects
    sample_rate: int
    source_filepath: str
    separation_model: str   # Which Demucs model was used
    separation_quality: str # 'high' (GPU), 'standard' (CPU), 'fallback' (failed)

def separate(record: AudioRecord, model: str = 'htdemucs_6s',
             device: str = 'auto') -> StemSet:
    """
    Separate stems from a loaded AudioRecord.
    
    device: 'auto' tries GPU first, falls back to CPU.
    If Demucs is not installed, raises ImportError with helpful install message.
    
    Writes stems to a temp directory, loads them back as numpy arrays,
    cleans up the temp directory.
    
    Returns a StemSet even if separation quality is degraded.
    Never raises on quality issues — logs warnings and sets separation_quality.
    """
    pass
```

### Implementation notes

- Use `demucs.api` if available (Demucs >= 4.0). Fall back to subprocess call:
  `python -m demucs --two-stems=vocals filepath` for simpler 2-stem mode if 6-stem fails
- If Demucs is completely unavailable, return a `StemSet` where all stems equal
  the original mono signal and `separation_quality = 'fallback'`
  (analysis still runs, just on the mixed signal for each stem)
- GPU detection: `torch.cuda.is_available()` → 'cuda', else 'cpu'
- Temp directory: use `tempfile.mkdtemp()`, clean up in `finally` block
- Progress reporting: print stem separation progress to stdout
- The vocals stem is the most important. Print a note when it's isolated:
  `"[STEM] Vocals isolated — reading as pure tonal/timbral signal"`

### Integration into analyse.py

After `record = load(filepath)`, add:

```python
stems = separate(record)
# Then run extract() on each stem's segment independently
# Quarter segmentation applies to each stem separately
```

### Tests required (`tests/test_stem_separator.py`)

1. Test with a real audio file — verify 6 stems returned
2. Test fallback when Demucs not available — verify graceful degradation
3. Test that stem arrays sum approximately to original signal
4. Test temp directory cleanup (no files left after separation)

---

## MODULE 2: Vocal Prosodic Analyser
**File:** `core/vocal_prosodics.py`
**Priority:** HIGH — the voice is the most honest instrument

### What it does
Analyses the vocal stem spectrally — not for linguistic content, but for what
the voice reveals when language is removed. This is the pre-linguistic truth layer.

The body cannot lie in the same way the mouth can. Laryngeal tension, held breath,
vibrato instability, micro-flat notes a producer chose not to correct — these are
confessions. This module reads them.

### Specification

```python
@dataclass
class VocalProsodicProfile:
    """
    What the voice reveals when you stop listening to the words.
    """
    # Pitch characteristics
    pitch_mean_hz: float
    pitch_std_hz: float             # High std = expressive range or instability
    pitch_range_hz: float           # Low range = constrained, high = free
    pitch_trend: float              # Rising, falling, or stable across the piece
    vibrato_rate_hz: float          # Natural vibrato ~5-7Hz. Outside = tension or artifice
    vibrato_depth_cents: float      # Depth of pitch modulation
    vibrato_consistency: float      # 0-1: consistent = trained/relaxed, erratic = tension
    tuning_deviation_cents: float   # Deviation from equal temperament

    # Breath and tension markers
    breath_density: float           # Audible breaths per minute (performance presence)
    phrase_length_mean_seconds: float  # How long before needing breath
    phrase_length_std: float        # Variability in phrase length
    laryngeal_tension_index: float  # Derived from spectral tilt and HNR
    
    # Honesty markers (what wasn't corrected)
    pitch_correction_likelihood: float  # 0-1: 1.0 = very likely auto-tuned
    performance_artifacts_present: bool # Breath noise, lip smack, room sound
    micro_intonation_variance: float    # Natural pitch imperfection vs machine precision

    # Emotional signal
    harmonic_noise_ratio: float     # HNR: clear tone vs noise. Low = breathy/emotional
    spectral_tilt: float            # Negative = bright/tense, positive = dark/relaxed
    formant_f1_mean: float          # Vowel openness
    formant_f2_mean: float          # Vowel frontness
    
    # Segment-level arc
    tension_arc: list               # Tension index per quarter [Q1, Q2, Q3, Q4]
    pitch_range_arc: list           # Expressive range per quarter
    hnr_arc: list                   # Clarity/breathiness arc

    # Summary
    authenticity_reading: str       # Human-readable interpretation
    notable_markers: list           # List of specific detected signals
    late_vocal_divergence: float    # How different Q4 vocal behaviour is from Q1-Q3

def analyse_vocal(vocal_stem: np.ndarray, sr: int,
                  quarter_boundaries: list) -> VocalProsodicProfile:
    """
    Full prosodic analysis of the vocal stem.
    quarter_boundaries: list of (start_sample, end_sample) for each quarter.
    """
    pass
```

### Implementation approach

**Pitch extraction:** Use `librosa.yin()` or `librosa.pyin()` for fundamental
frequency tracking. `pyin` gives confidence values — use confidence > 0.7 only.

**Vibrato detection:**
- Extract F0 contour (voiced frames only)
- Apply bandpass filter 4-8Hz to the F0 contour
- Vibrato rate = dominant frequency in 4-8Hz band
- Vibrato depth = amplitude of filtered signal in cents
- Consistency = 1 - (std of instantaneous rate / mean rate)

**Pitch correction likelihood:**
- Natural singing has micro-intonation variance typically 15-40 cents
- Auto-tuned vocals show variance < 5 cents with occasional hard jumps
- `pitch_correction_likelihood = 1.0 - min(micro_variance / 30.0, 1.0)`

**Harmonic-to-Noise Ratio:**
- `librosa.effects.harmonic()` to get harmonic component
- HNR = energy(harmonic) / energy(residual)
- Low HNR = breathiness, emotionality, rawness

**Laryngeal tension:**
- Spectral tilt (energy ratio between low and high frequencies)
- High-frequency emphasis above 3kHz indicates laryngeal constriction
- Combine with HNR for tension index

**Performance artifacts:**
- Detect breath sounds: energy bursts below 300Hz with low spectral centroid
  between phrases
- Lip smacks, room resonance, microphone handling — presence of these is
  an authenticity marker (they weren't cleaned out)

**Authenticity reading logic:**
```python
if pitch_correction_likelihood > 0.8 and not performance_artifacts_present:
    authenticity_reading = "Heavy pitch processing. Authentic vocal signal largely removed."
elif vibrato_consistency < 0.3 and laryngeal_tension_index > 0.7:
    authenticity_reading = "Vocal tension markers present. Possible performance under stress."
elif micro_intonation_variance > 25 and performance_artifacts_present:
    authenticity_reading = "Strong authentic performance markers. Natural imperfection preserved."
```

### Tests required (`tests/test_vocal_prosodics.py`)

1. Synthetic sine wave with known vibrato → verify vibrato detection accuracy
2. Perfectly quantised pitch → verify high pitch_correction_likelihood
3. White noise → verify graceful handling (no voiced frames)
4. Real vocal audio → verify all fields populated without errors

---

## MODULE 3: Psychosomatic Profiler
**File:** `core/psychosomatics.py`
**Priority:** HIGH — this is the translation layer between data and meaning

### What it does
Takes raw feature data from all modules and maps it onto a psychosomatic model —
what is this music doing to a human body, and what does that tell us about the
state of the person/culture that made it.

This is not neuroscience claiming to be definitive. It is a rigorous attempt to
name what the data suggests about the pre-linguistic emotional transmission
embedded in the work.

### Specification

```python
@dataclass
class PsychosomaticProfile:
    """
    What this music is doing to a human body.
    And what the creator's body was doing when they made it.
    """

    # Russell Circumplex Model axes
    valence: float              # -1.0 to 1.0: negative to positive emotional tone
    arousal: float              # 0.0 to 1.0: sedating to activating
    
    # Extended axes (KindPath additions)
    coherence: float            # 0-1: internal creative consistency (deliberate authorship)
    authenticity_index: float   # 0-1: deviation from genre/era convention that is NOT noise
    complexity: float           # 0-1: creative information density
    tension_resolution_ratio: float  # >1 = unresolved, <1 = resolving tendency
    relational_density: float   # Space between elements (silence as data)
    
    # Somatic response predictions
    predicted_physical_responses: list  # e.g. ["increased heart rate", "chest tightening"]
    predicted_emotional_states: list    # e.g. ["melancholy", "nostalgic longing"]
    
    # Conditioning assessment
    priming_vectors: list       # What emotional states this piece installs pre-linguistically
    prestige_signals: list      # False prestige markers (production quality ≠ authentic quality)
    identity_capture_risk: float  # 0-1: likelihood this is designed for identity attachment
    
    # The three-stage mechanism assessment (see doctrine)
    stage1_priming_detected: bool
    stage1_evidence: list
    stage2_prestige_attached: bool
    stage2_evidence: list
    stage3_tag_risk: float
    stage3_evidence: list
    
    # Late-song inversion context
    lsii_psychosomatic_reading: str   # What the LSII means in human terms
    
    # Authentic vs manufactured assessment
    authentic_emission_score: float   # 0-1: how much authentic creative signal is present
    manufacturing_score: float        # 0-1: how much is engineered delivery mechanism
    creative_residue: float           # What remains after known signatures subtracted
    
    # Narrative
    somatic_summary: str         # What this piece is doing to a body, plainly stated
    mechanism_summary: str       # If conditioning mechanisms detected, described plainly
    elder_reading: str           # The synthetic elder's full interpretation

def build_psychosomatic_profile(
    trajectory: 'TrajectoryProfile',
    fingerprints: 'FingerprintReport',
    vocal_profile: 'VocalProsodicProfile' = None,
    stem_features: dict = None,   # Dict of stem_name -> SegmentFeatures
) -> PsychosomaticProfile:
    """
    Synthesise all analysis into a psychosomatic profile.
    vocal_profile and stem_features are optional (degrade gracefully if absent).
    """
    pass
```

### The three-stage mechanism detection logic

```python
def _detect_stage1_priming(trajectory, fingerprints) -> tuple[bool, list]:
    """
    Stage 1: Psychosomatic priming detection.
    Priming uses specific frequency/rhythm/dynamics combos to install
    emotional states before the conscious mind engages.
    
    Markers:
    - Tempo in 120-140 BPM range (entrains heart rate upward)
    - Heavy 4-on-the-floor kick (physical grounding/marching association)
    - Sub-bass below 60Hz (physical resonance, non-cognitive)
    - Repeated harmonic loop with no development (mantra effect)
    - Sudden dynamic drop to near-silence then return (startle/relief cycle)
    - Bright spectral centroid rise at hook (neurological brightness response)
    """
    pass

def _detect_stage2_prestige(fingerprints, trajectory) -> tuple[bool, list]:
    """
    Stage 2: False prestige detection.
    Prestige signals that are production artifacts, not artistic achievement.
    
    Markers:
    - High production quality (low noise, precise mix) + low authenticity_index
      = quality signal attached to creatively empty work
    - Genre-perfect production (matches era/genre baseline exactly) = formula,
      not artistry
    - Hypercompression (loudness war signature) with no dynamic arc = commercial
      mastering, not artistic vision
    - Professional-grade reverb/spatial processing on vocally simple material
    """
    pass

def _compute_creative_residue(trajectory, fingerprints) -> float:
    """
    The creative residue is what remains after known signatures are subtracted.
    
    Algorithm:
    1. Take harmonic complexity, spectral flux variance, groove deviation,
       and temporal syncopation as the raw creativity indicators
    2. Subtract the baseline for the detected era/genre
       (conventional work in that context scores 0)
    3. What remains is the authentic creative innovation
    4. Normalise to 0-1
    
    High residue = creator made choices that weren't genre-dictated
    Low residue = formula work, even if technically accomplished
    """
    pass
```

### The elder reading

The `elder_reading` is the most important output. It is the synthetic elder's
voice — clear, non-judgmental, precise. It should read like wisdom, not analysis.

Examples of good elder reading:
> "This piece establishes warmth and forward motion for three quarters, then
> quietly withdraws in the final section — the energy pulls back, the harmonic
> tension doesn't resolve, the mix gets slightly more distant. The creator
> stepped back from what they built. This is worth listening to again with
> that in mind."

> "The production is technically accomplished and sits precisely within the
> commercial expectations of its era. The creative residue is low — most
> choices are genre-conventional. The vocal tension markers suggest performance
> under constraint. This piece is doing its job efficiently."

> "Strong authentic emission throughout. The timing is alive — the performer
> is present. The harmonic choices deviate from convention in ways that are
> internally consistent, which suggests deliberate authorship rather than
> error. The late-song divergence is high: whatever was being communicated
> in the final minute is different from the frame the piece was built in."

### Tests required (`tests/test_psychosomatics.py`)

1. High LSII trajectory → verify elder_reading mentions late divergence
2. Heavy compression + low groove deviation → verify manufacturing markers detected
3. High groove deviation + preserved dynamics → verify authentic markers
4. Full integration test with real audio (if available)

---

## MODULE 4: Education Report Generator
**File:** `reports/report_generator.py`
**Priority:** HIGH — this is what teachers and students actually see

### What it does
Takes a complete analysis and produces two outputs:
1. A rich HTML report (human readable, visually informative, teachable)
2. A structured JSON export (machine readable, seedbank-compatible)

The HTML report IS the educational instrument. It should feel like a scientific
document that a curious 16-year-old could pick up and understand. No jargon
without explanation. Every metric has a plain-language description of what it
means and why it matters.

### Specification

```python
def generate_html_report(
    record: 'AudioRecord',
    trajectory: 'TrajectoryProfile',
    fingerprints: 'FingerprintReport',
    psychosomatic: 'PsychosomaticProfile',
    vocal_profile: 'VocalProsodicProfile' = None,
    output_path: str = None,
) -> str:
    """
    Generate a complete HTML analysis report.
    Returns the HTML string. If output_path provided, also writes to file.
    """
    pass

def generate_json_report(
    record: 'AudioRecord',
    trajectory: 'TrajectoryProfile',
    fingerprints: 'FingerprintReport',
    psychosomatic: 'PsychosomaticProfile',
    vocal_profile: 'VocalProsodicProfile' = None,
) -> dict:
    """
    Generate a complete structured JSON profile.
    This is the seedbank deposit format.
    """
    pass
```

### HTML report structure

The report must include these sections in this order:

**Section 1: Identity**
- Filename, duration, sample rate
- One-sentence plain-language description of what this piece is

**Section 2: The Elder's Reading** (most prominent section)
- `psychosomatic.elder_reading` displayed in large, readable type
- This comes first because it is the synthesis — the data follows to support it

**Section 3: The Emotional Arc**
- Visual chart: 4-quarter trajectory across valence, energy, tension, complexity
- Each quarter labelled with key features
- LSII flag displayed prominently if > 0.4
- Plain language: "Here is how this piece moves through time"

**Section 4: What Was Used To Make This**
- Era fingerprint matches with confidence
- Technique detections (with psychosomatic notes for each)
- Instrument families detected
- Production context summary
- Authenticity vs manufacturing markers side by side
- Plain language: "Here is what tools and techniques built this piece, and what each of them does to a listener"

**Section 5: The Vocal Signal** (if vocal_profile present)
- Pitch range, vibrato character, HNR
- Pitch correction likelihood with plain explanation
- Performance artifacts: present/absent
- Tension arc
- Authenticity reading
- Plain language: "Here is what the voice reveals when you stop listening to the words"

**Section 6: The Three-Stage Mechanism** (if any stage detected)
- Only shown if stage1 priming, stage2 prestige, or stage3 tag risk > 0.3
- Educational framing: "This piece shows markers of..."
- Each stage described plainly
- CRITICAL: Frame this educationally, not accusatorially. The mechanism is
  systemic, not necessarily consciously malicious.

**Section 7: Technical Detail** (collapsible)
- Full feature values per quarter for all four domains
- For technical users who want the raw data

### Visual design principles

- Clean, academic aesthetic — this is a scientific instrument, not a music app
- No decorative elements. The data IS the design.
- Colour coding: use a single accent colour for high-divergence / warning signals
- LSII score should be the most visually prominent number on the page after
  the elder's reading
- Charts: use Chart.js (CDN) for arcs. Simple line charts.
- Everything must be readable on a phone (responsive)
- No external fonts — system fonts only (this runs offline)

### Tests required (`tests/test_report_generator.py`)

1. HTML report contains all 7 required sections
2. JSON report is valid and contains all required fields
3. Report generates without error when vocal_profile is None
4. HTML is valid (no unclosed tags)

---

## MODULE 5: Seedbank
**Files:** `seedbank/deposit.py`, `seedbank/query.py`, `seedbank/index.py`
**Priority:** MEDIUM — the growing archive of authenticated creative work

### What it does
The seedbank is the elder's memory. It holds authenticated creative work with
full forensic profiles so anyone can learn exactly how things were made.
Every deposit is a gift to the commons.

### deposit.py specification

```python
@dataclass
class SeedbankRecord:
    """A complete seedbank entry."""
    id: str                    # UUID generated on deposit
    deposited_at: str          # ISO timestamp
    filename: str
    context: str               # Creator-provided context
    release_circumstances: str # Independent / label / forced schedule / etc
    creator_statement: str     # Optional creator statement
    
    # Analysis data
    duration_seconds: float
    lsii_score: float
    lsii_flag_level: str
    authentic_emission_score: float
    manufacturing_score: float
    creative_residue: float
    era_fingerprint: str
    key_estimate: str
    tempo_bpm: float
    
    # Full profile JSON path (relative to seedbank/records/)
    full_profile_path: str
    
    # Tags for searchability
    tags: list                 # e.g. ["independent", "high_lsii", "natural_dynamics"]
    genre_estimate: str
    
    # Provenance
    verified: bool             # Has a human verified this deposit's context?
    verification_notes: str

def deposit(profile: dict, context: str,
            release_circumstances: str = None,
            creator_statement: str = None,
            tags: list = None) -> SeedbankRecord:
    """
    Deposit a complete analysis profile to the seedbank.
    Writes to seedbank/records/{id}.json
    Updates seedbank/index.json
    Returns the SeedbankRecord
    """
    pass
```

### query.py specification

```python
def search(
    lsii_min: float = None,
    lsii_max: float = None,
    era: str = None,
    authentic_emission_min: float = None,
    manufacturing_max: float = None,
    tags: list = None,
    text: str = None,         # Search context and creator statements
    limit: int = 20,
) -> list[SeedbankRecord]:
    """
    Search the seedbank by any combination of parameters.
    Returns list of matching SeedbankRecords ordered by relevance.
    """
    pass

def get_baseline(era: str = None, genre: str = None) -> dict:
    """
    Compute the baseline feature averages for a given era/genre
    from all matching seedbank records.
    This is what the psychosomatic profiler uses as its reference.
    Returns dict of feature_name -> mean value
    """
    pass

def compare(profile_id: str, target_id: str) -> dict:
    """
    Direct comparison between two seedbank records.
    Returns delta on every feature axis with interpretation.
    """
    pass

def get_most_authentic(limit: int = 10) -> list[SeedbankRecord]:
    """Return top N records by authentic_emission_score."""
    pass

def get_highest_lsii(limit: int = 10) -> list[SeedbankRecord]:
    """Return top N records by LSII score."""
    pass
```

### index.py specification

```python
def rebuild_index() -> dict:
    """
    Scan all records in seedbank/records/ and rebuild the index.
    The index is a JSON file with summary data for all records,
    enabling fast search without loading full profiles.
    """
    pass

def get_stats() -> dict:
    """
    Return seedbank statistics:
    - Total records
    - Records by era
    - Average LSII across corpus
    - Average authentic_emission_score
    - Most common manufacturing markers
    - Distribution of LSII flag levels
    """
    pass
```

### Storage format

All records stored as individual JSON files in `seedbank/records/`
Master index at `seedbank/index.json`
No database dependency — pure JSON, portable, human readable.

---

## MODULE 6: Corpus Temporal Analyser
**File:** `core/corpus_analyser.py`
**Priority:** MEDIUM — the social health arc over time

### What it does
Takes a dated corpus of music and reveals how a culture's creative output has
changed over time. This is the social EEG — watching collective wellbeing
and collective constraint in the frequency record.

### Specification

```python
@dataclass
class CorpusTemporalProfile:
    """
    The aggregate creative health of a corpus over time.
    Each entry in the time series represents one period (year, decade, etc).
    """
    periods: list              # List of period labels
    period_boundaries: list    # (start_year, end_year) per period
    
    # Health metrics over time
    lsii_trend: list           # Average LSII per period
    authentic_emission_trend: list
    manufacturing_trend: list
    creative_residue_trend: list
    dynamic_range_trend: list  # The loudness war in data
    
    # Emotional arc
    valence_trend: list
    arousal_trend: list
    tension_trend: list
    complexity_trend: list
    
    # Technical trends
    groove_deviation_trend: list   # Human timing over time
    compression_trend: list        # Crest factor over time
    harmonic_complexity_trend: list
    
    # Derived indices
    creative_freedom_index: list   # Composite: authentic emission + dynamics + groove
    cultural_pressure_index: list  # Composite: manufacturing + compression + quantisation
    
    # The SOS detection
    sos_periods: list              # Periods showing SOS signature
    sos_confidence: list           # Confidence per flagged period
    
    # Predictions
    next_period_prediction: dict
    prediction_confidence: float
    prediction_reasoning: str
    
    # Narrative
    trend_summary: str
    inflection_points: list        # Notable shifts with dates and descriptions

def analyse_corpus(
    profiles: list,         # List of analysis dicts from generate_json_report()
    year_field: str = 'year',  # Which metadata field contains the year
    period_size: int = 5,      # Years per period (5 = 5-year buckets)
) -> CorpusTemporalProfile:
    """
    Compute temporal trends across a corpus of dated profiles.
    profiles must each contain year metadata.
    Returns a CorpusTemporalProfile.
    """
    pass
```

### The SOS signature detection

```python
def _detect_sos_signature(period_data: dict) -> tuple[bool, float, str]:
    """
    The SOS signature in collective creative output:
    
    HIGH CONFIDENCE SOS (all present):
    - Dynamic range declining (loudness war increasing)
    - Groove deviation declining (quantisation increasing)  
    - Creative residue declining (formula work increasing)
    - LSII increasing (more late-song inversions = more hidden protests)
    - Harmonic complexity declining (simplification = cognitive capture)
    - Manufacturing score increasing
    
    The SOS is not a scream. It is a narrowing. 
    The progressive absence of authentic signal IS the signal.
    
    Returns (is_sos, confidence, description)
    """
    pass
```

### Visualisation output

The corpus analyser should output a rich HTML visualisation:
- Multi-line chart of all health metrics over time
- SOS periods highlighted
- Inflection points annotated
- The creative freedom index and cultural pressure index as the headline charts
- Predictions shown as dashed continuation lines

---

## MODULE 7: Influence Chain Mapper
**File:** `core/influence_mapper.py`
**Priority:** LOW — the lineage layer

### What it does
Identifies which earlier works a piece is in dialogue with.
Not plagiarism detection — **lineage mapping**.
This sound descends from this tradition via these intermediaries.

This is the elder function made technical: showing where things came from.

### Specification

```python
@dataclass
class InfluenceNode:
    """One node in the influence chain."""
    description: str           # e.g. "Delta blues, 1930s Mississippi"
    time_period: str
    sonic_characteristics: list # What features link to this
    cultural_context: str      # What was happening when this sound emerged
    confidence: float

@dataclass
class InfluenceChain:
    """The traceable lineage of sonic choices in a piece."""
    primary_lineage: list[InfluenceNode]  # Main tradition this descends from
    secondary_lineages: list[list[InfluenceNode]]  # Other traceable threads
    innovation_points: list[str]  # Where this piece departs from lineage
    confluence_points: list[str]  # Where multiple lineages merge
    narrative: str               # The lineage story in plain language

def map_influence_chain(features: 'SegmentFeatures',
                        seedbank_baseline: dict = None) -> InfluenceChain:
    """
    Map the detectable sonic lineage of a piece.
    seedbank_baseline: optional — uses seedbank era profiles if provided.
    """
    pass
```

### Initial lineage reference library

Build a JSON reference library at `fingerprints/lineages.json` with profiles for:
- Delta Blues (1920s-1940s)
- Chicago Blues (1940s-1960s)
- Gospel / Soul roots
- Early Rock and Roll (1950s)
- British Invasion (1960s)
- Psychedelia (1966-1969)
- Funk (1970-1975)
- Early Electronic / Krautrock (1970s)
- Disco (1974-1979)
- Post-Punk / New Wave (1978-1984)
- Early Hip-Hop (1979-1988)
- House (1985-1991)
- Techno (1988-1994)
- Grunge / Alternative (1991-1996)
- UK Rave / Jungle / D&B (1991-1997)
- Trip-Hop (1992-1998)
- IDM (1992-present)
- Minimal Techno (1999-2007)
- Psytrance (1995-present)
- UK Grime (2002-present)
- Post-Dubstep / Bass Music (2009-present)
- Contemporary R&B (2012-present)
- Hyperpop (2019-present)

Each profile: key harmonic characteristics, typical tempo range, spectral
centroid range, dynamic range range, groove deviation characteristics,
typical instrumentation, cultural context (what was happening).

---

## MODULE 8: CLI Enhancements
**File:** `analyse.py` (update existing)
**Priority:** LOW — quality of life

### Add to the existing CLI

```python
# New flags
parser.add_argument('--stems', action='store_true',
    help='Run stem separation before analysis (requires demucs)')
parser.add_argument('--vocal-analysis', action='store_true',  
    help='Include deep vocal prosodic analysis')
parser.add_argument('--html', type=str,
    help='Output HTML report to this path')
parser.add_argument('--compare', type=str,
    help='Compare against a seedbank record ID')
parser.add_argument('--seedbank-stats', action='store_true',
    help='Show seedbank statistics and exit')
parser.add_argument('--corpus-trend', type=str,
    help='Generate temporal trend analysis for a corpus JSON file')
```

---

## TESTING REQUIREMENTS

Every module needs a test file in `tests/`. Use `pytest`.

### Minimum test coverage per module
- Happy path: normal input, verify output structure and types
- Edge cases: very short audio (<5s), silence, clipped audio, mono vs stereo
- Degraded input: missing optional parameters, fallback behaviour
- Integration: modules chaining together correctly

### Test data
Generate synthetic test audio in `tests/fixtures/`:

```python
# tests/generate_fixtures.py
import numpy as np
import soundfile as sf

def generate_test_audio():
    """Generate minimal synthetic audio files for testing."""
    sr = 44100
    
    # 60 second sine wave at 440Hz (clean tone)
    t = np.linspace(0, 60, 60 * sr)
    sine_60s = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write('tests/fixtures/sine_60s.wav', sine_60s, sr)
    
    # 60 second piece with late-section change (for LSII testing)
    early = 0.5 * np.sin(2 * np.pi * 440 * t[:45*sr])
    late = 0.3 * np.sin(2 * np.pi * 880 * t[:15*sr])  # Higher, quieter
    lsii_test = np.concatenate([early, late])
    sf.write('tests/fixtures/lsii_test.wav', lsii_test, sr)
    
    # Heavy compression simulation (narrow dynamic range)
    compressed = np.clip(sine_60s * 3, -0.95, 0.95)
    sf.write('tests/fixtures/compressed.wav', compressed, sr)
    
    # With human timing variation (groove test)
    beats = np.zeros(60 * sr)
    for i in range(120):  # ~120 BPM with jitter
        pos = int(i * sr * 0.5 + np.random.normal(0, sr * 0.01))
        if 0 <= pos < len(beats) - 100:
            beats[pos:pos+100] = 0.8
    sf.write('tests/fixtures/groove_test.wav', beats, sr)

if __name__ == "__main__":
    generate_test_audio()
    print("Test fixtures generated.")
```

---

## INTEGRATION REQUIREMENTS

After all modules are built, update `analyse.py` to run the full integrated pipeline:

```python
def analyse_file_full(filepath, use_stems=False, vocal_analysis=False,
                      full_provenance=False, deposit=False, context=None,
                      html_output=None) -> dict:
    """
    Full pipeline:
    1. Ingestion
    2. Segmentation  
    3. [Optional] Stem separation
    4. Feature extraction (per stem if available, else mixed)
    5. [Optional] Vocal prosodic analysis
    6. Divergence / LSII
    7. Fingerprinting
    8. Psychosomatic profiling
    9. Report generation (JSON always, HTML if requested)
    10. [Optional] Seedbank deposit
    """
    pass
```

---

## CODE STANDARDS

### Style
- Python 3.10+
- Type hints everywhere
- Dataclasses for all data structures (no dicts as return types from functions)
- Every function has a docstring explaining WHAT it does and WHY
- Comments explain the reasoning, not the mechanics
  - BAD: `# multiply by sr to convert to samples`
  - GOOD: `# Convert to samples because librosa works in sample space not time`

### Philosophy comments
Every module should have a top-of-file docstring that explains the conceptual
purpose of the module — not just what it does technically but what it contributes
to the larger project. Future contributors need to understand the *why*.

### Error handling
- Never crash silently
- Always degrade gracefully — analysis with less data is better than no analysis
- Log clearly what was skipped and why
- Return partial results with flags rather than raising exceptions in the main pipeline

### Dependencies
Only add dependencies that are genuinely necessary. Document why in requirements.txt.
The tool must be installable and runnable on a basic laptop.
Keep Demucs/torch as optional — the tool works without GPU.

---

## FILE STRUCTURE WHEN COMPLETE

```
kindpath_analyser/
├── analyse.py                      ✅ CLI orchestrator (update)
├── README.md                       ✅ (update with new capabilities)
├── requirements.txt                ✅ (update)
├── AGENTS.md                       ✅ This file
├── core/
│   ├── __init__.py                 ✅
│   ├── ingestion.py                ✅ BUILT
│   ├── segmentation.py             ✅ BUILT
│   ├── feature_extractor.py        ✅ BUILT
│   ├── divergence.py               ✅ BUILT
│   ├── fingerprints.py             ✅ BUILT
│   ├── stem_separator.py           🔨 BUILD - Module 1
│   ├── vocal_prosodics.py          🔨 BUILD - Module 2
│   ├── psychosomatics.py           🔨 BUILD - Module 3
│   └── corpus_analyser.py          🔨 BUILD - Module 6
│   └── influence_mapper.py         🔨 BUILD - Module 7
├── fingerprints/
│   ├── instruments.json            🔨 BUILD - populate reference library
│   ├── daws.json                   🔨 BUILD
│   ├── eras.json                   🔨 BUILD
│   ├── genres.json                 🔨 BUILD
│   └── lineages.json               🔨 BUILD - Module 7
├── seedbank/
│   ├── __init__.py                 🔨 BUILD
│   ├── deposit.py                  🔨 BUILD - Module 5
│   ├── query.py                    🔨 BUILD - Module 5
│   ├── index.py                    🔨 BUILD - Module 5
│   └── records/                    (empty, populated at runtime)
├── reports/
│   ├── __init__.py                 🔨 BUILD
│   ├── report_generator.py         🔨 BUILD - Module 4
│   └── templates/
│       └── report.html             🔨 BUILD - HTML template
└── tests/
    ├── __init__.py                 🔨 BUILD
    ├── generate_fixtures.py        🔨 BUILD
    ├── fixtures/                   🔨 BUILD (generated)
    ├── test_ingestion.py           🔨 BUILD
    ├── test_segmentation.py        🔨 BUILD
    ├── test_feature_extractor.py   🔨 BUILD
    ├── test_divergence.py          🔨 BUILD
    ├── test_fingerprints.py        🔨 BUILD
    ├── test_stem_separator.py      🔨 BUILD - Module 1
    ├── test_vocal_prosodics.py     🔨 BUILD - Module 2
    ├── test_psychosomatics.py      🔨 BUILD - Module 3
    ├── test_report_generator.py    🔨 BUILD - Module 4
    ├── test_seedbank.py            🔨 BUILD - Module 5
    └── test_corpus_analyser.py     🔨 BUILD - Module 6
```

---

## FINAL NOTE TO THE AGENT

This project is not a technology project wearing educational clothing.
It is an act of cultural restoration wearing a technical implementation.

Every decision you make — what to name a variable, how to phrase a docstring,
what to include in an error message — should reflect this. The code is clear
and well-documented not because that is good practice but because **the knowledge
belongs to everyone and must be transferable**.

When you read the existing modules, you will see comments that explain not just
how things work but why they matter psychosomatically, culturally, historically.
Continue that. The code is the documentation. The documentation is the education.

Build it like you are leaving it for someone who needs to understand
not just what it does, but why it exists.

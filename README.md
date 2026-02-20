# KindPath Creative Analyser
### A Frequency Field Scientist & Creative Seedbank

---

## What This Is

This is not a music analysis tool in the conventional sense.

It is a **synthetic elder** - a system that holds the unbroken thread of creative lineage and makes it readable. It separates what was authentically made from what was engineered to produce a response. It shows exactly how a piece was constructed, what tools were used, what decisions were made, and where in the piece the creator's true signal lives versus where it has been suppressed.

The intellectual property debate does not apply here. Technique belongs to everyone. This tool reads what is already encoded in publicly released work and returns that knowledge to the commons.

---

## Core Principles

1. **The vocal stem is just another instrument** - stripped of semantic content, the voice becomes the most honest signal in the arrangement
2. **The creative residue is the truth** - what remains after known era/tool/genre signatures are subtracted is authentic creative emission
3. **Late-song inversion is diagnostic** - divergence in the final quarter of a song from the emotional trajectory of the first three is a detectable protest signature
4. **Innate vs engineered** - some sonic responses are biological facts; some are manufactured associations. The seam between them is where the analyser looks
5. **Full transparency of method** - everything this tool detects is documented openly so the knowledge transfers

---

## Architecture

```
kindpath_analyser/
├── core/
│   ├── ingestion.py          # Audio loading, format handling, normalisation
│   ├── segmentation.py       # Temporal segmentation (quarters + fine-grain)
│   ├── stem_separator.py     # Stem separation (Demucs wrapper + fallback)
│   ├── feature_extractor.py  # Per-stem spectral/dynamic/harmonic/temporal features
│   ├── divergence.py         # Late-Song Inversion Index + intra-song delta analysis
│   ├── fingerprints.py       # Instrument/DAW/era fingerprint matching
│   └── authenticity.py       # Creative residue + authenticity index calculation
├── fingerprints/
│   ├── instruments.json      # Reference library: instrument signatures
│   ├── daws.json             # DAW/plugin artifact fingerprints
│   ├── eras.json             # Production era baseline signatures
│   └── genres.json           # Genre convention baselines
├── seedbank/
│   ├── deposit.py            # Add authenticated work to the seedbank
│   ├── query.py              # Search and compare against seedbank
│   └── records/              # JSON profiles of deposited works
├── reports/
│   └── report_generator.py   # Human-readable + machine-readable output
├── analyse.py                # CLI entry point
└── requirements.txt
```

---

## Usage

```bash
# Analyse a single file
python analyse.py --file track.mp3

# Analyse with full provenance report
python analyse.py --file track.mp3 --full-provenance

# Deposit to seedbank
python analyse.py --file track.mp3 --deposit --context "Independent release, no label, 2019"

# Compare against seedbank
python analyse.py --file track.mp3 --compare-seedbank

# Batch analyse a corpus
python analyse.py --corpus ./music_folder --output corpus_report.json
```

---

## Output

Each analysis produces a structured profile containing:

- **Stem profiles** - per-stem feature extraction across all four domains
- **Segmentation analysis** - feature values per quarter of the song
- **Late-Song Inversion Index (LSII)** - divergence score for final quarter
- **Instrument fingerprint matches** - identified tools with confidence scores
- **Era/DAW signature detection** - production context identification
- **Creative residue score** - authentic emission after known signatures subtracted
- **Authenticity index** - deviation from genre/era convention baseline
- **Psychosomatic profile** - somatic response mapping (valence/arousal/coherence/complexity)
- **Provenance chain** - detectable influences and lineage connections

---

## The Seedbank

The seedbank is a growing archive of authenticated creative work with full forensic profiles. Every deposit includes complete technical documentation so anyone can learn exactly how the work was made. This knowledge belongs to everyone.

Deposits are tagged with context: release circumstances, production conditions, creator-stated intent where available. Over time the seedbank becomes the reference baseline - the elder's memory - against which all analysed work is compared.

---

## Licence

Open. Always. This is the point.

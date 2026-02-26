# Contributing to kindpath-analyser

kindpath-analyser is the KindPath Creative Analyser — a frequency field scientist and creative seedbank. It analyses acoustic signatures, maps divergence, extracts fingerprints, and builds a living record of creative resonance.

## Before You Contribute

Read [AGENTS.md](./AGENTS.md) — the detailed agent instructions and module roadmap.
Read [KINDFIELD.md](https://github.com/S4mu3lD4v1d/KindField/blob/main/KINDFIELD.md) for the epistemological foundation.

## Development Setup

```bash
git clone https://github.com/KindPath-Collective/kindpath-analyser.git
cd kindpath-analyser
pip install -r requirements.txt
python analyse.py --help
```

## Core Module Structure

```
core/
  __init__.py        — Package init
  ingestion.py       — Audio file loading and validation
  segmentation.py    — Beat, phrase, and section detection
  feature_extractor.py — Spectral, rhythmic, and timbral features
  divergence.py      — Divergence measurement between audio segments
  fingerprints.py    — Acoustic fingerprint generation and matching
```

The next module to implement is `core/stem_separator.py` — see AGENTS.md for specification.

## Code Standards

- Python 3.10+
- Type hints on all public functions and classes
- Docstrings on all public classes and methods
- Pydantic models for data structures where appropriate
- Pytest for all new modules
- No hardcoded file paths — all paths via configuration or arguments

## Pull Request Process

1. Ensure `python analyse.py` runs without errors
2. Add tests in `tests/` for any new `core/` module
3. Include in your PR: what module was added, what audio analysis capability it unlocks, and what is still unknown

## KindPath Domain Language in Code

- `fingerprint` — not just acoustic hash; the unique resonance signature of a creative work
- `divergence` — not just difference; the measured distance between creative fields
- `seedbank` — the growing record of fingerprinted creative works
- `frequency_field` — the combined spectral-emotional-structural landscape of audio

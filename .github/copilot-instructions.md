# Copilot Instructions — kindpath-analyser

## What This Repository Is

kindpath-analyser is the KindPath Creative Analyser — a frequency field scientist and creative seedbank. It analyses acoustic signatures, maps creative divergence, extracts fingerprints, and builds a living record of creative resonance.

Read `AGENTS.md` for the full specification and module roadmap before making any changes.
KINDFIELD foundation: https://github.com/S4mu3lD4v1d/KindField

## Stack

- **Language:** Python 3.10+
- **Audio processing:** librosa, numpy, scipy, soundfile (see requirements.txt)
- **Data modelling:** Pydantic
- **Testing:** pytest
- **Entry point:** `analyse.py`

## Core Module Roadmap (from AGENTS.md)

| Module | Status | Priority |
|--------|--------|----------|
| `core/ingestion.py` | ✅ Done | — |
| `core/segmentation.py` | ✅ Done | — |
| `core/feature_extractor.py` | ✅ Done | — |
| `core/divergence.py` | ✅ Done | — |
| `core/fingerprints.py` | ✅ Done | — |
| `core/stem_separator.py` | ⏳ Next | 🔴 Critical |

## Domain Language

- `fingerprint` — unique resonance signature of a creative work
- `divergence` — measured distance between creative fields
- `seedbank` — the growing record of fingerprinted creative works
- `frequency_field` — combined spectral-emotional-structural landscape of audio
- `stem` — isolated audio component (vocals, drums, bass, other)

## Code Standards

- Type hints on all public functions
- Pydantic models for data structures
- Docstrings on all public classes and methods
- Tests for all new `core/` modules (pytest)
- No hardcoded file paths

## What Not To Do

- Do not modify existing `core/` modules without explicit instruction
- Do not modify `AGENTS.md` or `README.md`
- Do not add dependencies not documented in `requirements.txt`
- Do not skip tests — every new core module needs a test file

#!/usr/bin/env python3
"""
generate_fixtures.py — Generate synthetic audio test fixtures for kindpath-analyser.

Run once before running the test suite:
    python3 tests/generate_fixtures.py

Generates:
    tests/fixtures/sine_60s.wav        — Clean 440Hz tone, 60 seconds
    tests/fixtures/lsii_test.wav       — Early section 440Hz, late section 880Hz (quieter)
    tests/fixtures/compressed.wav      — Heavily clipped/compressed signal
    tests/fixtures/groove_test.wav     — Drum-like beats with human timing jitter
    tests/fixtures/silence_5s.wav      — 5 seconds of silence (edge case)
    tests/fixtures/short_3s.wav        — 3 second clip (edge case: < minimum segment)
    tests/fixtures/chord_test.wav      — Chords: C major → A minor → G → F (harmonic test)
"""

import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SR = 44100


def _write(filename: str, audio: np.ndarray, sr: int = SR) -> None:
    path = os.path.join(FIXTURES_DIR, filename)
    sf.write(path, audio, sr)
    duration = len(audio) / sr
    print(f"  ✓ {filename} ({duration:.1f}s)")


def generate_sine_60s() -> None:
    """Clean 440Hz sine wave, 60 seconds."""
    t = np.linspace(0, 60, 60 * SR, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    _write("sine_60s.wav", audio)


def generate_lsii_test() -> None:
    """
    LSII test: first 45s at 440Hz, last 15s at 880Hz (higher, quieter).
    Q4 should differ significantly from Q1-Q3.
    """
    t_early = np.linspace(0, 45, 45 * SR, endpoint=False)
    t_late = np.linspace(0, 15, 15 * SR, endpoint=False)
    early = 0.5 * np.sin(2 * np.pi * 440 * t_early)
    late = 0.2 * np.sin(2 * np.pi * 880 * t_late)  # Higher pitch, lower amplitude
    audio = np.concatenate([early, late])
    _write("lsii_test.wav", audio)


def generate_compressed() -> None:
    """
    Heavily compressed signal — near-zero dynamic range.
    Simulates loudness war mastering.
    """
    t = np.linspace(0, 60, 60 * SR, endpoint=False)
    raw = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Overdrive then clip hard → very low crest factor
    compressed = np.clip(raw * 4.0, -0.98, 0.98)
    _write("compressed.wav", compressed)


def generate_groove_test() -> None:
    """
    Drum-like beats at ~120 BPM with human timing jitter.
    Groove deviation should be detectable vs quantised.
    """
    audio = np.zeros(60 * SR)
    bpm = 120
    beat_interval = SR * 60 / bpm   # samples per beat
    n_beats = int(60 * bpm / 60)

    rng = np.random.default_rng(seed=42)
    for i in range(n_beats):
        # Jitter ±12ms = ±529 samples at 44100Hz
        jitter = rng.normal(0, 0.012 * SR)
        pos = int(i * beat_interval + jitter)
        if 0 <= pos < len(audio) - 200:
            # Short transient burst simulating a kick drum
            decay = np.exp(-np.arange(200) / 20)
            audio[pos:pos + 200] += 0.7 * decay

    audio = np.clip(audio, -1.0, 1.0)
    _write("groove_test.wav", audio)


def generate_silence_5s() -> None:
    """5 seconds of near-silence (edge case for feature extractor)."""
    # Very faint noise floor rather than perfect silence (avoids log(0) in some features)
    rng = np.random.default_rng(seed=0)
    audio = rng.normal(0, 1e-6, 5 * SR)
    _write("silence_5s.wav", audio)


def generate_short_3s() -> None:
    """3 second clip — tests handling of very short audio."""
    t = np.linspace(0, 3, 3 * SR, endpoint=False)
    audio = 0.4 * np.sin(2 * np.pi * 440 * t)
    _write("short_3s.wav", audio)


def generate_chord_test() -> None:
    """
    Four-chord progression: C major → A minor → G major → F major.
    Each chord lasts 15 seconds. Total: 60 seconds.
    Tests harmonic analyser key/tension detection.
    """
    chord_freqs = {
        "C_major": [261.63, 329.63, 392.00],     # C4, E4, G4
        "A_minor": [220.00, 261.63, 329.63],     # A3, C4, E4
        "G_major": [196.00, 246.94, 293.66],     # G3, B3, D4
        "F_major": [174.61, 220.00, 261.63],     # F3, A3, C4
    }
    segments = []
    t = np.linspace(0, 15, 15 * SR, endpoint=False)
    for freqs in chord_freqs.values():
        chord = np.zeros(15 * SR)
        for f in freqs:
            chord += (1.0 / len(freqs)) * 0.4 * np.sin(2 * np.pi * f * t)
        segments.append(chord)
    audio = np.concatenate(segments)
    _write("chord_test.wav", audio)


def main() -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    print(f"Generating test fixtures in: {FIXTURES_DIR}\n")
    generate_sine_60s()
    generate_lsii_test()
    generate_compressed()
    generate_groove_test()
    generate_silence_5s()
    generate_short_3s()
    generate_chord_test()
    print(f"\nDone. {len(os.listdir(FIXTURES_DIR))} fixture files ready.")


if __name__ == "__main__":
    main()

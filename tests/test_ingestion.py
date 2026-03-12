"""
Tests for core/ingestion.py — AudioRecord loading and metadata extraction.
"""
import os
import sys
import pytest
import numpy as np

# Add parent directory to path so 'core' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {name}. Run tests/generate_fixtures.py first.")
    return path


# ── Import ──────────────────────────────────────────────────────────────────

from core.ingestion import load, AudioRecord


# ── Tests ───────────────────────────────────────────────────────────────────

class TestAudioRecord:

    def test_load_sine_returns_audiorecord(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert isinstance(record, AudioRecord)

    def test_load_duration(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert abs(record.duration_seconds - 60.0) < 1.0

    def test_mono_array_present(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert record.y_mono is not None
        assert isinstance(record.y_mono, np.ndarray)
        assert len(record.y_mono) > 0

    def test_sample_rate(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert record.sample_rate > 0

    def test_lsii_fixture_loads(self):
        path = _fixture('lsii_test.wav')
        record = load(path)
        assert abs(record.duration_seconds - 60.0) < 1.0

    def test_short_clip_loads(self):
        path = _fixture('short_3s.wav')
        record = load(path)
        assert record.duration_seconds < 5.0

    def test_silence_loads(self):
        path = _fixture('silence_5s.wav')
        record = load(path)
        assert record.duration_seconds > 0

    def test_compressed_clip_detection(self):
        path = _fixture('compressed.wav')
        record = load(path)
        # Compressed file should have high peak amplitude or clipping flag
        assert record.peak_amplitude > 0.0

    def test_to_dict_excludes_audio_arrays(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        d = record.to_dict()
        assert 'y_mono' not in d or d.get('y_mono') is None
        assert 'y_stereo' not in d or d.get('y_stereo') is None

    def test_rms_amplitude_nonzero_for_sine(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert record.rms_amplitude > 0.0

    def test_metadata_fields_present(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        assert record.filepath == path
        assert record.filename
        assert record.format

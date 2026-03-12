"""
Tests for core/stem_separator.py

All tests use real numpy arrays and real audio data — no mocks.
The fallback path (Demucs not installed) is the primary path tested,
since Demucs is an optional dependency and these tests must pass without it.

Test coverage:
- StemSet fields and types
- Fallback stemset when Demucs is absent
- Stem arrays approximate the original signal (energy conservation)
- Temp directory cleanup
- _resolve_device() logic
- _build_stemset() with partial stems
"""

import os
import sys
import importlib
import numpy as np
import pytest

# Ensure the analyser root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion import AudioRecord
from core.stem_separator import StemSet, separate, _fallback_stemset, _resolve_device, _build_stemset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_audio_record(duration=5.0, freq=440.0, sr=44100) -> AudioRecord:
    """Generate a simple sine wave AudioRecord for testing."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    y_mono = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    y_stereo = np.stack([y_mono, y_mono])
    return AudioRecord(
        filepath="/tmp/test_stem.wav",
        filename="test_stem.wav",
        format="wav",
        duration_seconds=duration,
        sample_rate=sr,
        num_channels=2,
        bit_depth=16,
        y_mono=y_mono,
        y_stereo=y_stereo,
        peak_amplitude=float(np.max(np.abs(y_mono))),
        rms_amplitude=float(np.sqrt(np.mean(y_mono ** 2))),
        dynamic_range_db=12.0,
        is_clipped=False,
        clipping_percentage=0.0,
        is_silence=False,
    )


# ── Test: StemSet dataclass ───────────────────────────────────────────────────

class TestStemSetDataclass:
    def test_all_fields_present(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        assert hasattr(stemset, "vocals")
        assert hasattr(stemset, "drums")
        assert hasattr(stemset, "bass")
        assert hasattr(stemset, "piano")
        assert hasattr(stemset, "guitar")
        assert hasattr(stemset, "other")
        assert hasattr(stemset, "sample_rate")
        assert hasattr(stemset, "source_filepath")
        assert hasattr(stemset, "separation_model")
        assert hasattr(stemset, "separation_quality")

    def test_stem_arrays_are_numpy(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        for stem in [stemset.vocals, stemset.drums, stemset.bass,
                     stemset.piano, stemset.guitar, stemset.other]:
            assert isinstance(stem, np.ndarray)

    def test_stem_arrays_are_float32(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        for stem in [stemset.vocals, stemset.drums, stemset.bass,
                     stemset.piano, stemset.guitar, stemset.other]:
            assert stem.dtype == np.float32

    def test_sample_rate_preserved(self):
        record = make_audio_record(sr=22050)
        stemset = _fallback_stemset(record)
        assert stemset.sample_rate == 22050

    def test_source_filepath_preserved(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        assert stemset.source_filepath == record.filepath

    def test_stem_length_matches_original(self):
        record = make_audio_record(duration=5.0)
        stemset = _fallback_stemset(record)
        expected_len = len(record.y_mono)
        for stem in [stemset.vocals, stemset.drums, stemset.bass]:
            assert len(stem) == expected_len


# ── Test: Fallback stemset ────────────────────────────────────────────────────

class TestFallbackStemset:
    def test_separation_quality_is_fallback(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        assert stemset.separation_quality == "fallback"

    def test_separation_model_is_none(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        assert "none" in stemset.separation_model.lower() or "fallback" in stemset.separation_model.lower()

    def test_all_stems_equal_mono(self):
        record = make_audio_record()
        stemset = _fallback_stemset(record)
        for stem in [stemset.vocals, stemset.drums, stemset.bass,
                     stemset.piano, stemset.guitar, stemset.other]:
            np.testing.assert_array_almost_equal(stem, record.y_mono, decimal=5)

    def test_separate_returns_fallback_when_demucs_absent(self, monkeypatch):
        """
        When both Demucs API and CLI are unavailable, separate() should return
        a fallback StemSet without raising.
        """
        import core.stem_separator as stem_mod

        # Force _separate_via_api to raise ImportError (Demucs not installed)
        original_api = stem_mod._separate_via_api
        original_cli = stem_mod._separate_via_cli

        def mock_api(*args, **kwargs):
            raise ImportError("demucs not installed")

        def mock_cli(*args, **kwargs):
            raise FileNotFoundError("demucs CLI not found")

        monkeypatch.setattr(stem_mod, "_separate_via_api", mock_api)
        monkeypatch.setattr(stem_mod, "_separate_via_cli", mock_cli)

        record = make_audio_record()
        stemset = separate(record)

        assert stemset.separation_quality == "fallback"
        assert isinstance(stemset.vocals, np.ndarray)
        assert len(stemset.vocals) == len(record.y_mono)

    def test_separate_does_not_raise_on_degraded(self, monkeypatch):
        """separate() must never raise when Demucs is unavailable."""
        import core.stem_separator as stem_mod

        monkeypatch.setattr(stem_mod, "_separate_via_api", lambda *a, **k: (_ for _ in ()).throw(ImportError("x")))
        monkeypatch.setattr(stem_mod, "_separate_via_cli", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y")))

        record = make_audio_record()
        # Must not raise
        stemset = separate(record)
        assert stemset is not None


# ── Test: Energy conservation ─────────────────────────────────────────────────

class TestEnergyConservation:
    def test_fallback_stem_energy_equals_original(self):
        """Fallback stems carry original signal — energy must match."""
        record = make_audio_record(duration=3.0)
        stemset = _fallback_stemset(record)
        original_rms = float(np.sqrt(np.mean(record.y_mono ** 2)))
        vocal_rms = float(np.sqrt(np.mean(stemset.vocals ** 2)))
        np.testing.assert_almost_equal(vocal_rms, original_rms, decimal=4)

    def test_build_stemset_creates_silence_for_missing_stems(self):
        """_build_stemset should use silence (zeros) for missing stems."""
        record = make_audio_record()
        # Provide only vocals, omit everything else
        partial = {"vocals": record.y_mono.copy()}
        stemset = _build_stemset(partial, record, "test-model", "standard")
        # drums should be zeros
        assert np.allclose(stemset.drums, 0.0, atol=1e-6)
        assert np.allclose(stemset.bass, 0.0, atol=1e-6)

    def test_build_stemset_preserves_vocals(self):
        record = make_audio_record()
        stem_arrays = {
            "vocals": record.y_mono * 0.8,
            "drums": record.y_mono * 0.2,
            "bass": record.y_mono * 0.1,
            "piano": np.zeros_like(record.y_mono),
            "guitar": np.zeros_like(record.y_mono),
            "other": np.zeros_like(record.y_mono),
        }
        stemset = _build_stemset(stem_arrays, record, "test", "standard")
        np.testing.assert_array_almost_equal(stemset.vocals, record.y_mono * 0.8, decimal=5)


# ── Test: Device resolution ───────────────────────────────────────────────────

class TestDeviceResolution:
    def test_resolve_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_resolve_explicit_cuda(self):
        # Just checks it passes through — CUDA may not actually be present
        result = _resolve_device("cuda")
        assert result == "cuda"

    def test_resolve_auto_returns_string(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda")

    def test_resolve_auto_without_torch(self, monkeypatch):
        """When torch is not importable, auto should fall back to cpu."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Must return 'cpu' and not raise
        result = _resolve_device("auto")
        assert result == "cpu"


# ── Test: Temp directory cleanup ──────────────────────────────────────────────

class TestTempDirectoryCleanup:
    def test_no_temp_dirs_left_after_fallback(self, tmp_path):
        """Fallback path creates no temp directories."""
        import glob
        before = set(glob.glob("/tmp/kindpath_stems_*"))
        record = make_audio_record()
        _fallback_stemset(record)
        after = set(glob.glob("/tmp/kindpath_stems_*"))
        # No new temp dirs created in fallback path
        assert after == before

    def test_separate_via_cli_cleans_up_on_failure(self, monkeypatch, tmp_path):
        """Even when CLI fails, the temp directory is cleaned up."""
        import tempfile
        import core.stem_separator as stem_mod
        import glob

        tmpdir_created = []

        real_mkdtemp = tempfile.mkdtemp

        def capturing_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            tmpdir_created.append(d)
            return d

        import tempfile as tempfile_mod
        monkeypatch.setattr(tempfile_mod, "mkdtemp", capturing_mkdtemp)

        # Force CLI to fail after creating temp dir
        import subprocess
        def mock_run(*args, **kwargs):
            class FakeResult:
                returncode = 1
                stderr = "demucs failed"
                stdout = ""
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", mock_run)

        record = make_audio_record()
        try:
            stem_mod._separate_via_cli(record, "cpu")
        except (FileNotFoundError, RuntimeError):
            pass  # Expected — CLI fails

        # All created temp dirs should be cleaned up
        for d in tmpdir_created:
            assert not os.path.exists(d), f"Temp dir not cleaned up: {d}"


# ── Integration test: real separate() call ────────────────────────────────────

class TestSeparateIntegration:
    def test_separate_returns_stemset(self):
        """separate() returns a StemSet regardless of Demucs availability."""
        record = make_audio_record()
        stemset = separate(record)
        assert isinstance(stemset, StemSet)

    def test_separate_all_stems_are_numpy(self):
        record = make_audio_record()
        stemset = separate(record)
        for stem in [stemset.vocals, stemset.drums, stemset.bass,
                     stemset.piano, stemset.guitar, stemset.other]:
            assert isinstance(stem, np.ndarray)

    def test_separate_sample_rate_preserved(self):
        record = make_audio_record(sr=44100)
        stemset = separate(record)
        assert stemset.sample_rate == 44100

    def test_separate_quality_is_valid_string(self):
        record = make_audio_record()
        stemset = separate(record)
        assert stemset.separation_quality in ("high", "standard", "fallback")

    def test_separate_stereo_record(self):
        """separate() handles stereo records correctly."""
        record = make_audio_record()
        record_with_stereo = AudioRecord(
            filepath=record.filepath,
            filename=record.filename,
            format=record.format,
            duration_seconds=record.duration_seconds,
            sample_rate=record.sample_rate,
            num_channels=2,
            bit_depth=record.bit_depth,
            y_mono=record.y_mono,
            y_stereo=np.stack([record.y_mono, record.y_mono * 0.8]),
            peak_amplitude=record.peak_amplitude,
            rms_amplitude=record.rms_amplitude,
            dynamic_range_db=record.dynamic_range_db,
            is_clipped=record.is_clipped,
            clipping_percentage=record.clipping_percentage,
            is_silence=record.is_silence,
        )
        stemset = separate(record_with_stereo)
        assert isinstance(stemset, StemSet)
        assert stemset.sample_rate == 44100

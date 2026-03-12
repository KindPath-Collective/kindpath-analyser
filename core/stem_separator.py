"""
KindPath Analyser :: Stem Separator

Separates a mixed audio file into isolated stems using Meta's Demucs model.
When stems are isolated, each one gets its own full feature extraction.
The vocal stem, stripped of language and read as pure signal, is the most
honest instrument in the mix. This module makes that separation possible.

If Demucs is not installed, the module degrades gracefully: all stems carry
the original mixed signal and separation_quality is marked 'fallback'.
The analysis pipeline continues — less precise, but not silent.

Demucs installation:
    pip install demucs
    # GPU (optional, recommended for large files):
    pip install torch torchaudio
"""

import os
import tempfile
import shutil
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.ingestion import AudioRecord

logger = logging.getLogger(__name__)


@dataclass
class StemSet:
    """
    The six stems from the Demucs htdemucs_6s model.
    Each stem is a numpy array of audio at the source sample rate.
    If a stem is absent or inaudible in the source, the array is near-silence
    (or the original mono signal if separation failed entirely).
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
    separation_quality: str  # 'high' (GPU), 'standard' (CPU), 'fallback' (failed)


def separate(
    record: AudioRecord,
    model: str = "htdemucs_6s",
    device: str = "auto",
) -> StemSet:
    """
    Separate stems from a loaded AudioRecord.

    device: 'auto' tries GPU first, falls back to CPU.

    If Demucs >= 4.0 is installed, uses demucs.api.
    If only the CLI is available, falls back to subprocess.
    If Demucs is entirely absent, returns a StemSet with all stems equal
    to the original mono signal and separation_quality='fallback'.

    Writes stems to a temp directory, loads them back, cleans up.
    Never raises on quality degradation — logs warnings instead.
    """
    # Try Demucs API (>= 4.0) first
    try:
        return _separate_via_api(record, model, device)
    except ImportError:
        pass  # Demucs not installed — try subprocess

    # Try Demucs CLI as subprocess fallback
    try:
        return _separate_via_cli(record, device)
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        logger.warning(f"[STEM] Demucs unavailable ({e}). Using fallback: all stems = mixed signal.")
        return _fallback_stemset(record)


# ── Demucs API path (Demucs >= 4.0) ──────────────────────────────────────────

def _separate_via_api(record: AudioRecord, model: str, device: str) -> StemSet:
    """Use demucs.api.Separator (Demucs 4.0+)."""
    try:
        import demucs.api as demucs_api
    except ImportError:
        raise ImportError("demucs not installed. Run: pip install demucs")

    try:
        import torch
        actual_device = _resolve_device(device)
    except ImportError:
        actual_device = "cpu"
        logger.warning("[STEM] torch not installed — using CPU-only mode")

    print(f"[STEM] Separating via Demucs API (model={model}, device={actual_device})")

    tmpdir = tempfile.mkdtemp(prefix="kindpath_stems_")
    try:
        separator = demucs_api.Separator(model=model, device=actual_device)

        # Demucs API expects waveform as tensor: (channels, samples)
        audio = record.y_stereo if record.y_stereo is not None else np.stack([record.y_mono, record.y_mono])
        import torch
        waveform = torch.from_numpy(audio.astype(np.float32))

        _, stems = separator.separate_tensor(waveform, record.sample_rate)
        # stems is a dict: {"vocals": tensor, "drums": tensor, ...}

        stem_arrays = {name: t.cpu().numpy().mean(axis=0) for name, t in stems.items()}
        quality = "high" if actual_device != "cpu" else "standard"

        _print_stem_isolation_note()
        return _build_stemset(stem_arrays, record, model, quality)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Demucs CLI path (subprocess fallback) ─────────────────────────────────────

def _separate_via_cli(record: AudioRecord, device: str) -> StemSet:
    """Invoke the Demucs CLI as a subprocess for environments without the Python API."""
    import subprocess
    import soundfile as sf

    # Verify demucs CLI is available
    result = subprocess.run(
        ["python", "-m", "demucs", "--help"],
        capture_output=True, timeout=30
    )
    if result.returncode != 0:
        raise FileNotFoundError("Demucs CLI not available")

    tmpdir = tempfile.mkdtemp(prefix="kindpath_stems_")
    try:
        # Write the audio to a temp file for Demucs to process
        tmp_audio_path = os.path.join(tmpdir, "input.wav")
        audio = record.y_stereo if record.y_stereo is not None else np.stack([record.y_mono, record.y_mono])
        sf.write(tmp_audio_path, audio.T, record.sample_rate)

        print(f"[STEM] Separating via Demucs CLI (htdemucs, device=auto)")
        cli_result = subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals",
             "--out", tmpdir, tmp_audio_path],
            capture_output=True, text=True, timeout=600
        )
        if cli_result.returncode != 0:
            raise RuntimeError(f"Demucs CLI failed: {cli_result.stderr[:500]}")

        # Load stems from output directory
        stem_arrays = _load_cli_stems(tmpdir, record)
        _print_stem_isolation_note()
        return _build_stemset(stem_arrays, record, "htdemucs (cli)", "standard")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _load_cli_stems(tmpdir: str, record: AudioRecord) -> dict:
    """
    Load stems written by the Demucs CLI.
    CLI output structure: {tmpdir}/{model_name}/input/{stem_name}.wav
    """
    import soundfile as sf

    mono = record.y_mono
    stem_arrays = {name: mono.copy() for name in ["vocals", "drums", "bass", "piano", "guitar", "other"]}

    for root, dirs, files in os.walk(tmpdir):
        for fname in files:
            if not fname.endswith(".wav"):
                continue
            stem_name = fname.replace(".wav", "")
            if stem_name in stem_arrays:
                path = os.path.join(root, fname)
                try:
                    data, _ = sf.read(path)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    stem_arrays[stem_name] = data.astype(np.float32)
                except Exception as e:
                    logger.warning(f"[STEM] Failed to load {fname}: {e}")

    # Two-stem mode produces 'vocals' and 'no_vocals'; map no_vocals → other
    if "no_vocals" in stem_arrays:
        stem_arrays["other"] = stem_arrays.pop("no_vocals")

    return stem_arrays


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_stemset(record: AudioRecord) -> StemSet:
    """
    When stem separation is impossible, return a StemSet where every stem
    carries the original mono signal. The analysis still runs — it just
    analyses the full mix for each stem position.
    """
    mono = record.y_mono.copy()
    return StemSet(
        vocals=mono,
        drums=mono,
        bass=mono,
        piano=mono,
        guitar=mono,
        other=mono,
        sample_rate=record.sample_rate,
        source_filepath=record.filepath,
        separation_model="none (fallback)",
        separation_quality="fallback",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_stemset(stem_arrays: dict, record: AudioRecord, model: str, quality: str) -> StemSet:
    """Build a StemSet from a dict of stem numpy arrays."""
    mono = record.y_mono

    def _get(name: str) -> np.ndarray:
        arr = stem_arrays.get(name)
        if arr is None:
            logger.warning(f"[STEM] Stem '{name}' not found — using silence")
            return np.zeros_like(mono)
        return arr.astype(np.float32)

    return StemSet(
        vocals=_get("vocals"),
        drums=_get("drums"),
        bass=_get("bass"),
        piano=_get("piano"),
        guitar=_get("guitar"),
        other=_get("other"),
        sample_rate=record.sample_rate,
        source_filepath=record.filepath,
        separation_model=model,
        separation_quality=quality,
    )


def _print_stem_isolation_note() -> None:
    print("[STEM] Vocals isolated — reading as pure tonal/timbral signal")
    print("[STEM] Language removed. What remains is what the body responds to.")

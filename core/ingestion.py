"""
KindPath Analyser :: Ingestion Module

Loads audio from any common format, normalises to a consistent internal
representation, and extracts basic metadata. This is the first act of
the analysis - receiving the work on its own terms before anything is
imposed on it.

Supported formats: MP3, WAV, FLAC, AIFF, OGG, M4A
"""

import librosa
import numpy as np
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AudioRecord:
    """
    The internal representation of an audio file.
    Everything downstream works from this.
    """
    filepath: str
    filename: str
    format: str
    duration_seconds: float
    sample_rate: int
    num_channels: int
    bit_depth: Optional[int]

    # The actual audio data
    # y_mono: mono mix for whole-track analysis
    # y_stereo: stereo if available, for spatial analysis
    y_mono: np.ndarray = None
    y_stereo: Optional[np.ndarray] = None

    # Loudness metadata
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    dynamic_range_db: float = 0.0

    # Integrity flags
    is_clipped: bool = False
    clipping_percentage: float = 0.0
    is_silence: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove raw audio arrays from serialised output
        d.pop('y_mono', None)
        d.pop('y_stereo', None)
        return d


def load(filepath: str, target_sr: int = 44100) -> AudioRecord:
    """
    Load an audio file and return an AudioRecord.

    target_sr: sample rate to resample to for consistent analysis.
    44100 is the standard. Don't change unless you have a reason.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file at: {filepath}")

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower().strip('.')

    # Load mono for analysis, stereo for spatial features
    y_mono, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Attempt stereo load
    y_stereo = None
    try:
        y_stereo, _ = librosa.load(filepath, sr=target_sr, mono=False)
        if y_stereo.ndim == 1:
            y_stereo = None  # Was mono all along
        num_channels = 2 if y_stereo is not None else 1
    except Exception:
        num_channels = 1

    duration = librosa.get_duration(y=y_mono, sr=sr)

    # Amplitude analysis
    peak = float(np.max(np.abs(y_mono)))
    rms = float(np.sqrt(np.mean(y_mono ** 2)))
    rms_db = librosa.amplitude_to_db(np.array([rms]))[0] if rms > 0 else -120.0
    peak_db = librosa.amplitude_to_db(np.array([peak]))[0] if peak > 0 else -120.0
    dynamic_range = float(peak_db - rms_db)

    # Clipping detection - samples at or very near 0dBFS
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(y_mono) >= clipping_threshold)
    clipping_pct = float(clipped_samples / len(y_mono) * 100)
    is_clipped = clipping_pct > 0.01  # More than 0.01% clipped is a flag

    # Silence detection
    is_silence = rms < 0.001

    record = AudioRecord(
        filepath=filepath,
        filename=filename,
        format=ext,
        duration_seconds=float(duration),
        sample_rate=int(sr),
        num_channels=num_channels,
        bit_depth=None,  # librosa doesn't expose this directly
        y_mono=y_mono,
        y_stereo=y_stereo,
        peak_amplitude=peak,
        rms_amplitude=rms,
        dynamic_range_db=dynamic_range,
        is_clipped=is_clipped,
        clipping_percentage=clipping_pct,
        is_silence=is_silence,
    )

    _print_load_report(record)
    return record


def _print_load_report(record: AudioRecord):
    print(f"\n[INGESTION] {record.filename}")
    print(f"  Format     : {record.format.upper()}")
    print(f"  Duration   : {record.duration_seconds:.2f}s ({record.duration_seconds/60:.2f} min)")
    print(f"  Sample Rate: {record.sample_rate} Hz")
    print(f"  Channels   : {record.num_channels}")
    print(f"  Peak       : {librosa.amplitude_to_db(np.array([record.peak_amplitude]))[0]:.1f} dBFS")
    print(f"  RMS        : {librosa.amplitude_to_db(np.array([record.rms_amplitude]))[0]:.1f} dBFS")
    print(f"  Dyn Range  : {record.dynamic_range_db:.1f} dB")
    if record.is_clipped:
        print(f"  ⚠ CLIPPING DETECTED: {record.clipping_percentage:.3f}% of samples")
    if record.is_silence:
        print(f"  ⚠ FILE APPEARS SILENT")

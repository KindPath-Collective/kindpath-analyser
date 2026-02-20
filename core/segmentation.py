"""
KindPath Analyser :: Segmentation Module

Divides the audio into analytical segments. This is critical architecture.
The Late-Song Inversion Index depends on comparing the final quarter against
the established emotional trajectory of the first three quarters.

We segment at two resolutions:
- MACRO: quarters of the total duration (the protest detection layer)
- MICRO: 4-second windows across the full track (the fine-grain texture layer)

We also detect structural boundaries using onset/energy analysis,
so the tool understands where sections actually begin and end
rather than imposing arbitrary time cuts.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple
from core.ingestion import AudioRecord


@dataclass
class Segment:
    """A time-bounded portion of the audio for isolated analysis."""
    label: str              # e.g. "Q1", "Q2", "Q3", "Q4", "intro", "verse_1"
    segment_type: str       # "quarter", "structural", "micro"
    start_time: float       # seconds
    end_time: float         # seconds
    start_sample: int
    end_sample: int
    y: np.ndarray           # audio data for this segment
    duration: float         # seconds


@dataclass  
class SegmentationResult:
    quarters: List[Segment]         # The 4 equal-time quarters
    structural: List[Segment]       # Energy-boundary detected sections
    micro_windows: List[Segment]    # Fine-grain 4s windows


def segment(record: AudioRecord, micro_window_seconds: float = 4.0) -> SegmentationResult:
    """
    Segment the audio at macro (quarter), structural, and micro levels.
    """
    y = record.y_mono
    sr = record.sample_rate
    duration = record.duration_seconds

    quarters = _segment_quarters(y, sr, duration)
    structural = _segment_structural(y, sr, duration)
    micro = _segment_micro(y, sr, micro_window_seconds)

    _print_segmentation_report(quarters, structural, len(micro))
    return SegmentationResult(quarters=quarters, structural=structural, micro_windows=micro)


def _segment_quarters(y: np.ndarray, sr: int, duration: float) -> List[Segment]:
    """
    Divide into four equal temporal quarters.
    Q1-Q3 establish the trajectory. Q4 is where the protest lives.
    """
    quarter_dur = duration / 4.0
    quarters = []

    for i in range(4):
        start_t = i * quarter_dur
        end_t = (i + 1) * quarter_dur
        start_s = int(start_t * sr)
        end_s = int(end_t * sr)
        end_s = min(end_s, len(y))

        quarters.append(Segment(
            label=f"Q{i+1}",
            segment_type="quarter",
            start_time=start_t,
            end_time=end_t,
            start_sample=start_s,
            end_sample=end_s,
            y=y[start_s:end_s],
            duration=quarter_dur,
        ))

    return quarters


def _segment_structural(y: np.ndarray, sr: int, duration: float) -> List[Segment]:
    """
    Detect section boundaries using spectral novelty / energy flux.
    This finds where the music actually changes, not where the clock ticks.
    """
    # Compute onset strength envelope
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Compute recurrence matrix for structure detection
    # Uses chroma features to find repeated sections
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    R = librosa.segment.recurrence_matrix(chroma, mode='affinity', sym=True)
    R_filtered = librosa.segment.path_enhance(R, 15)

    # Detect boundaries
    bounds = librosa.segment.agglomerative(R_filtered, k=min(8, max(2, int(duration / 30))))
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)

    # Build segments from boundaries
    all_times = np.concatenate([[0], bound_times, [duration]])
    all_times = np.unique(all_times)

    segments = []
    labels = ['intro', 'section_2', 'section_3', 'section_4',
              'section_5', 'section_6', 'section_7', 'outro']

    for i in range(len(all_times) - 1):
        start_t = float(all_times[i])
        end_t = float(all_times[i + 1])
        start_s = int(start_t * sr)
        end_s = min(int(end_t * sr), len(y))
        label = labels[i] if i < len(labels) else f"section_{i+1}"

        segments.append(Segment(
            label=label,
            segment_type="structural",
            start_time=start_t,
            end_time=end_t,
            start_sample=start_s,
            end_sample=end_s,
            y=y[start_s:end_s],
            duration=end_t - start_t,
        ))

    return segments


def _segment_micro(y: np.ndarray, sr: int, window_seconds: float) -> List[Segment]:
    """
    Sliding window segmentation for fine-grain texture analysis.
    No overlap - clean sequential windows across the full track.
    """
    window_samples = int(window_seconds * sr)
    segments = []
    idx = 0
    window_num = 0

    while idx + window_samples <= len(y):
        start_t = idx / sr
        end_t = (idx + window_samples) / sr
        segments.append(Segment(
            label=f"W{window_num:04d}",
            segment_type="micro",
            start_time=start_t,
            end_time=end_t,
            start_sample=idx,
            end_sample=idx + window_samples,
            y=y[idx:idx + window_samples],
            duration=window_seconds,
        ))
        idx += window_samples
        window_num += 1

    return segments


def _print_segmentation_report(quarters, structural, micro_count):
    print(f"\n[SEGMENTATION]")
    print(f"  Quarters   : 4 ({quarters[0].duration:.1f}s each)")
    print(f"  Structural : {len(structural)} sections detected")
    for s in structural:
        print(f"    {s.label:12s} {s.start_time:.1f}s - {s.end_time:.1f}s ({s.duration:.1f}s)")
    print(f"  Micro Win  : {micro_count} windows")

"""
KindPath Analyser :: Feature Extractor

The heart of the scientific instrument. Extracts features across four domains
for any given audio segment. Works on whole tracks, individual stems, or any
Segment object.

FOUR DOMAINS:

1. SPECTRAL  - what frequencies are present and how they behave
2. DYNAMIC   - loudness, compression, and energy behaviour
3. HARMONIC  - pitch, tonality, tension, resolution
4. TEMPORAL  - rhythm, groove, timing precision, syncopation

Each feature is chosen because it carries real information about the
emotional and creative state of the work - not just technical description
but signal about the human condition that produced it.
"""

import numpy as np
import librosa
from dataclasses import dataclass, asdict
from typing import Optional
from core.segmentation import Segment


@dataclass
class SpectralFeatures:
    """
    How the frequency content is distributed and behaves.
    Centroid rising = brightness, energy, aggression or openness.
    Centroid falling = warmth, retreat, weight, fatigue.
    Flux = rate of spectral change. High flux = dynamic, alive.
    Low flux in late sections = emotional flatness or resignation.
    """
    centroid_mean: float        # Brightness centre of mass (Hz)
    centroid_std: float         # Variability of brightness
    centroid_trend: float       # Direction over time (+= brightening, -= darkening)
    rolloff_mean: float         # Frequency below which 85% of energy sits
    flux_mean: float            # Rate of spectral change
    flux_std: float
    bandwidth_mean: float       # Spectral width - narrow = compressed, wide = open
    contrast_mean: float        # Peak vs valley ratio - higher = more defined
    flatness_mean: float        # 0=tonal, 1=noise-like
    mfcc: list                  # 13 MFCCs - the timbral fingerprint
    harmonic_ratio: float       # Harmonic vs percussive energy ratio


@dataclass
class DynamicFeatures:
    """
    How energy behaves over time.
    Dynamic range is political: hypercompression is the loudness war
    but it's also the sonic signature of a culture under pressure.
    Crest factor measures the ratio of peaks to average - 
    high = punchy and present, low = crushed and fatigued.
    """
    rms_mean: float             # Average energy
    rms_std: float              # Energy variability
    rms_trend: float            # Is it building or decaying?
    peak_mean: float            # Average peak level
    crest_factor_db: float      # Peak to RMS ratio (compression signature)
    dynamic_range_db: float     # Difference between loudest and quietest
    zero_crossing_rate: float   # Correlates with noisiness/transient density
    onset_density: float        # Events per second - busyness metric
    loudness_lufs: float        # Integrated loudness (perceptual)


@dataclass
class HarmonicFeatures:
    """
    Pitch and tonal organisation.
    Chroma tells us what notes are present regardless of octave.
    Tonality strength measures how 'in key' the music is -
    low tonality can mean jazz sophistication OR tonal confusion OR
    genuine harmonic risk-taking, which is one marker of authentic creativity.
    Tension ratio is the proportion of time spent in dissonant harmonic states.
    Resolution index measures whether tension is resolved or left hanging.
    """
    key_estimate: str           # Detected key (e.g. "C major", "A minor")
    key_confidence: float       # 0-1 confidence
    chroma_mean: list           # 12-element chroma vector
    chroma_std: list            # Variability per pitch class
    tonality_strength: float    # 0-1 how strongly tonal
    tuning_offset_cents: float  # Deviation from A440 (performance marker)
    harmonic_complexity: float  # Weighted chord change density
    tension_ratio: float        # Proportion in dissonant states
    resolution_index: float     # How often tension resolves


@dataclass
class TemporalFeatures:
    """
    Rhythm, timing, and groove.
    Groove deviation is the distance from perfect quantisation -
    human timing variation that makes music breathe.
    Low groove deviation = rigid, over-produced, or MIDI-quantised.
    High groove deviation = alive, human, present.
    Rhythmic entropy measures unpredictability in the onset pattern.
    Syncopation index measures off-beat emphasis.
    """
    tempo_bpm: float
    tempo_confidence: float
    beat_regularity: float      # 0-1 how metrically stable
    groove_deviation_ms: float  # RMS timing deviation from perfect grid (ms)
    rhythmic_entropy: float     # Unpredictability of onset pattern
    syncopation_index: float    # Off-beat emphasis ratio
    onset_strength_mean: float
    onset_strength_trend: float # Is rhythmic energy building or fading?
    meter_estimate: int         # 3 or 4 (or 0 if unclear)


@dataclass
class SegmentFeatures:
    """Complete feature set for one segment."""
    segment_label: str
    start_time: float
    end_time: float
    spectral: SpectralFeatures
    dynamic: DynamicFeatures
    harmonic: HarmonicFeatures
    temporal: TemporalFeatures

    def to_dict(self) -> dict:
        return {
            'segment_label': self.segment_label,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'spectral': asdict(self.spectral),
            'dynamic': asdict(self.dynamic),
            'harmonic': asdict(self.harmonic),
            'temporal': asdict(self.temporal),
        }


def extract(segment: Segment, sr: int) -> SegmentFeatures:
    """
    Extract all features from a segment.
    """
    y = segment.y
    if len(y) < sr * 0.5:  # Skip segments shorter than 0.5s
        return None

    spectral = _extract_spectral(y, sr)
    dynamic = _extract_dynamic(y, sr)
    harmonic = _extract_harmonic(y, sr)
    temporal = _extract_temporal(y, sr)

    return SegmentFeatures(
        segment_label=segment.label,
        start_time=segment.start_time,
        end_time=segment.end_time,
        spectral=spectral,
        dynamic=dynamic,
        harmonic=harmonic,
        temporal=temporal,
    )


def _extract_spectral(y: np.ndarray, sr: int) -> SpectralFeatures:
    hop = 512
    S = np.abs(librosa.stft(y, hop_length=hop))

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Harmonic/percussive separation
    H, P = librosa.decompose.hpss(S)
    harmonic_energy = np.mean(H ** 2)
    percussive_energy = np.mean(P ** 2)
    harmonic_ratio = float(harmonic_energy / (harmonic_energy + percussive_energy + 1e-10))

    # Trend: linear regression slope over time
    t = np.arange(len(centroid))
    centroid_trend = float(np.polyfit(t, centroid, 1)[0]) if len(t) > 1 else 0.0

    return SpectralFeatures(
        centroid_mean=float(np.mean(centroid)),
        centroid_std=float(np.std(centroid)),
        centroid_trend=centroid_trend,
        rolloff_mean=float(np.mean(rolloff)),
        flux_mean=float(np.mean(flux)),
        flux_std=float(np.std(flux)),
        bandwidth_mean=float(np.mean(bandwidth)),
        contrast_mean=float(np.mean(contrast)),
        flatness_mean=float(np.mean(flatness)),
        mfcc=[float(np.mean(mfcc[i])) for i in range(13)],
        harmonic_ratio=harmonic_ratio,
    )


def _extract_dynamic(y: np.ndarray, sr: int) -> DynamicFeatures:
    hop = 512

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    t = np.arange(len(rms))
    rms_trend = float(np.polyfit(t, rms, 1)[0]) if len(t) > 1 else 0.0

    # Peak
    peak_mean = float(np.mean(np.abs(y)))

    # Crest factor
    peak_val = float(np.max(np.abs(y)))
    rms_val = float(np.sqrt(np.mean(y ** 2)))
    if rms_val > 0 and peak_val > 0:
        crest_db = float(20 * np.log10(peak_val / rms_val))
    else:
        crest_db = 0.0

    # Dynamic range
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

    # Onset density
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    onset_density = float(len(onset_frames) / (len(y) / sr)) if len(y) > 0 else 0.0

    # Approximate LUFS (simplified - true LUFS requires K-weighting)
    loudness_lufs = float(librosa.amplitude_to_db(np.array([rms_val]))[0]) if rms_val > 0 else -120.0

    return DynamicFeatures(
        rms_mean=rms_mean,
        rms_std=rms_std,
        rms_trend=rms_trend,
        peak_mean=peak_mean,
        crest_factor_db=crest_db,
        dynamic_range_db=dynamic_range,
        zero_crossing_rate=float(np.mean(zcr)),
        onset_density=onset_density,
        loudness_lufs=loudness_lufs,
    )


def _extract_harmonic(y: np.ndarray, sr: int) -> HarmonicFeatures:
    hop = 512

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma_mean = [float(np.mean(chroma[i])) for i in range(12)]
    chroma_std = [float(np.std(chroma[i])) for i in range(12)]

    # Key estimation using Krumhansl-Schmuckler profiles
    key_idx, key_confidence = _estimate_key(chroma_mean)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Major vs minor discrimination (simplified)
    major_template = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    minor_template = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    chroma_norm = np.array(chroma_mean)
    if chroma_norm.max() > 0:
        chroma_norm = chroma_norm / chroma_norm.max()

    major_score = np.dot(chroma_norm, np.roll(major_template, key_idx))
    minor_score = np.dot(chroma_norm, np.roll(minor_template, key_idx))
    mode = "major" if major_score > minor_score else "minor"
    key_estimate = f"{note_names[key_idx]} {mode}"

    # Tonality strength: how concentrated energy is in key-scale tones
    if mode == "major":
        scale_mask = np.roll([1,0,1,0,1,1,0,1,0,1,0,1], key_idx).astype(bool)
    else:
        scale_mask = np.roll([1,0,1,1,0,1,0,1,1,0,0,1], key_idx).astype(bool)

    chroma_arr = np.array(chroma_mean)
    total_energy = np.sum(chroma_arr) + 1e-10
    in_scale_energy = np.sum(chroma_arr[scale_mask])
    tonality_strength = float(in_scale_energy / total_energy)

    # Tuning offset
    tuning = librosa.estimate_tuning(y=y, sr=sr)

    # Harmonic complexity: chroma flux as proxy for chord change rate
    chroma_flux = np.mean(np.diff(chroma, axis=1) ** 2)
    harmonic_complexity = float(chroma_flux)

    # Tension: energy in tritone and minor 2nd intervals (dissonant intervals)
    # Simplified: energy NOT in scale tones / total
    tension_ratio = float(1.0 - (in_scale_energy / total_energy))

    # Resolution: does tension decrease toward phrase ends?
    # Approximated by whether late chroma becomes more tonal
    if chroma.shape[1] > 4:
        early_chroma = chroma[:, :chroma.shape[1]//2]
        late_chroma = chroma[:, chroma.shape[1]//2:]
        early_tension = 1.0 - (np.sum(early_chroma[scale_mask]) / (np.sum(early_chroma) + 1e-10))
        late_tension = 1.0 - (np.sum(late_chroma[scale_mask]) / (np.sum(late_chroma) + 1e-10))
        resolution_index = float(early_tension - late_tension)  # Positive = resolving
    else:
        resolution_index = 0.0

    return HarmonicFeatures(
        key_estimate=key_estimate,
        key_confidence=float(key_confidence),
        chroma_mean=chroma_mean,
        chroma_std=chroma_std,
        tonality_strength=tonality_strength,
        tuning_offset_cents=float(tuning * 100),
        harmonic_complexity=harmonic_complexity,
        tension_ratio=tension_ratio,
        resolution_index=resolution_index,
    )


def _extract_temporal(y: np.ndarray, sr: int) -> TemporalFeatures:
    hop = 512

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempo = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

    # Beat regularity: std of inter-beat intervals (lower = more regular)
    if len(beats) > 2:
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
        ibi = np.diff(beat_times)
        ibi_std = float(np.std(ibi))
        ibi_mean = float(np.mean(ibi))
        beat_regularity = float(1.0 - min(ibi_std / (ibi_mean + 1e-10), 1.0))
        # Groove deviation: RMS timing deviation in milliseconds
        groove_deviation_ms = float(ibi_std * 1000)
        # Meter: if mean IBI ~0.5s then ~120bpm in 4/4
        meter_estimate = 4  # Simplified - full meter detection needs more
    else:
        beat_regularity = 0.5
        groove_deviation_ms = 0.0
        meter_estimate = 0

    # Rhythmic entropy: entropy of onset strength distribution
    onset_hist, _ = np.histogram(onset_env, bins=20, density=True)
    onset_hist = onset_hist + 1e-10
    rhythmic_entropy = float(-np.sum(onset_hist * np.log2(onset_hist)))

    # Syncopation: energy on off-beats
    if len(beats) > 2:
        beat_frames = set(beats)
        # Off-beat frames (halfway between beats)
        offbeat_frames = set([(beats[i] + beats[i+1]) // 2 for i in range(len(beats)-1)])
        beat_energy = np.mean([onset_env[min(b, len(onset_env)-1)] for b in beat_frames])
        offbeat_energy = np.mean([onset_env[min(b, len(onset_env)-1)] for b in offbeat_frames])
        syncopation_index = float(offbeat_energy / (beat_energy + 1e-10))
    else:
        syncopation_index = 0.5

    # Onset strength trend
    t = np.arange(len(onset_env))
    onset_trend = float(np.polyfit(t, onset_env, 1)[0]) if len(t) > 1 else 0.0

    return TemporalFeatures(
        tempo_bpm=tempo,
        tempo_confidence=min(float(np.max(onset_env)) / 10.0, 1.0),
        beat_regularity=beat_regularity,
        groove_deviation_ms=groove_deviation_ms,
        rhythmic_entropy=rhythmic_entropy,
        syncopation_index=syncopation_index,
        onset_strength_mean=float(np.mean(onset_env)),
        onset_strength_trend=onset_trend,
        meter_estimate=meter_estimate,
    )


def _estimate_key(chroma_mean: list) -> tuple:
    """
    Krumhansl-Schmuckler key estimation.
    Returns (key_index, confidence) where key_index is 0=C, 1=C#, etc.
    """
    # Krumhansl-Kessler key profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    chroma = np.array(chroma_mean)
    best_key = 0
    best_corr = -2.0

    for i in range(12):
        major_corr = np.corrcoef(chroma, np.roll(major_profile, i))[0, 1]
        minor_corr = np.corrcoef(chroma, np.roll(minor_profile, i))[0, 1]
        max_corr = max(major_corr, minor_corr)
        if max_corr > best_corr:
            best_corr = max_corr
            best_key = i

    confidence = (best_corr + 1) / 2  # Normalise -1..1 to 0..1
    return best_key, float(confidence)

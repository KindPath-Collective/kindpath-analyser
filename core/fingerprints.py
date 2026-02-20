"""
KindPath Analyser :: Fingerprint Module

Identifies the tools and techniques used to make a piece of music.
This is the provenance layer - the part that shows exactly how it was made.

Knowledge belongs to everyone. Technique is not proprietary.
What's detectable in the signal was put there by the creator
and released into the world with the work. We're building
better instruments to read it.

DETECTION CATEGORIES:

1. INSTRUMENT FINGERPRINTS
   Specific hardware instruments have characteristic signatures:
   - Harmonic overtone series (unique per instrument family)
   - Attack transient shapes
   - Noise floor characteristics
   - Resonance and decay curves

2. PRODUCTION ERA SIGNATURES
   Each decade of recorded music has characteristic artifacts:
   - Pre-1970s: tape saturation, limited high-frequency response
   - 1970s: analogue warmth, natural dynamic range
   - 1980s: gated reverb, digital sheen, early MIDI rigidity
   - 1990s: early digital artifacts, loudness creep begins
   - 2000s: loudness war peak, hypercompression, CD limiting
   - 2010s: streaming normalisation shift, bass-heavy mastering
   - 2020s: spatial audio experiments, lo-fi aesthetics as reaction

3. DAW/PLUGIN ARTIFACTS
   Software tools leave detectable signatures in the residual signal.
   Not perfect fingerprinting, but statistically informative.

4. PRODUCTION TECHNIQUE MARKERS
   Parallel compression, sidechain relationships, stereo field choices,
   automation patterns - all leave readable traces.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Dict, Optional
from core.ingestion import AudioRecord


@dataclass
class FingerprintMatch:
    category: str           # 'instrument', 'era', 'technique', 'daw'
    name: str               # What was matched
    confidence: float       # 0.0-1.0
    evidence: List[str]     # What features triggered this match
    description: str        # Human-readable explanation


@dataclass
class FingerprintReport:
    likely_instruments: List[FingerprintMatch]
    likely_era: List[FingerprintMatch]
    likely_techniques: List[FingerprintMatch]
    production_context: str
    authenticity_markers: List[str]
    manufacturing_markers: List[str]


# ─────────────────────────────────────────────
# ERA SIGNATURE PROFILES
# Each era has a signature in terms of measurable audio characteristics
# ─────────────────────────────────────────────

ERA_PROFILES = {
    "pre_1970": {
        "crest_factor_range": (12, 25),    # High crest = uncompressed
        "high_freq_rolloff": 10000,         # Limited high frequency
        "noise_floor_range": (-50, -35),    # Audible tape noise
        "dynamic_range_range": (15, 30),    # Wide dynamic range
        "description": "Pre-1970s analogue. Wide dynamics, tape saturation, limited frequency range."
    },
    "1970s": {
        "crest_factor_range": (10, 20),
        "high_freq_rolloff": 14000,
        "noise_floor_range": (-60, -45),
        "dynamic_range_range": (12, 25),
        "description": "1970s analogue warmth. Natural dynamics, analogue tape character."
    },
    "1980s": {
        "crest_factor_range": (8, 16),
        "high_freq_rolloff": 18000,
        "noise_floor_range": (-75, -55),
        "dynamic_range_range": (8, 18),
        "description": "1980s digital/analogue hybrid. Gated reverb, early digital brightness, MIDI rigidity."
    },
    "1990s": {
        "crest_factor_range": (6, 14),
        "high_freq_rolloff": 20000,
        "noise_floor_range": (-90, -65),
        "dynamic_range_range": (6, 14),
        "description": "1990s early digital. Clean noise floor, loudness creep beginning."
    },
    "2000s": {
        "crest_factor_range": (2, 8),      # Loudness war - crushed dynamics
        "high_freq_rolloff": 20000,
        "noise_floor_range": (-95, -70),
        "dynamic_range_range": (2, 8),     # Very narrow dynamic range
        "description": "2000s loudness war. Hypercompression, limited dynamic range, maximised loudness."
    },
    "2010s": {
        "crest_factor_range": (4, 10),     # Streaming began pulling back from peak loudness war
        "high_freq_rolloff": 20000,
        "noise_floor_range": (-95, -75),
        "dynamic_range_range": (5, 12),
        "description": "2010s streaming era. Moderate compression, bass-heavy mastering, loudness normalisation influence."
    },
    "2020s": {
        "crest_factor_range": (5, 12),
        "high_freq_rolloff": 20000,
        "noise_floor_range": (-95, -75),
        "dynamic_range_range": (6, 14),
        "description": "2020s. Spatial audio experiments, lo-fi aesthetics as authentic reaction to over-production."
    },
}


# ─────────────────────────────────────────────
# PRODUCTION TECHNIQUE MARKERS
# ─────────────────────────────────────────────

TECHNIQUE_SIGNATURES = {
    "heavy_compression": {
        "crest_factor_max": 6,
        "dynamic_range_max": 6,
        "description": "Heavy dynamic compression applied. Peaks and RMS are close together.",
        "psychosomatic_note": "Hypercompression creates perceptual fatigue. Associated with commercial pressure."
    },
    "natural_dynamics": {
        "crest_factor_min": 12,
        "dynamic_range_min": 12,
        "description": "Natural dynamic range preserved. Minimal compression applied.",
        "psychosomatic_note": "Preserved dynamics indicate production choices that prioritise listener experience over loudness competition."
    },
    "reverb_heavy": {
        "description": "Significant reverb/space processing detected via RT60 estimation.",
        "psychosomatic_note": "Heavy reverb creates distance and dreamlike states. Can indicate escapism or grandeur."
    },
    "quantised_rhythm": {
        "groove_deviation_max_ms": 5,
        "description": "Rhythmic performance tightly quantised to grid.",
        "psychosomatic_note": "Perfect quantisation removes human timing variation. The body misses the groove."
    },
    "human_performance": {
        "groove_deviation_min_ms": 15,
        "description": "Significant human timing variation detected - live performance markers present.",
        "psychosomatic_note": "Human micro-timing is the biological signature of authentic presence. The body recognises it."
    },
    "stereo_width_narrow": {
        "description": "Narrow stereo field - possibly mono-compatible production or limited spatial processing.",
    },
    "stereo_width_wide": {
        "description": "Wide stereo field - spatial production techniques applied.",
    },
}


def analyse_fingerprints(record: AudioRecord,
                         groove_deviation_ms: float = None,
                         crest_factor_db: float = None,
                         dynamic_range_db: float = None) -> FingerprintReport:
    """
    Run all fingerprint analyses and return a composite report.
    
    groove_deviation_ms, crest_factor_db, dynamic_range_db can be passed
    from the feature extractor output for integrated analysis.
    """
    y = record.y_mono
    sr = record.sample_rate

    # If not passed, compute from raw audio
    if crest_factor_db is None:
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        crest_factor_db = float(20 * np.log10(peak / (rms + 1e-10))) if rms > 0 else 10.0
    if dynamic_range_db is None:
        dynamic_range_db = record.dynamic_range_db

    era_matches = _match_era(crest_factor_db, dynamic_range_db, y, sr)
    technique_matches = _match_techniques(crest_factor_db, dynamic_range_db, groove_deviation_ms)
    spectral_matches = _analyse_spectral_fingerprints(y, sr)
    instrument_matches = _estimate_instruments(y, sr)

    authenticity_markers = _extract_authenticity_markers(
        crest_factor_db, dynamic_range_db, groove_deviation_ms, record
    )
    manufacturing_markers = _extract_manufacturing_markers(
        crest_factor_db, dynamic_range_db, groove_deviation_ms
    )

    # Production context summary
    if era_matches:
        top_era = era_matches[0]
        context = f"Production signature most consistent with {top_era.name}. "
        context += top_era.description
    else:
        context = "Era signature unclear or spans multiple periods."

    return FingerprintReport(
        likely_instruments=instrument_matches,
        likely_era=era_matches,
        likely_techniques=technique_matches + spectral_matches,
        production_context=context,
        authenticity_markers=authenticity_markers,
        manufacturing_markers=manufacturing_markers,
    )


def _match_era(crest_db: float, dynamic_range: float, y: np.ndarray, sr: int) -> List[FingerprintMatch]:
    """Match audio characteristics against known era profiles."""
    
    # Estimate high frequency content (rolloff at 85%)
    S = np.abs(librosa.stft(y[:sr*5] if len(y) > sr*5 else y))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    high_freq = float(np.mean(rolloff))

    matches = []
    for era_name, profile in ERA_PROFILES.items():
        score = 0.0
        evidence = []
        max_score = 0.0

        # Check crest factor
        cf_min, cf_max = profile["crest_factor_range"]
        max_score += 1.0
        if cf_min <= crest_db <= cf_max:
            score += 1.0
            evidence.append(f"Crest factor {crest_db:.1f}dB fits {era_name} range")
        elif abs(crest_db - cf_min) < 3 or abs(crest_db - cf_max) < 3:
            score += 0.5
            evidence.append(f"Crest factor {crest_db:.1f}dB near {era_name} range")

        # Check dynamic range
        dr_min, dr_max = profile["dynamic_range_range"]
        max_score += 1.0
        if dr_min <= dynamic_range <= dr_max:
            score += 1.0
            evidence.append(f"Dynamic range {dynamic_range:.1f}dB fits {era_name}")

        # Check high frequency rolloff
        max_score += 0.5
        hf_limit = profile["high_freq_rolloff"]
        if high_freq < hf_limit * 0.9:
            score += 0.5
            evidence.append(f"High frequency rolloff {high_freq:.0f}Hz suggests pre-{hf_limit}Hz limit")

        confidence = score / max_score if max_score > 0 else 0.0
        if confidence > 0.3:
            matches.append(FingerprintMatch(
                category='era',
                name=era_name,
                confidence=confidence,
                evidence=evidence,
                description=profile["description"]
            ))

    matches.sort(key=lambda x: x.confidence, reverse=True)
    return matches[:3]


def _match_techniques(crest_db: float, dynamic_range: float,
                       groove_ms: Optional[float]) -> List[FingerprintMatch]:
    matches = []

    # Heavy compression
    if crest_db < 6 or dynamic_range < 6:
        sig = TECHNIQUE_SIGNATURES["heavy_compression"]
        matches.append(FingerprintMatch(
            category='technique',
            name='heavy_compression',
            confidence=min(1.0, (6 - min(crest_db, 6)) / 6 + 0.3),
            evidence=[f"Crest factor: {crest_db:.1f}dB", f"Dynamic range: {dynamic_range:.1f}dB"],
            description=sig["description"] + " | " + sig["psychosomatic_note"]
        ))

    # Natural dynamics
    if crest_db > 12 and dynamic_range > 12:
        sig = TECHNIQUE_SIGNATURES["natural_dynamics"]
        matches.append(FingerprintMatch(
            category='technique',
            name='natural_dynamics',
            confidence=min(1.0, crest_db / 25.0),
            evidence=[f"Crest factor: {crest_db:.1f}dB", f"Dynamic range: {dynamic_range:.1f}dB"],
            description=sig["description"] + " | " + sig["psychosomatic_note"]
        ))

    # Rhythm quantisation
    if groove_ms is not None:
        if groove_ms < 5:
            sig = TECHNIQUE_SIGNATURES["quantised_rhythm"]
            matches.append(FingerprintMatch(
                category='technique',
                name='quantised_rhythm',
                confidence=max(0.0, 1.0 - groove_ms / 5.0),
                evidence=[f"Groove deviation: {groove_ms:.1f}ms"],
                description=sig["description"] + " | " + sig["psychosomatic_note"]
            ))
        elif groove_ms > 15:
            sig = TECHNIQUE_SIGNATURES["human_performance"]
            matches.append(FingerprintMatch(
                category='technique',
                name='human_performance',
                confidence=min(1.0, groove_ms / 40.0),
                evidence=[f"Groove deviation: {groove_ms:.1f}ms"],
                description=sig["description"] + " | " + sig["psychosomatic_note"]
            ))

    return matches


def _analyse_spectral_fingerprints(y: np.ndarray, sr: int) -> List[FingerprintMatch]:
    """Detect reverb, stereo width, and other spatial characteristics."""
    matches = []

    # Simple RT60 estimation via energy decay
    # Compute how quickly energy decays after transients
    rms_frames = librosa.feature.rms(y=y, hop_length=512)[0]
    if len(rms_frames) > 10:
        # Look for decay rate after peaks
        peaks = np.where(rms_frames > np.percentile(rms_frames, 75))[0]
        if len(peaks) > 0:
            # Rough reverb detection: sustained energy after peaks
            sustained = np.mean(rms_frames) / (np.max(rms_frames) + 1e-10)
            if sustained > 0.4:
                matches.append(FingerprintMatch(
                    category='technique',
                    name='significant_reverb',
                    confidence=float(sustained),
                    evidence=[f"Energy sustain ratio: {sustained:.2f}"],
                    description="Significant reverb or room ambience detected. Space is part of the sonic language here."
                ))

    return matches


def _estimate_instruments(y: np.ndarray, sr: int) -> List[FingerprintMatch]:
    """
    Rough instrument family estimation from spectral characteristics.
    Not claiming exact identification - indicating presence of families.
    Full instrument fingerprinting requires a trained classifier.
    This is the rule-based starter layer.
    """
    matches = []
    
    # Separate harmonic and percussive
    H, P = librosa.decompose.hpss(librosa.stft(y[:sr*30] if len(y) > sr*30 else y))
    harmonic_energy = np.mean(H ** 2)
    percussive_energy = np.mean(P ** 2)

    # Sub-bass presence (below 80Hz) - bass instruments / 808s / kick
    S = np.abs(librosa.stft(y[:sr*30] if len(y) > sr*30 else y))
    freqs = librosa.fft_frequencies(sr=sr)
    sub_bass_mask = freqs < 80
    mid_mask = (freqs > 200) & (freqs < 2000)
    hi_mask = freqs > 6000

    sub_bass_energy = np.mean(S[sub_bass_mask, :] ** 2) if sub_bass_mask.any() else 0
    mid_energy = np.mean(S[mid_mask, :] ** 2) if mid_mask.any() else 0
    hi_energy = np.mean(S[hi_mask, :] ** 2) if hi_mask.any() else 0
    total = sub_bass_energy + mid_energy + hi_energy + 1e-10

    if sub_bass_energy / total > 0.15:
        matches.append(FingerprintMatch(
            category='instrument',
            name='bass_presence',
            confidence=min(1.0, float(sub_bass_energy / total * 5)),
            evidence=[f"Sub-bass energy ratio: {sub_bass_energy/total:.2f}"],
            description="Significant sub-bass content. Bass instrument, 808, or kick with sub content."
        ))

    if harmonic_energy / (harmonic_energy + percussive_energy + 1e-10) > 0.6:
        matches.append(FingerprintMatch(
            category='instrument',
            name='harmonic_dominant',
            confidence=float(harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)),
            evidence=["Harmonic/percussive ratio > 0.6"],
            description="Harmonic content dominates. Melodic/harmonic instruments (synth, guitar, keys, strings, vocals) are primary."
        ))
    elif percussive_energy / (harmonic_energy + percussive_energy + 1e-10) > 0.5:
        matches.append(FingerprintMatch(
            category='instrument',
            name='percussive_dominant',
            confidence=float(percussive_energy / (harmonic_energy + percussive_energy + 1e-10)),
            evidence=["Harmonic/percussive ratio < 0.5"],
            description="Percussive content dominates. Drum-heavy or rhythmically focused arrangement."
        ))

    if hi_energy / total > 0.1:
        matches.append(FingerprintMatch(
            category='instrument',
            name='high_frequency_content',
            confidence=min(1.0, float(hi_energy / total * 8)),
            evidence=[f"High frequency energy ratio: {hi_energy/total:.2f}"],
            description="Significant high frequency content. Cymbals, bright synths, acoustic instruments, or air frequencies present."
        ))

    return matches


def _extract_authenticity_markers(crest_db, dynamic_range, groove_ms, record) -> List[str]:
    markers = []
    if crest_db > 10:
        markers.append(f"Preserved dynamic range (crest factor {crest_db:.1f}dB) - resists loudness war compression")
    if groove_ms and groove_ms > 12:
        markers.append(f"Human timing variation present ({groove_ms:.1f}ms groove deviation) - live performance markers")
    if record.dynamic_range_db > 12:
        markers.append("Wide dynamic range - suggests production priorities listener experience over loudness")
    if not record.is_clipped:
        markers.append("No clipping detected - headroom preserved")
    return markers


def _extract_manufacturing_markers(crest_db, dynamic_range, groove_ms) -> List[str]:
    markers = []
    if crest_db < 6:
        markers.append(f"Heavy limiting applied (crest factor {crest_db:.1f}dB) - loudness war signature")
    if dynamic_range < 6:
        markers.append(f"Severely compressed dynamic range ({dynamic_range:.1f}dB) - commercial mastering pressure")
    if groove_ms and groove_ms < 3:
        markers.append(f"Extreme rhythmic quantisation ({groove_ms:.1f}ms) - human timing removed")
    return markers

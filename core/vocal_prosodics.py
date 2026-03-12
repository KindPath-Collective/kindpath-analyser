"""
KindPath Analyser :: Vocal Prosodic Analyser

The voice is where the body writes its autobiography.
Strip the language, read the signal: laryngeal tension, held breath,
micro-intonation imprecision, vibrato instability. These are confessions
the conscious mind does not consent to.

This module reads the vocal stem spectrally — not for semantic content
but for what the pre-linguistic signal reveals about the state
of the person who made it. Heavy pitch correction obscures this.
Performance artifacts (breath noise, lip smacks) preserve it.

This is the instrument for reading what was not edited out.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VocalProsodicProfile:
    """
    What the voice reveals when you stop listening to the words.
    All fields are computed from the vocal stem signal alone — no lyrics,
    no semantic analysis. Only what the body left in the frequency record.
    """

    # ── Pitch characteristics ──────────────────────────────────────────────
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0           # High = expressive range or instability
    pitch_range_hz: float = 0.0         # Low = constrained, high = free
    pitch_trend: float = 0.0            # +ve = rising, -ve = falling across piece
    vibrato_rate_hz: float = 0.0        # Natural vibrato ~5-7Hz; outside = tension or artifice
    vibrato_depth_cents: float = 0.0    # Depth of pitch modulation
    vibrato_consistency: float = 0.0    # 0-1: consistent = trained/relaxed, erratic = tension
    tuning_deviation_cents: float = 0.0  # Deviation from equal temperament

    # ── Breath and tension markers ─────────────────────────────────────────
    breath_density: float = 0.0         # Audible breaths per minute
    phrase_length_mean_seconds: float = 0.0  # Time before needing breath
    phrase_length_std: float = 0.0      # Variability in phrase length
    laryngeal_tension_index: float = 0.0  # Derived from spectral tilt and HNR

    # ── Honesty markers (what wasn't corrected) ────────────────────────────
    pitch_correction_likelihood: float = 0.0  # 0-1: 1.0 = very likely auto-tuned
    performance_artifacts_present: bool = False  # Breath noise, room sound detected
    micro_intonation_variance: float = 0.0  # Natural pitch imperfection vs machine precision

    # ── Emotional signal ───────────────────────────────────────────────────
    harmonic_noise_ratio: float = 0.0   # Clear tone vs noise. Low = breathy/emotional
    spectral_tilt: float = 0.0          # Negative = bright/tense, positive = dark/relaxed
    formant_f1_mean: float = 0.0        # Vowel openness
    formant_f2_mean: float = 0.0        # Vowel frontness

    # ── Segment-level arc ─────────────────────────────────────────────────
    tension_arc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    pitch_range_arc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    hnr_arc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # ── Summary ───────────────────────────────────────────────────────────
    authenticity_reading: str = ""
    notable_markers: List[str] = field(default_factory=list)
    late_vocal_divergence: float = 0.0  # How different Q4 vocal behaviour is from Q1-Q3


def analyse_vocal(
    vocal_stem: np.ndarray,
    sr: int,
    quarter_boundaries: Optional[List[Tuple[int, int]]] = None,
) -> VocalProsodicProfile:
    """
    Full prosodic analysis of the vocal stem.

    vocal_stem: 1D float32 numpy array of the isolated vocal signal.
    sr: sample rate.
    quarter_boundaries: list of (start_sample, end_sample) per quarter.
                        If None, the stem is divided into 4 equal quarters.

    Returns a VocalProsodicProfile. Never raises — degrades gracefully
    if the signal is silence, noise, or very short.
    """
    profile = VocalProsodicProfile()

    # Guard: handle empty or silence
    if vocal_stem is None or len(vocal_stem) < sr:
        profile.authenticity_reading = "Signal too short for meaningful analysis."
        return profile

    # Normalise to float32
    audio = np.asarray(vocal_stem, dtype=np.float32)

    # Guard: near-silence
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        profile.authenticity_reading = "Near-silence. No vocal signal detected."
        return profile

    # Quarter boundaries (used for arc computations)
    bounds = quarter_boundaries if quarter_boundaries else _default_quarters(len(audio))

    # ── Pitch extraction ──────────────────────────────────────────────────
    try:
        f0, f0_voiced = _extract_f0(audio, sr)
        profile = _fill_pitch_features(profile, f0, f0_voiced, sr, bounds, audio)
    except Exception as e:
        logger.warning(f"[VOCAL] Pitch extraction failed: {e}")

    # ── Vibrato ───────────────────────────────────────────────────────────
    try:
        if len(f0_voiced) > 20:
            rate, depth, consistency = _compute_vibrato(f0_voiced, sr)
            profile.vibrato_rate_hz = rate
            profile.vibrato_depth_cents = depth
            profile.vibrato_consistency = consistency
    except Exception as e:
        logger.warning(f"[VOCAL] Vibrato computation failed: {e}")

    # ── HNR and spectral tilt ──────────────────────────────────────────────
    try:
        profile.harmonic_noise_ratio = _compute_hnr(audio, sr)
        profile.spectral_tilt = _compute_spectral_tilt(audio, sr)
        profile.laryngeal_tension_index = _compute_laryngeal_tension(
            profile.spectral_tilt, profile.harmonic_noise_ratio
        )
    except Exception as e:
        logger.warning(f"[VOCAL] HNR/spectral computation failed: {e}")

    # ── Formants (F1, F2) ─────────────────────────────────────────────────
    try:
        f1, f2 = _estimate_formants(audio, sr)
        profile.formant_f1_mean = f1
        profile.formant_f2_mean = f2
    except Exception as e:
        logger.warning(f"[VOCAL] Formant estimation failed: {e}")

    # ── Breath detection ──────────────────────────────────────────────────
    try:
        density, mean_phrase, phrase_std = _detect_breath_patterns(audio, sr)
        profile.breath_density = density
        profile.phrase_length_mean_seconds = mean_phrase
        profile.phrase_length_std = phrase_std
        profile.performance_artifacts_present = density > 2.0  # > 2 breaths/min detected
    except Exception as e:
        logger.warning(f"[VOCAL] Breath detection failed: {e}")

    # ── Pitch correction likelihood ───────────────────────────────────────
    profile.pitch_correction_likelihood = _compute_pitch_correction_likelihood(
        profile.micro_intonation_variance
    )

    # ── Segment arcs ──────────────────────────────────────────────────────
    try:
        tension_arc, pitch_arc, hnr_arc = _compute_quarter_arcs(audio, sr, bounds)
        profile.tension_arc = tension_arc
        profile.pitch_range_arc = pitch_arc
        profile.hnr_arc = hnr_arc
    except Exception as e:
        logger.warning(f"[VOCAL] Quarter arc computation failed: {e}")

    # ── Late vocal divergence ─────────────────────────────────────────────
    profile.late_vocal_divergence = _compute_late_divergence(
        profile.tension_arc, profile.pitch_range_arc, profile.hnr_arc
    )

    # ── Authenticity reading ──────────────────────────────────────────────
    profile.authenticity_reading, profile.notable_markers = _build_authenticity_reading(profile)

    return profile


# ── Pitch extraction ──────────────────────────────────────────────────────────

def _extract_f0(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fundamental frequency using librosa.pyin.
    Returns (f0_full, f0_voiced) where f0_voiced contains only high-confidence frames.
    pYIN gives per-frame voiced probability — we keep frames > 0.7 confidence.
    """
    import librosa
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz — below human voice floor
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — above soprano ceiling
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    # Replace NaN with 0 for easier handling
    f0 = np.where(np.isnan(f0), 0.0, f0)
    # Voiced frames only (probability > 0.7)
    f0_voiced = f0[voiced_prob > 0.7]
    f0_voiced = f0_voiced[f0_voiced > 0]
    return f0, f0_voiced


def _fill_pitch_features(
    profile: VocalProsodicProfile,
    f0: np.ndarray,
    f0_voiced: np.ndarray,
    sr: int,
    bounds: List[Tuple[int, int]],
    audio: np.ndarray,
) -> VocalProsodicProfile:
    """Compute pitch-level statistics and fill the profile."""
    import librosa

    if len(f0_voiced) == 0:
        profile.authenticity_reading = "No pitched content detected."
        return profile

    profile.pitch_mean_hz = float(np.mean(f0_voiced))
    profile.pitch_std_hz = float(np.std(f0_voiced))
    profile.pitch_range_hz = float(np.max(f0_voiced) - np.min(f0_voiced))

    # Micro-intonation variance in cents
    # Convert Hz to MIDI (log scale); variance in semitones × 100
    midi_notes = librosa.hz_to_midi(f0_voiced)
    micro_var_semitones = float(np.std(midi_notes))
    profile.micro_intonation_variance = micro_var_semitones * 100.0  # in cents

    # Pitch trend: linear regression slope on voiced F0 over time
    if len(f0_voiced) > 10:
        x = np.arange(len(f0_voiced), dtype=np.float32)
        slope = float(np.polyfit(x, f0_voiced, 1)[0])
        profile.pitch_trend = slope  # Hz per frame

    # Tuning deviation from A440
    # Find the most common MIDI note, compute deviation
    bins = np.round(midi_notes).astype(int)
    deviations = (midi_notes - bins) * 100.0  # cents
    profile.tuning_deviation_cents = float(np.mean(np.abs(deviations)))

    return profile


# ── Vibrato ───────────────────────────────────────────────────────────────────

def _compute_vibrato(f0_voiced: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """
    Detect vibrato characteristics from voiced F0 contour.

    Natural vibrato rate: 5-7Hz
    Vibrato depth: typically 20-100 cents (0.2–1 semitone)

    Algorithm:
    1. Convert F0 to cent deviations from smooth trend
    2. Apply FFT to the cent contour
    3. Find dominant frequency in 4-8 Hz range
    4. Measure amplitude at that frequency
    5. Consistency = stability of the rate over time
    """
    hop_length_seconds = 512 / sr  # Default pyin hop

    # Convert to cents relative to mean
    cents = 1200.0 * np.log2(f0_voiced / (np.mean(f0_voiced) + 1e-9))

    # FFT of the cent contour
    n = len(cents)
    fft_result = np.abs(np.fft.rfft(cents, n=n))
    freqs = np.fft.rfftfreq(n, d=hop_length_seconds)

    # Find dominant frequency in vibrato range (4-8 Hz)
    mask = (freqs >= 4.0) & (freqs <= 8.0)
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    vibrato_fft = fft_result * mask
    peak_idx = int(np.argmax(vibrato_fft))
    if vibrato_fft[peak_idx] < 1.0:
        return 0.0, 0.0, 0.0

    vibrato_rate = float(freqs[peak_idx])
    vibrato_depth = float(vibrato_fft[peak_idx] * 2.0 / n)  # Peak amplitude in cents

    # Consistency: compute rate via short-time analysis
    frame_size = min(64, len(cents) // 4)
    if frame_size < 8:
        return vibrato_rate, vibrato_depth, 0.5

    rates = []
    for i in range(0, len(cents) - frame_size, frame_size // 2):
        chunk = cents[i:i + frame_size]
        chunk_fft = np.abs(np.fft.rfft(chunk))
        chunk_freqs = np.fft.rfftfreq(frame_size, d=hop_length_seconds)
        chunk_mask = (chunk_freqs >= 4.0) & (chunk_freqs <= 8.0)
        if np.any(chunk_mask):
            local_idx = int(np.argmax(chunk_fft * chunk_mask))
            rates.append(float(chunk_freqs[local_idx]))

    if len(rates) < 2:
        return vibrato_rate, vibrato_depth, 0.5

    rate_std = float(np.std(rates))
    rate_mean = float(np.mean(rates))
    consistency = float(1.0 - min(rate_std / (rate_mean + 1e-9), 1.0))
    return vibrato_rate, vibrato_depth, consistency


# ── HNR and spectral features ─────────────────────────────────────────────────

def _compute_hnr(audio: np.ndarray, sr: int) -> float:
    """
    Harmonic-to-Noise Ratio.
    Separates harmonic and percussive components; ratio of their energies.
    Low HNR = breathy, emotional, raw. High HNR = clear, controlled, possibly processed.
    """
    import librosa
    harmonic = librosa.effects.harmonic(audio)
    residual = audio - harmonic
    energy_harmonic = float(np.mean(harmonic ** 2))
    energy_residual = float(np.mean(residual ** 2))
    if energy_residual < 1e-10:
        return 40.0  # Perfectly harmonic
    return float(10.0 * np.log10(energy_harmonic / (energy_residual + 1e-10)))


def _compute_spectral_tilt(audio: np.ndarray, sr: int) -> float:
    """
    Spectral tilt: the slope of the spectral envelope from low to high frequencies.
    Negative = bright/tense (energy concentrated in high frequencies).
    Positive = dark/relaxed (low frequency dominance).
    Computed as linear regression slope on the log-magnitude spectrum.
    """
    import librosa
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    mean_spectrum = np.mean(S, axis=1)
    log_spectrum = np.log1p(mean_spectrum)
    freqs = np.arange(len(log_spectrum), dtype=np.float32)
    if len(freqs) < 2:
        return 0.0
    slope = float(np.polyfit(freqs, log_spectrum, 1)[0])
    return slope


def _compute_laryngeal_tension(spectral_tilt: float, hnr: float) -> float:
    """
    Laryngeal tension index: composite of spectral emphasis in 3kHz+ region
    and HNR. High-frequency energy emphasis + lower HNR = tension.
    Returns 0-1 where 1 = maximum tension.
    """
    # Spectral tilt: more negative = brighter = more tension
    tilt_component = float(max(0.0, min(1.0, (-spectral_tilt + 0.01) / 0.02)))
    # HNR: higher = clearer = less tension (tension adds breathiness)
    hnr_component = float(max(0.0, min(1.0, 1.0 - (hnr / 40.0))))
    return (tilt_component + hnr_component) / 2.0


# ── Formants ──────────────────────────────────────────────────────────────────

def _estimate_formants(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Estimate F1 and F2 formant frequencies via LPC analysis.
    F1 (300-800 Hz): vowel openness — open mouth = high F1.
    F2 (800-2500 Hz): vowel frontness — front vowels = high F2.
    Uses scipy's signal processing for LPC coefficient extraction.
    """
    from scipy.signal import lfilter
    from scipy.signal import find_peaks

    # Down-sample to 11025 Hz for LPC stability
    target_sr = 11025
    if sr > target_sr:
        step = sr // target_sr
        audio_ds = audio[::step]
    else:
        audio_ds = audio

    # Pre-emphasis to enhance formant peaks
    pre_emphasis = np.diff(audio_ds, prepend=audio_ds[0])
    frame_size = min(1024, len(pre_emphasis) // 4)
    if frame_size < 32:
        return 500.0, 1500.0  # Sensible defaults

    # Extract middle 1s of audio for a representative frame
    mid = len(pre_emphasis) // 2
    frame = pre_emphasis[mid:mid + frame_size]

    # LPC order for speech: 2 + sr/1000
    order = int(2 + target_sr / 1000)
    a = _lpc(frame, order)

    # LPC roots to formant frequencies
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs_hz = sorted(angles * target_sr / (2 * np.pi))
    formant_freqs = [f for f in freqs_hz if 200 < f < 3500]

    f1 = float(formant_freqs[0]) if len(formant_freqs) > 0 else 500.0
    f2 = float(formant_freqs[1]) if len(formant_freqs) > 1 else 1500.0
    return f1, f2


def _lpc(signal: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients via the autocorrelation method (Levinson-Durbin)."""
    # Autocorrelation
    r = np.correlate(signal, signal, mode='full')
    r = r[len(r) // 2:]

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0]) + 1e-9

    for i in range(1, order + 1):
        lam = -sum(a[j] * r[i - j] for j in range(i)) / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        e = e * (1.0 - lam ** 2)
        a = a_new

    return a


# ── Breath detection ──────────────────────────────────────────────────────────

def _detect_breath_patterns(
    audio: np.ndarray, sr: int
) -> Tuple[float, float, float]:
    """
    Detect breath-like events between phrases.

    Breath signatures:
    - Short energy burst below 300Hz
    - Low spectral centroid (below 500Hz)
    - Low overall amplitude (0.01-0.15 × peak)

    Returns (breaths_per_minute, mean_phrase_length_secs, phrase_length_std).
    """
    import librosa

    hop = 512
    frame_len = 2048
    duration = len(audio) / sr

    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Spectral centroid per frame
    S = np.abs(librosa.stft(audio, n_fft=frame_len, hop_length=hop))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    # Breath-like frames: low energy, low centroid, not silence
    rms_norm = rms / (np.max(rms) + 1e-9)
    breath_frames = (rms_norm > 0.01) & (rms_norm < 0.15) & (centroid < 500)

    # Cluster adjacent breath frames into events
    breath_count = 0
    phrase_boundaries = []
    in_breath = False
    breath_start = 0

    for i, is_breath in enumerate(breath_frames):
        if is_breath and not in_breath:
            in_breath = True
            breath_start = i
        elif not is_breath and in_breath:
            in_breath = False
            breath_count += 1
            phrase_boundaries.append(rms_times[breath_start])

    # Phrase lengths: time between consecutive phrase boundaries
    if len(phrase_boundaries) > 1:
        phrase_lengths = np.diff(phrase_boundaries)
        mean_phrase = float(np.mean(phrase_lengths))
        phrase_std = float(np.std(phrase_lengths))
    else:
        mean_phrase = duration
        phrase_std = 0.0

    breaths_per_minute = breath_count / (duration / 60.0) if duration > 0 else 0.0
    return breaths_per_minute, mean_phrase, phrase_std


# ── Pitch correction likelihood ───────────────────────────────────────────────

def _compute_pitch_correction_likelihood(micro_intonation_variance_cents: float) -> float:
    """
    Natural singing has micro-intonation variance of ~15-40 cents.
    Auto-tuned vocals show variance < 5 cents with occasional hard jumps.
    Returns 0-1: 1.0 = very likely pitch-corrected.
    """
    if micro_intonation_variance_cents <= 0:
        return 0.5  # Insufficient data
    # Lower variance = higher correction likelihood
    natural_baseline = 30.0  # cents
    score = 1.0 - min(micro_intonation_variance_cents / natural_baseline, 1.0)
    return float(max(0.0, min(1.0, score)))


# ── Quarter arc computation ───────────────────────────────────────────────────

def _compute_quarter_arcs(
    audio: np.ndarray, sr: int, bounds: List[Tuple[int, int]]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute per-quarter values for tension, pitch range, and HNR.
    Returns three 4-element lists.
    """
    tension_arc = []
    pitch_range_arc = []
    hnr_arc = []

    for start, end in bounds:
        chunk = audio[start:end]
        if len(chunk) < sr // 4:
            tension_arc.append(0.0)
            pitch_range_arc.append(0.0)
            hnr_arc.append(0.0)
            continue

        # HNR for this quarter
        try:
            hnr = _compute_hnr(chunk, sr)
            tilt = _compute_spectral_tilt(chunk, sr)
            tension = _compute_laryngeal_tension(tilt, hnr)
        except Exception:
            hnr, tension = 0.0, 0.0

        # Pitch range for this quarter
        try:
            _, f0_voiced = _extract_f0(chunk, sr)
            p_range = float(np.max(f0_voiced) - np.min(f0_voiced)) if len(f0_voiced) > 5 else 0.0
        except Exception:
            p_range = 0.0

        tension_arc.append(tension)
        pitch_range_arc.append(p_range)
        hnr_arc.append(hnr)

    return tension_arc, pitch_range_arc, hnr_arc


def _compute_late_divergence(
    tension_arc: List[float],
    pitch_arc: List[float],
    hnr_arc: List[float],
) -> float:
    """
    Measure how different Q4 behaviour is from Q1-Q3 average.
    Mirrors the LSII logic but applied to vocal-specific dimensions.
    Returns 0-1.
    """
    if len(tension_arc) < 4:
        return 0.0

    def _norm_delta(q4_val: float, baseline: float) -> float:
        return float(np.tanh(abs(q4_val - baseline) / (abs(baseline) + 1e-6)))

    baseline_tension = float(np.mean(tension_arc[:3]))
    baseline_pitch = float(np.mean(pitch_arc[:3]))
    baseline_hnr = float(np.mean(hnr_arc[:3]))

    delta_t = _norm_delta(tension_arc[3], baseline_tension)
    delta_p = _norm_delta(pitch_arc[3], baseline_pitch)
    delta_h = _norm_delta(hnr_arc[3], baseline_hnr)

    return float((delta_t + delta_p + delta_h) / 3.0)


# ── Authenticity reading ──────────────────────────────────────────────────────

def _build_authenticity_reading(profile: VocalProsodicProfile) -> Tuple[str, List[str]]:
    """
    Build the human-readable authenticity reading and a list of notable markers.
    The reading does not judge — it reads what is there.
    """
    markers = []

    # Pitch correction
    if profile.pitch_correction_likelihood > 0.8 and not profile.performance_artifacts_present:
        markers.append("heavy_pitch_correction")
    elif profile.pitch_correction_likelihood < 0.3:
        markers.append("natural_intonation")

    # Vibrato
    if 4.5 <= profile.vibrato_rate_hz <= 7.5:
        markers.append("natural_vibrato_range")
    elif profile.vibrato_rate_hz > 0 and (profile.vibrato_rate_hz < 4.0 or profile.vibrato_rate_hz > 8.0):
        markers.append("atypical_vibrato_rate")

    if profile.vibrato_consistency < 0.3 and profile.vibrato_rate_hz > 0:
        markers.append("vibrato_instability")

    # Tension
    if profile.laryngeal_tension_index > 0.7:
        markers.append("high_laryngeal_tension")
    elif profile.laryngeal_tension_index < 0.3:
        markers.append("relaxed_larynx")

    # Breath / performance presence
    if profile.performance_artifacts_present:
        markers.append("breath_artifacts_present")

    # HNR
    if profile.harmonic_noise_ratio < 5.0:
        markers.append("breathy_or_emotional_quality")
    elif profile.harmonic_noise_ratio > 25.0:
        markers.append("clear_controlled_tone")

    # Late divergence
    if profile.late_vocal_divergence > 0.5:
        markers.append("late_vocal_shift")

    # Build reading
    pcl = profile.pitch_correction_likelihood
    art = profile.performance_artifacts_present
    mic = profile.micro_intonation_variance
    lt = profile.laryngeal_tension_index

    if pcl > 0.8 and not art:
        reading = (
            "Heavy pitch processing detected. The authentic vocal signal "
            "has been largely replaced by corrected pitch. "
            "What remains is technically accurate and emotionally filtered."
        )
    elif lt > 0.7 and profile.vibrato_consistency < 0.3:
        reading = (
            "Vocal tension markers are present. The vibrato is inconsistent "
            "and the spectral emphasis suggests laryngeal constriction — "
            "performance under stress or constraint."
        )
    elif mic > 25.0 and art:
        reading = (
            "Strong authentic performance markers. Natural pitch imperfection "
            "is preserved, and breath artifacts are present — the recording "
            "kept the body in the signal."
        )
    elif profile.harmonic_noise_ratio < 5.0:
        reading = (
            "Breathy, exposed vocal quality. Low harmonic-to-noise ratio "
            "suggests emotional openness or deliberate stylistic rawness. "
            "The body is close to the surface of this performance."
        )
    else:
        reading = (
            "Moderate authenticity markers. Some natural intonation variation "
            "present. Tension levels within expected range for trained voice."
        )

    if profile.late_vocal_divergence > 0.5:
        reading += (
            " The final section shows a measurable shift in vocal character — "
            "tension, range, or clarity changes significantly from the earlier sections."
        )

    return reading, markers


# ── Utilities ─────────────────────────────────────────────────────────────────

def _default_quarters(n_samples: int) -> List[Tuple[int, int]]:
    """Divide audio into 4 equal quarters."""
    q = n_samples // 4
    return [
        (0, q),
        (q, 2 * q),
        (2 * q, 3 * q),
        (3 * q, n_samples),
    ]

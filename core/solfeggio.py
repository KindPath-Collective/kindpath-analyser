"""
KindPath Analyser :: Solfeggio Alignment

The oldest layer of the frequency field science.

Guido d'Arezzo's hexachord (~1025 AD) encoded the original Solfeggio scale
in the Hymn for St John the Baptist (Ut queant laxis):

  Ut  = 396 Hz  — liberation from fear and guilt (the ground tone)
  Re  = 417 Hz  — undoing situations, facilitation of change
  Mi  = 528 Hz  — transformation; associated with DNA repair resonance
  Fa  = 639 Hz  — interpersonal resonance and connection
  Sol = 741 Hz  — awakening intuition, expression, solutions
  La  = 852 Hz  — returning to spiritual order; pure tone alignment

These frequencies were not arbitrary. They map to specific integer ratios
within the natural harmonic series, and they align with the biological vocal
fundamental (~130 Hz, C3) — the frequency at which the adult human speaking
voice rests when the body is not under pressure.

In 1939, the Nazi Propaganda Ministry pushed for A440 standardisation.
By 1955 the ISO had enshrined it as global law. Equal temperament at A440
does not align cleanly with the Solfeggio series — most Solfeggio frequencies
sit 15–45 cents sharp of their nearest A440 equal-temperament equivalent.

The effect: music tuned to A440 sits consistently flat relative to the
biological resonance frequencies. The body registers this as ambient wrongness —
not consciously, but somatically. This is not metaphor. It is measurable.

This module reads three things simultaneously:

1. VOCAL FUNDAMENTAL DEVIATION — how far is this piece's tonal root from
   the biological vocal reference (~130 Hz, C3)?

2. SOLFEGGIO GRID PROXIMITY — how closely do the piece's prominent frequencies
   align to the pre-institutional harmonic grid?

3. INSTITUTIONAL DISTANCE — the composite gap between biological alignment
   and the A440 institutional standard.

High institutional_distance = piece anchored to institutional standard,
operating outside biological resonance range.
Low institutional_distance = piece closer to biological reference field.

This metric does not judge the music. It locates it.

See also: kindpath-canon/FREQUENCY_FIELD_ARCHITECTURE.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Reference constants ───────────────────────────────────────────────────────

BIOLOGICAL_VOCAL_MEAN_HZ = 130.81
"""
C3 at A440 — mean adult vocal fundamental at rest.
The frequency at which the body is not efforting.
See kindpath-canon/FREQUENCY_FIELD_ARCHITECTURE.md
"""

A440_HZ = 440.0
"""Institutional reference pitch (ISO 16, 1955). The standardised anchor."""

A432_HZ = 432.0
"""
Pre-standardisation common reference pitch, closer to biological harmonic series.
The Solfeggio frequencies align more naturally at this reference.
"""

# Cents offset from A440 to A432 — the measurable distance to pre-institutional tuning
A432_CENTS_FROM_A440: float = 1200.0 * np.log2(A432_HZ / A440_HZ)
"""Approximately −31.77 cents. Negative = below A440 = toward biological reference."""

# Reference: C4 at A440
C4_HZ = 261.6255653  # 440 * 2^(-9/12)

# The original Solfeggio table — (frequency_hz, syllable, functional_description)
SOLFEGGIO_TABLE = [
    (396.0, "Ut",  "Liberation from fear and guilt — the ground tone"),
    (417.0, "Re",  "Undoing situations, facilitation of change"),
    (528.0, "Mi",  "Transformation; associated with DNA repair resonance"),
    (639.0, "Fa",  "Interpersonal resonance and connection"),
    (741.0, "Sol", "Awakening intuition — expression and solutions"),
    (852.0, "La",  "Returning to spiritual order; pure tone alignment"),
]

SOLFEGGIO_HZ: list = [row[0] for row in SOLFEGGIO_TABLE]
SOLFEGGIO_NAMES: list = [f"{row[1]} ({int(row[0])} Hz)" for row in SOLFEGGIO_TABLE]

# Pitch class names (0=C, 1=C#, ..., 11=B)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Flat → sharp normalisation for key parsing
FLAT_TO_SHARP = {
    'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
    'B♭': 'A#', 'D♭': 'C#', 'E♭': 'D#', 'G♭': 'F#', 'A♭': 'G#',
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SolfeggioAlignment:
    """
    Three simultaneous readings of a piece's frequency field alignment.

    These three numbers do not judge the music. They locate it.
    A piece that lives far from the biological reference is not inferior —
    it is operating in a different field, with different somatic effects.
    Naming the field is what makes the choice legible.
    """

    # Distance in Hz between the tonal root and the biological vocal reference (130.81 Hz).
    # 0.0 = root lands exactly on biological vocal mean (C3 at A440).
    # Increases with distance from that anchor.
    vocal_fundamental_deviation_hz: float

    # 0.0–1.0: How closely the piece's chroma energy aligns to the Solfeggio grid.
    # 1.0 = prominent frequencies land on or very near Solfeggio values.
    # 0.0 = no meaningful alignment — standard A440 equal temperament throughout.
    solfeggio_grid_proximity: float

    # The Solfeggio frequency (in Hz) nearest to the piece's tonal root.
    nearest_solfeggio_hz: float

    # Human-readable name: e.g. "Mi (528 Hz)"
    nearest_solfeggio_name: str

    # 0.0–1.0: Composite gap between biological alignment and A440 institutional standard.
    # 0.0 = piece operates in biological resonance field.
    # 1.0 = piece is maximally anchored to institutional A440 standard.
    institutional_distance: float

    # How far the piece's tuning deviates from A440 (cents, from feature_extractor).
    # Negative = flat (toward A432 / pre-institutional).
    # Positive = sharp (further from biological reference).
    tuning_deviation_cents: float

    # Plain-English interpretation of all three readings.
    alignment_reading: str

    # The dominant Solfeggio frequency name if solfeggio_grid_proximity > 0.5.
    # None if alignment is too diffuse to name a dominant anchor.
    dominant_solfeggio: Optional[str] = None


# ── Main computation ──────────────────────────────────────────────────────────

def compute_solfeggio_alignment(
    chroma_mean: list,
    key_estimate: str,
    tuning_offset_cents: float,
) -> SolfeggioAlignment:
    """
    Compute Solfeggio alignment from harmonic feature data extracted per-segment.

    chroma_mean:         12-element list — average energy per pitch class (0=C .. 11=B)
    key_estimate:        string like "C major" or "F# minor"
    tuning_offset_cents: deviation from A440 in cents (librosa.estimate_tuning output)

    Returns a SolfeggioAlignment with three simultaneous frequency field readings.
    """
    chroma = np.array(chroma_mean, dtype=float)

    # ── Tonal root frequency at biological reference octave ───────────────────
    key_idx = _parse_key_index(key_estimate)

    # Root note at C4 octave reference (A440 standard), adjusted for tuning
    root_freq_c4 = C4_HZ * (2.0 ** (key_idx / 12.0))
    freq_adjustment = 2.0 ** (tuning_offset_cents / 1200.0)
    root_freq_adjusted = root_freq_c4 * freq_adjustment

    # Transpose to the octave closest to the biological vocal mean (130.81 Hz C3)
    root_freq_bio_octave = _find_nearest_octave(root_freq_adjusted, BIOLOGICAL_VOCAL_MEAN_HZ)

    vocal_fundamental_deviation_hz = abs(root_freq_bio_octave - BIOLOGICAL_VOCAL_MEAN_HZ)

    # ── Nearest Solfeggio frequency to the tonal root ─────────────────────────
    nearest_sf_hz, nearest_sf_name, _ = _nearest_solfeggio(root_freq_bio_octave)

    # ── Solfeggio grid proximity across full chroma ───────────────────────────
    solfeggio_grid_proximity = _compute_solfeggio_grid_proximity(chroma, tuning_offset_cents)

    # ── Dominant solfeggio anchor ─────────────────────────────────────────────
    dominant_solfeggio = nearest_sf_name if solfeggio_grid_proximity > 0.5 else None

    # ── Institutional distance ────────────────────────────────────────────────
    institutional_distance = _compute_institutional_distance(
        vocal_fundamental_deviation_hz,
        solfeggio_grid_proximity,
        tuning_offset_cents,
    )

    # ── Plain-language reading ────────────────────────────────────────────────
    alignment_reading = _build_alignment_reading(
        vocal_dev=vocal_fundamental_deviation_hz,
        solfeggio_proximity=solfeggio_grid_proximity,
        institutional_distance=institutional_distance,
        tuning_cents=tuning_offset_cents,
        nearest_sf_name=nearest_sf_name,
        root_freq=root_freq_bio_octave,
        key_estimate=key_estimate,
    )

    return SolfeggioAlignment(
        vocal_fundamental_deviation_hz=round(vocal_fundamental_deviation_hz, 2),
        solfeggio_grid_proximity=round(solfeggio_grid_proximity, 4),
        nearest_solfeggio_hz=nearest_sf_hz,
        nearest_solfeggio_name=nearest_sf_name,
        institutional_distance=round(institutional_distance, 4),
        tuning_deviation_cents=round(tuning_offset_cents, 2),
        alignment_reading=alignment_reading,
        dominant_solfeggio=dominant_solfeggio,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_key_index(key_estimate: str) -> int:
    """
    Parse pitch class index (0=C, 1=C#, ..., 11=B) from a key_estimate string.
    Handles "C major", "F# minor", "A# minor", "unknown", etc.
    Returns 0 (C) for any unparseable input — the biological reference tonic.
    """
    if not key_estimate or key_estimate.lower().startswith('unknown'):
        return 0

    # Extract the note name (first token before major/minor)
    note_str = key_estimate.strip().split()[0]

    # Standardise flat spellings to sharp equivalents
    note_str = FLAT_TO_SHARP.get(note_str, note_str)

    try:
        return NOTE_NAMES.index(note_str)
    except ValueError:
        return 0


def _find_nearest_octave(freq: float, target: float) -> float:
    """
    Transpose freq to the octave closest to target.
    Multiplies or divides by 2 until the result is as near to target as possible.
    """
    if freq <= 0 or target <= 0:
        return target
    n_octaves = round(np.log2(target / freq))
    return freq * (2.0 ** n_octaves)


def _nearest_solfeggio(freq: float) -> tuple:
    """
    Find the Solfeggio frequency (across all octaves) nearest to freq.
    Returns (nearest_base_hz, name_string, cents_deviation).

    Searches the original Solfeggio values (396–852 Hz) and their octave
    equivalents — the Solfeggio series extends infinitely in both directions
    via octave doubling/halving.
    """
    best_hz = SOLFEGGIO_HZ[0]
    best_name = SOLFEGGIO_NAMES[0]
    best_cents = float('inf')

    for sf_hz, sf_name in zip(SOLFEGGIO_HZ, SOLFEGGIO_NAMES):
        # Bring this Solfeggio frequency to the octave nearest our target
        sf_at_target_octave = _find_nearest_octave(sf_hz, freq)
        if sf_at_target_octave <= 0:
            continue
        cents = abs(1200.0 * np.log2(freq / sf_at_target_octave))
        if cents < best_cents:
            best_cents = cents
            best_hz = sf_hz
            best_name = sf_name

    return best_hz, best_name, best_cents


def _compute_solfeggio_grid_proximity(chroma: np.ndarray, tuning_offset_cents: float) -> float:
    """
    Measure how closely the piece's chroma energy distribution aligns to the Solfeggio grid.

    For each pitch class (weighted by chroma energy), find the minimum cents distance
    to any Solfeggio frequency at the nearest octave. Average across all pitch classes
    weighted by energy. Convert to 0–1 proximity score.

    A 100-cent threshold (one equal-temperament semitone) is generous enough to catch
    loose alignment while remaining discriminating — genuine alignment is typically
    within 20–30 cents.
    """
    if chroma.max() <= 0:
        return 0.0

    chroma_norm = chroma / chroma.max()
    freq_adjustment = 2.0 ** (tuning_offset_cents / 1200.0)

    total_weight = 0.0
    weighted_cents = 0.0

    for pitch_class in range(12):
        weight = float(chroma_norm[pitch_class])
        if weight < 0.05:
            # Skip very quiet pitch classes — they don't define the frequency field
            continue

        # Reference frequency for this pitch class at C4 octave, tuning-adjusted
        ref_freq = C4_HZ * (2.0 ** (pitch_class / 12.0)) * freq_adjustment

        # Find minimum cents distance to any Solfeggio frequency at any octave
        min_cents = float('inf')
        for sf_hz in SOLFEGGIO_HZ:
            sf_at_ref = _find_nearest_octave(sf_hz, ref_freq)
            if sf_at_ref <= 0:
                continue
            cents = abs(1200.0 * np.log2(ref_freq / sf_at_ref))
            if cents < min_cents:
                min_cents = cents

        weighted_cents += weight * min_cents
        total_weight += weight

    if total_weight <= 0:
        return 0.0

    avg_cents = weighted_cents / total_weight
    # 0 cents deviation → proximity 1.0; 100 cents → proximity 0.0
    proximity = max(0.0, 1.0 - avg_cents / 100.0)
    return float(np.clip(proximity, 0.0, 1.0))


def _compute_institutional_distance(
    vocal_fundamental_deviation_hz: float,
    solfeggio_grid_proximity: float,
    tuning_offset_cents: float,
) -> float:
    """
    Compute the institutional distance — the composite gap between a piece's
    frequency field and the biological/pre-institutional reference.

    Three components:
    1. Solfeggio grid alignment (40%) — structural; most historically significant
    2. Vocal fundamental deviation from biological mean (30%)
    3. Tuning reference proximity: A432 (biological) vs A440 (institutional) (30%)

    0.0 = deeply aligned with biological resonance field
    1.0 = maximally anchored to A440 institutional standard
    """
    # Component 1: inverse Solfeggio proximity (0=aligned, 1=not aligned)
    solfeggio_component = 1.0 - solfeggio_grid_proximity

    # Component 2: bio vocal reference deviation
    # 0 Hz deviation → 0 (perfectly bio); 30 Hz deviation → 1.0 (maximal drift)
    bio_component = float(np.clip(vocal_fundamental_deviation_hz / 30.0, 0.0, 1.0))

    # Component 3: tuning proximity to institutional standard vs pre-institutional
    # A432 (≈ −31.77 cents from A440) = biological reference end.
    # A440 (0 cents) = institutional standard.
    # Above A440 (positive) = further from biological.
    # Map range [A432_CENTS_FROM_A440, +50 cents] → [0.0, 1.0]
    tuning_range = abs(50.0 - A432_CENTS_FROM_A440)  # ≈ 81.77 cents
    tuning_component = float(
        np.clip(
            (tuning_offset_cents - A432_CENTS_FROM_A440) / tuning_range,
            0.0, 1.0
        )
    )

    institutional_distance = (
        0.4 * solfeggio_component
        + 0.3 * bio_component
        + 0.3 * tuning_component
    )
    return float(np.clip(institutional_distance, 0.0, 1.0))


def aggregate_solfeggio(quarter_features: list) -> Optional['SolfeggioAlignment']:
    """
    Compute an aggregate SolfeggioAlignment from a list of SegmentFeatures (one per quarter).

    Averages the three key numeric measurements across all quarters that have solfeggio data.
    Returns None if no quarters have solfeggio alignment computed.

    Intended use: aggregate across the four quarters produced by compute_trajectory,
    then pass the result to build_psychosomatic_profile.
    """
    alignments = []
    for q in quarter_features:
        if q is None:
            continue
        harmonic = getattr(q, 'harmonic', None)
        if harmonic is None:
            continue
        sa = getattr(harmonic, 'solfeggio_alignment', None)
        if sa is not None:
            alignments.append(sa)

    if not alignments:
        return None

    # Average the three core numeric axes across all quarters
    avg_vocal_dev = float(np.mean([a.vocal_fundamental_deviation_hz for a in alignments]))
    avg_proximity = float(np.mean([a.solfeggio_grid_proximity for a in alignments]))
    avg_distance = float(np.mean([a.institutional_distance for a in alignments]))
    avg_tuning = float(np.mean([a.tuning_deviation_cents for a in alignments]))

    # Use the nearest Solfeggio reference from the first quarter (most stable)
    nearest_sf_hz = alignments[0].nearest_solfeggio_hz
    nearest_sf_name = alignments[0].nearest_solfeggio_name
    dominant_solfeggio = nearest_sf_name if avg_proximity > 0.5 else None

    # Re-derive an approximate root frequency for the reading text
    # (vocal_dev is distance from biological mean, so root ≈ biological_mean + dev)
    # This is a valid approximation for the aggregate summary text
    approx_root = BIOLOGICAL_VOCAL_MEAN_HZ + avg_vocal_dev

    alignment_reading = _build_alignment_reading(
        vocal_dev=avg_vocal_dev,
        solfeggio_proximity=avg_proximity,
        institutional_distance=avg_distance,
        tuning_cents=avg_tuning,
        nearest_sf_name=nearest_sf_name,
        root_freq=approx_root,
        key_estimate="(aggregate)",
    )

    return SolfeggioAlignment(
        vocal_fundamental_deviation_hz=round(avg_vocal_dev, 2),
        solfeggio_grid_proximity=round(avg_proximity, 4),
        nearest_solfeggio_hz=nearest_sf_hz,
        nearest_solfeggio_name=nearest_sf_name,
        institutional_distance=round(avg_distance, 4),
        tuning_deviation_cents=round(avg_tuning, 2),
        alignment_reading=alignment_reading,
        dominant_solfeggio=dominant_solfeggio,
    )


def _build_alignment_reading(
    vocal_dev: float,
    solfeggio_proximity: float,
    institutional_distance: float,
    tuning_cents: float,
    nearest_sf_name: str,
    root_freq: float,
    key_estimate: str,
) -> str:
    """
    Generate a plain-English reading of the three alignment measurements.

    Reads like an elder describing the frequency field — not what it means
    morally, but what it is physically and what it does to a body.
    """
    key_name = key_estimate.split()[0] if key_estimate else '?'
    parts = []

    # Tonal root position relative to biological reference
    if vocal_dev <= 2.0:
        parts.append(
            f"Tonal root ({key_name}) at {root_freq:.1f} Hz — "
            f"closely aligned with the biological vocal fundamental (C3, 130.8 Hz). "
            f"The piece is operating at the body's own resonance anchor."
        )
    elif vocal_dev <= 10.0:
        parts.append(
            f"Tonal root ({key_name}) at {root_freq:.1f} Hz — "
            f"{vocal_dev:.1f} Hz from the biological vocal reference (130.8 Hz). "
            f"A small but measurable drift from the biological anchor."
        )
    else:
        parts.append(
            f"Tonal root ({key_name}) at {root_freq:.1f} Hz — "
            f"{vocal_dev:.1f} Hz from biological vocal reference (130.8 Hz, C3). "
            f"The piece's foundational pitch sits outside the biological resonance range."
        )

    # Solfeggio proximity reading
    if solfeggio_proximity > 0.65:
        parts.append(
            f"Solfeggio grid proximity: {solfeggio_proximity:.0%}. "
            f"Prominent frequencies align closely with the pre-institutional harmonic series. "
            f"Nearest anchor: {nearest_sf_name}."
        )
    elif solfeggio_proximity > 0.35:
        parts.append(
            f"Solfeggio grid proximity: {solfeggio_proximity:.0%}. "
            f"Some prominent frequencies approach the pre-institutional harmonic series. "
            f"Nearest reference: {nearest_sf_name}."
        )
    else:
        parts.append(
            f"Solfeggio grid proximity: {solfeggio_proximity:.0%}. "
            f"The piece operates entirely within A440 equal temperament, "
            f"removed from the pre-institutional Solfeggio series."
        )

    # Tuning deviation note (only if significant)
    if abs(tuning_cents) > 15:
        direction = "flat" if tuning_cents < 0 else "sharp"
        if tuning_cents < -20:
            parts.append(
                f"Tuning {abs(tuning_cents):.0f} cents {direction} of A440 — "
                f"approaching pre-standardisation reference pitch (A432, −31.8 cents). "
                f"The instrument is pulling toward the biological field."
            )
        else:
            parts.append(
                f"Tuning {abs(tuning_cents):.0f} cents {direction} of A440 standard."
            )

    # Institutional distance summary
    if institutional_distance > 0.65:
        parts.append(
            f"Institutional distance {institutional_distance:.2f} — "
            f"piece anchored within the A440 institutional standard, "
            f"operating outside biological resonance range. "
            f"The body registers this as a field condition, not as a flaw."
        )
    elif institutional_distance > 0.35:
        parts.append(
            f"Institutional distance {institutional_distance:.2f} — "
            f"mixed field: some features align with biological reference, "
            f"others with the institutional standard."
        )
    else:
        parts.append(
            f"Institutional distance {institutional_distance:.2f} — "
            f"piece operates close to the biological resonance reference. "
            f"The frequency field is relatively in alignment with the pre-institutional harmonic series."
        )

    return " ".join(parts)

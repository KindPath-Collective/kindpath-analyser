"""
KindPath Analyser :: Divergence Module

The Late-Song Inversion Index (LSII) and intra-song divergence analysis.

This is the protest detection layer.

The LSII measures the degree to which the final quarter of a song diverges
from the emotional and sonic trajectory established by the first three quarters.
A high LSII indicates that something fundamentally different is happening
at the end of the piece - the creator has stepped outside the frame they built.

This is not always a protest. It can be:
- Deliberate artistic resolution (trajectory completes)
- Unexpected emotional collapse (the mask slips)
- Conscious subversion (the hidden message)
- Production override (the label changed the ending)

The tool doesn't interpret which - it detects the divergence and flags it.
The human educator or researcher interprets.

LSII SCALE:
0.0 - 0.2 : Low divergence - consistent emotional trajectory throughout
0.2 - 0.4 : Moderate divergence - notable shift but within established range
0.4 - 0.6 : High divergence - significant departure from trajectory  
0.6 - 0.8 : Very high divergence - strong inversion signature
0.8 - 1.0 : Extreme divergence - the final quarter is a different work
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from core.feature_extractor import SegmentFeatures


@dataclass
class DivergenceVector:
    """The divergence between Q4 and the Q1-Q3 baseline on each feature axis."""
    spectral_centroid_delta: float      # Brightness change
    spectral_flux_delta: float          # Energy volatility change
    spectral_complexity_delta: float    # Spectral richness change
    dynamic_range_delta: float          # Compression change
    dynamic_energy_delta: float         # Volume/energy change
    harmonic_tension_delta: float       # Tension increase/decrease
    harmonic_tonality_delta: float      # Tonal stability change
    harmonic_complexity_delta: float    # Chord complexity change
    temporal_groove_delta: float        # Timing precision change
    temporal_syncopation_delta: float   # Rhythmic character change
    temporal_onset_density_delta: float # Busyness change


@dataclass
class LatesonginversionResult:
    """
    The full Late-Song Inversion Index result.

    lsii: 0.0-1.0 overall inversion score
    direction: whether Q4 is warmer/brighter/tighter/looser than Q1-Q3
    dominant_axis: which feature axis shows the greatest divergence
    narrative: human-readable interpretation of what the data shows
    """
    lsii: float
    direction: str
    dominant_axis: str
    divergence: DivergenceVector
    q1_q3_baseline: Dict
    q4_values: Dict
    trajectory_description: str
    inversion_description: str
    flag_level: str  # 'none', 'low', 'moderate', 'high', 'extreme'
    flag_notes: str


@dataclass
class TrajectoryProfile:
    """
    The emotional/sonic arc across all four quarters.
    Shows how the work moves through time.
    """
    quarters: List[Dict]
    valence_arc: List[float]        # Emotional positivity per quarter
    energy_arc: List[float]         # Energy level per quarter
    complexity_arc: List[float]     # Creative complexity per quarter
    coherence_arc: List[float]      # Internal consistency per quarter
    tension_arc: List[float]        # Harmonic tension per quarter
    lsii_result: LatesonginversionResult


def compute_lsii(quarter_features: List[SegmentFeatures]) -> LatesonginversionResult:
    """
    Compute the Late-Song Inversion Index from four quarter feature sets.
    """
    if len(quarter_features) != 4:
        raise ValueError(f"Expected 4 quarters, got {len(quarter_features)}")

    # Compute baseline from Q1-Q3
    baseline = _compute_baseline(quarter_features[:3])
    q4 = _extract_scalar_features(quarter_features[3])

    # Compute divergence on each axis
    div = _compute_divergence_vector(baseline, q4)

    # Aggregate LSII score
    divergences = [
        abs(div.spectral_centroid_delta),
        abs(div.spectral_flux_delta),
        abs(div.dynamic_range_delta),
        abs(div.dynamic_energy_delta),
        abs(div.harmonic_tension_delta),
        abs(div.harmonic_tonality_delta),
        abs(div.harmonic_complexity_delta),
        abs(div.temporal_groove_delta),
        abs(div.temporal_syncopation_delta),
    ]
    lsii = float(np.mean(divergences))
    lsii = min(lsii, 1.0)

    # Dominant axis
    axis_names = [
        'spectral_brightness', 'spectral_energy_volatility', 
        'dynamic_range', 'dynamic_energy',
        'harmonic_tension', 'harmonic_tonality', 'harmonic_complexity',
        'temporal_groove', 'temporal_syncopation'
    ]
    dominant_idx = int(np.argmax(divergences))
    dominant_axis = axis_names[dominant_idx]

    # Direction
    direction = _interpret_direction(div)

    # Descriptions
    trajectory = _describe_trajectory(quarter_features[:3])
    inversion = _describe_inversion(div, lsii, dominant_axis)

    # Flag level
    if lsii < 0.2:
        flag_level = 'none'
        flag_notes = "Consistent trajectory throughout. No significant late-song divergence."
    elif lsii < 0.4:
        flag_level = 'low'
        flag_notes = "Minor late-song variation. May indicate natural musical development."
    elif lsii < 0.6:
        flag_level = 'moderate'
        flag_notes = "Notable divergence in final quarter. Warrants closer examination."
    elif lsii < 0.8:
        flag_level = 'high'
        flag_notes = "Strong late-song inversion. Possible protest signature or emotional break."
    else:
        flag_level = 'extreme'
        flag_notes = "Extreme divergence. Final quarter is sonically/emotionally inconsistent with the preceding work."

    return LatesonginversionResult(
        lsii=lsii,
        direction=direction,
        dominant_axis=dominant_axis,
        divergence=div,
        q1_q3_baseline=baseline,
        q4_values=q4,
        trajectory_description=trajectory,
        inversion_description=inversion,
        flag_level=flag_level,
        flag_notes=flag_notes,
    )


def compute_trajectory(quarter_features: List[SegmentFeatures]) -> TrajectoryProfile:
    """Build the full arc profile across all four quarters."""
    
    quarters_data = []
    valence_arc = []
    energy_arc = []
    complexity_arc = []
    coherence_arc = []
    tension_arc = []

    for qf in quarter_features:
        scalars = _extract_scalar_features(qf)
        quarters_data.append(scalars)

        # Valence proxy: brightness + harmonic major-ness - tension
        valence = _compute_valence(qf)
        valence_arc.append(valence)

        # Energy: RMS normalised
        energy = min(float(qf.dynamic.rms_mean * 50), 1.0)
        energy_arc.append(energy)

        # Complexity: harmonic + spectral combined
        complexity = min(
            (float(qf.harmonic.harmonic_complexity) * 5 +
             float(qf.spectral.flux_mean) * 0.1) / 2, 1.0
        )
        complexity_arc.append(complexity)

        # Coherence: inverse of variability across domains
        coherence = 1.0 - min(
            (float(qf.spectral.centroid_std) / (float(qf.spectral.centroid_mean) + 1e-10) +
             float(qf.dynamic.rms_std) / (float(qf.dynamic.rms_mean) + 1e-10)) / 2,
            1.0
        )
        coherence_arc.append(float(coherence))

        tension_arc.append(float(qf.harmonic.tension_ratio))

    lsii_result = compute_lsii(quarter_features)

    return TrajectoryProfile(
        quarters=quarters_data,
        valence_arc=valence_arc,
        energy_arc=energy_arc,
        complexity_arc=complexity_arc,
        coherence_arc=coherence_arc,
        tension_arc=tension_arc,
        lsii_result=lsii_result,
    )


def _compute_valence(features: SegmentFeatures) -> float:
    """
    Approximate emotional valence (positive/negative) from sonic features.
    Not a simple mapping - deliberately rough. Exact valence isn't the goal.
    Direction and change are what matter.
    """
    # Brightness contributes positively to perceived valence up to a point
    brightness_score = min(features.spectral.centroid_mean / 4000.0, 1.0)
    # Major tonality contributes positively
    tonality_score = features.harmonic.tonality_strength
    # High tension reduces valence
    tension_penalty = features.harmonic.tension_ratio
    # High dynamic range suggests more expressive (higher valence)
    range_score = min(features.dynamic.dynamic_range_db / 20.0, 1.0)

    valence = (brightness_score * 0.3 + tonality_score * 0.4 +
               range_score * 0.2 - tension_penalty * 0.3)
    return float(max(0.0, min(1.0, valence)))


def _extract_scalar_features(f: SegmentFeatures) -> Dict:
    return {
        'centroid': f.spectral.centroid_mean,
        'flux': f.spectral.flux_mean,
        'dynamic_range': f.dynamic.dynamic_range_db,
        'rms': f.dynamic.rms_mean,
        'crest_factor': f.dynamic.crest_factor_db,
        'tension': f.harmonic.tension_ratio,
        'tonality': f.harmonic.tonality_strength,
        'harmonic_complexity': f.harmonic.harmonic_complexity,
        'groove_deviation': f.temporal.groove_deviation_ms,
        'syncopation': f.temporal.syncopation_index,
        'onset_density': f.dynamic.onset_density,
    }


def _compute_baseline(features_list: List[SegmentFeatures]) -> Dict:
    """Average scalar features across Q1-Q3."""
    scalars = [_extract_scalar_features(f) for f in features_list]
    keys = scalars[0].keys()
    return {k: float(np.mean([s[k] for s in scalars])) for k in keys}


def _compute_divergence_vector(baseline: Dict, q4: Dict) -> DivergenceVector:
    """
    Normalised divergence on each axis.
    Normalisation prevents high-magnitude features dominating.
    """
    def norm_delta(key, scale=1.0):
        b = baseline.get(key, 0)
        q = q4.get(key, 0)
        delta = (q - b) / (abs(b) + 1e-6)
        return float(np.tanh(delta * scale))  # Bound to -1..1

    return DivergenceVector(
        spectral_centroid_delta=norm_delta('centroid', 0.5),
        spectral_flux_delta=norm_delta('flux', 2.0),
        spectral_complexity_delta=norm_delta('flux', 1.0),
        dynamic_range_delta=norm_delta('dynamic_range', 1.0),
        dynamic_energy_delta=norm_delta('rms', 3.0),
        harmonic_tension_delta=norm_delta('tension', 2.0),
        harmonic_tonality_delta=norm_delta('tonality', 2.0),
        harmonic_complexity_delta=norm_delta('harmonic_complexity', 1.0),
        temporal_groove_delta=norm_delta('groove_deviation', 0.2),
        temporal_syncopation_delta=norm_delta('syncopation', 2.0),
        temporal_onset_density_delta=norm_delta('onset_density', 1.0),
    )


def _interpret_direction(div: DivergenceVector) -> str:
    directions = []
    if div.spectral_centroid_delta > 0.2:
        directions.append("brighter")
    elif div.spectral_centroid_delta < -0.2:
        directions.append("darker")
    if div.dynamic_energy_delta < -0.2:
        directions.append("quieter")
    elif div.dynamic_energy_delta > 0.2:
        directions.append("louder")
    if div.harmonic_tension_delta > 0.2:
        directions.append("more tense")
    elif div.harmonic_tension_delta < -0.2:
        directions.append("more resolved")
    if div.temporal_groove_delta > 0.2:
        directions.append("looser in time")
    elif div.temporal_groove_delta < -0.2:
        directions.append("more rigid")
    if div.dynamic_range_delta > 0.2:
        directions.append("more dynamic")
    elif div.dynamic_range_delta < -0.2:
        directions.append("more compressed")
    return ", ".join(directions) if directions else "minimal directional shift"


def _describe_trajectory(q1_q3: List[SegmentFeatures]) -> str:
    energies = [f.dynamic.rms_mean for f in q1_q3]
    tensions = [f.harmonic.tension_ratio for f in q1_q3]
    if energies[-1] > energies[0] * 1.2:
        energy_desc = "building energy"
    elif energies[-1] < energies[0] * 0.8:
        energy_desc = "fading energy"
    else:
        energy_desc = "stable energy"
    if tensions[-1] > tensions[0] + 0.1:
        tension_desc = "increasing tension"
    elif tensions[-1] < tensions[0] - 0.1:
        tension_desc = "resolving tension"
    else:
        tension_desc = "stable tension"
    return f"Q1-Q3 establishes {energy_desc} with {tension_desc}"


def _describe_inversion(div: DivergenceVector, lsii: float, dominant: str) -> str:
    if lsii < 0.2:
        return "Q4 continues the established trajectory without significant deviation."
    dominant_readable = dominant.replace('_', ' ')
    return (f"Q4 shows significant divergence on {dominant_readable} axis "
            f"(LSII: {lsii:.3f}). The work departs from its established emotional frame "
            f"in the final quarter. Direction: {_interpret_direction(div)}.")

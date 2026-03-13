"""
KindPath Analyser :: Psychosomatic Profiler

Takes the raw feature data from all analysis modules and maps it onto
a psychosomatic model: what is this music doing to a human body, and what
does that tell us about the state of the person or culture that made it?

This is not neuroscience claiming to be definitive. It is a rigorous
attempt to name what the data suggests about the pre-linguistic emotional
transmission embedded in the work.

The three-stage mechanism this module detects:
  Stage 1 - PSYCHOSOMATIC PRIMING: specific frequency/rhythm/dynamics
             combos that install emotional states before conscious mind engages
  Stage 2 - FALSE PRESTIGE ATTACHMENT: production quality signals attached
             to creatively empty work
  Stage 3 - IDENTITY CAPTURE: engineered attachment response, repeat-listen
             compulsion without commensurate creative content

Understanding the mechanism is not accusation. It is literacy.
The tool does not judge creators. It reads signals.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from core.divergence import TrajectoryProfile, LatesonginversionResult
from core.fingerprints import FingerprintReport
from core.solfeggio import SolfeggioAlignment


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class PsychosomaticProfile:
    """
    What this music is doing to a human body.
    And what the creator's body was doing when they made it.
    """

    # Russell Circumplex Model axes
    valence: float              # -1.0 to 1.0: negative to positive emotional tone
    arousal: float              # 0.0 to 1.0: sedating to activating

    # Extended KindPath axes
    coherence: float            # 0-1: internal creative consistency
    authenticity_index: float   # 0-1: authentic deviation from convention
    complexity: float           # 0-1: creative information density
    tension_resolution_ratio: float  # >1 = unresolved, <1 = resolving
    relational_density: float   # Silence as data — space between elements

    # Somatic response predictions
    predicted_physical_responses: List[str]
    predicted_emotional_states: List[str]

    # Conditioning assessment
    priming_vectors: List[str]          # What states this installs pre-linguistically
    prestige_signals: List[str]         # False prestige markers
    identity_capture_risk: float        # 0-1: likelihood of identity attachment

    # Three-stage mechanism
    stage1_priming_detected: bool
    stage1_evidence: List[str]
    stage2_prestige_attached: bool
    stage2_evidence: List[str]
    stage3_tag_risk: float
    stage3_evidence: List[str]

    # LSII psychosomatic reading
    lsii_psychosomatic_reading: str

    # Authentic vs manufactured
    authentic_emission_score: float     # 0-1: authentic creative signal
    manufacturing_score: float          # 0-1: engineered delivery mechanism
    creative_residue: float             # What remains after known signatures subtracted

    # Frequency field alignment (Solfeggio / biological reference measurement).
    # Three simultaneous readings: vocal fundamental deviation, Solfeggio grid proximity,
    # and institutional distance from biological reference. Optional — present when
    # quarter_features were available during profile construction.
    solfeggio_alignment: Optional[SolfeggioAlignment] = None

    # Narrative — the synthetic elder's voice
    somatic_summary: str = ''           # What this is doing to a body
    mechanism_summary: str = ''         # Conditioning mechanics if detected
    elder_reading: str = ''             # Full synthesis


# ── Builder ──────────────────────────────────────────────────────────────────

def build_psychosomatic_profile(
    trajectory: TrajectoryProfile,
    fingerprints: FingerprintReport,
    stem_features: Optional[Dict] = None,
    solfeggio_alignment: Optional[SolfeggioAlignment] = None,
) -> PsychosomaticProfile:
    """
    Synthesise all analysis into a psychosomatic profile.

    Args:
        trajectory: TrajectoryProfile from core.divergence
        fingerprints: FingerprintReport from core.fingerprints
        stem_features: Optional dict of stem_name -> SegmentFeatures
    """
    lsii = trajectory.lsii_result

    valence = _compute_valence(trajectory)
    arousal = _compute_arousal(trajectory)
    coherence = _compute_coherence(trajectory)
    complexity = _compute_complexity(trajectory)
    tension_ratio = _compute_tension_resolution_ratio(trajectory)
    relational_density = _compute_relational_density(trajectory)
    authenticity_index = _compute_authenticity_index(trajectory, fingerprints)

    stage1_detected, stage1_evidence = _detect_stage1_priming(trajectory, fingerprints)
    stage2_detected, stage2_evidence = _detect_stage2_prestige(fingerprints, trajectory)
    stage3_risk, stage3_evidence = _compute_stage3_tag_risk(trajectory, fingerprints)

    authentic_emission = _compute_authentic_emission(trajectory, fingerprints)
    manufacturing = _compute_manufacturing_score(fingerprints, trajectory)
    creative_residue = _compute_creative_residue(trajectory, fingerprints)

    priming_vectors = _build_priming_vectors(trajectory, fingerprints, solfeggio_alignment)
    prestige_signals = _extract_prestige_signals(fingerprints)
    identity_capture_risk = min(0.98, (stage3_risk + manufacturing * 0.3) / 1.3)

    physical = _predict_physical(arousal, valence, trajectory)
    emotional = _predict_emotional(valence, arousal, tension_ratio)

    lsii_reading = _lsii_psychosomatic_reading(lsii)

    somatic_summary = _build_somatic_summary(arousal, valence, tension_ratio, trajectory)
    mechanism_summary = _build_mechanism_summary(
        stage1_detected, stage2_detected, stage3_risk, priming_vectors, prestige_signals
    )
    elder_reading = _build_elder_reading(
        trajectory=trajectory,
        lsii=lsii,
        authentic_emission=authentic_emission,
        manufacturing=manufacturing,
        creative_residue=creative_residue,
        stage1_detected=stage1_detected,
        stage2_detected=stage2_detected,
        stage3_risk=stage3_risk,
        valence=valence,
        arousal=arousal,
        tension_ratio=tension_ratio,
        solfeggio_alignment=solfeggio_alignment,
    )

    return PsychosomaticProfile(
        valence=valence,
        arousal=arousal,
        coherence=coherence,
        authenticity_index=authenticity_index,
        complexity=complexity,
        tension_resolution_ratio=tension_ratio,
        relational_density=relational_density,
        predicted_physical_responses=physical,
        predicted_emotional_states=emotional,
        priming_vectors=priming_vectors,
        prestige_signals=prestige_signals,
        identity_capture_risk=identity_capture_risk,
        stage1_priming_detected=stage1_detected,
        stage1_evidence=stage1_evidence,
        stage2_prestige_attached=stage2_detected,
        stage2_evidence=stage2_evidence,
        stage3_tag_risk=stage3_risk,
        stage3_evidence=stage3_evidence,
        lsii_psychosomatic_reading=lsii_reading,
        authentic_emission_score=authentic_emission,
        manufacturing_score=manufacturing,
        creative_residue=creative_residue,
        solfeggio_alignment=solfeggio_alignment,
        somatic_summary=somatic_summary,
        mechanism_summary=mechanism_summary,
        elder_reading=elder_reading,
    )


# ── Axis computations ─────────────────────────────────────────────────────────

def _compute_valence(traj: TrajectoryProfile) -> float:
    """
    Valence from harmonic tension arc and energy arc.
    Lower tension + higher energy → positive valence.
    Higher tension + decaying energy → negative.
    """
    if not traj.tension_arc or not traj.energy_arc:
        return 0.0
    mean_tension = float(np.mean(traj.tension_arc))
    mean_energy = float(np.mean(traj.energy_arc))
    # Tension pulls valence negative; energy pulls it toward neutral-positive
    raw = (1.0 - mean_tension) * 0.6 + (mean_energy - 0.5) * 0.4
    return float(np.clip(raw * 2 - 1, -1.0, 1.0))


def _compute_arousal(traj: TrajectoryProfile) -> float:
    """
    Arousal from energy arc mean. High energy = high arousal.
    """
    if not traj.energy_arc:
        return 0.5
    return float(np.clip(np.mean(traj.energy_arc), 0.0, 1.0))


def _compute_coherence(traj: TrajectoryProfile) -> float:
    if not traj.coherence_arc:
        return 0.5
    return float(np.clip(np.mean(traj.coherence_arc), 0.0, 1.0))


def _compute_complexity(traj: TrajectoryProfile) -> float:
    if not traj.complexity_arc:
        return 0.5
    return float(np.clip(np.mean(traj.complexity_arc), 0.0, 1.0))


def _compute_tension_resolution_ratio(traj: TrajectoryProfile) -> float:
    """
    Ratio of mean tension in Q4 vs Q1-Q3.
    >1 = unresolved (Q4 more tense than buildup)
    <1 = resolving (Q4 less tense)
    """
    if not traj.tension_arc or len(traj.tension_arc) < 4:
        return 1.0
    baseline_tension = float(np.mean(traj.tension_arc[:3]))
    q4_tension = traj.tension_arc[3]
    if baseline_tension < 1e-6:
        return 1.0
    return float(q4_tension / baseline_tension)


def _compute_relational_density(traj: TrajectoryProfile) -> float:
    """
    Amount of space in the music — silence, sparse texture.
    Derived from the inverse of energy density and spectral bandwidth.
    Lower energy_arc mean → more space → higher relational density.
    """
    if not traj.energy_arc:
        return 0.5
    energy = float(np.mean(traj.energy_arc))
    return float(np.clip(1.0 - energy, 0.0, 1.0))


def _compute_authenticity_index(traj: TrajectoryProfile, fp: FingerprintReport) -> float:
    """
    Deviation from genre/era convention that is not noise.
    Low manufacturing markers + high coherence + present groove deviation → high authenticity.
    """
    manufacturing = _compute_manufacturing_score(fp, traj)
    coherence = _compute_coherence(traj)
    return float(np.clip((1.0 - manufacturing) * 0.6 + coherence * 0.4, 0.0, 1.0))


# ── Three-stage detection ─────────────────────────────────────────────────────

def _detect_stage1_priming(traj: TrajectoryProfile, fp: FingerprintReport):
    """
    Stage 1: Psychosomatic priming detection.

    Priming uses specific combinations to install emotional states
    before the conscious mind engages. Markers:
    - Tempo in 120–140 BPM range (heart rate entrainment)
    - High arousal in Q1 (hook-first structure)
    - Sub-bass prominence combined with rhythmic density
    - Startle/relief cycle (sudden silence then return)
    - Very fast spectral centroid rise at entry
    """
    evidence = []
    detected = False

    if traj.quarters:
        q1 = traj.quarters[0]
        tempo = q1.get('tempo_bpm', 0) or 0
        if 118 <= tempo <= 142:
            evidence.append(f"Tempo {tempo:.0f} BPM — heart-rate entrainment range (120–140)")
            detected = True

        onset_density = q1.get('onset_density', 0) or 0
        if onset_density > 5.0:
            evidence.append(f"High onset density in Q1 ({onset_density:.1f}/s) — immediate rhythmic activation")
            detected = True

    if traj.energy_arc and len(traj.energy_arc) >= 2:
        # High Q1 energy → hook-first priming
        if traj.energy_arc[0] > 0.7:
            evidence.append("High initial energy (hook-first) — engages before critical faculties set up")
            detected = True

    # Check for manufacturing markers containing "compression" or "sidechain"
    manufacturing_markers = getattr(fp, 'manufacturing_markers', []) or []
    for m in manufacturing_markers:
        if 'sidechain' in str(m).lower():
            evidence.append("Sidechain compression detected — rhythmic pumping creates subliminal pulse")
            detected = True
            break

    return detected, evidence


def _detect_stage2_prestige(fp: FingerprintReport, traj: TrajectoryProfile):
    """
    Stage 2: False prestige detection.

    High production quality + low creative residue = prestige attached to
    creatively empty work.
    """
    evidence = []
    detected = False

    manufacturing_markers = getattr(fp, 'manufacturing_markers', []) or []
    authenticity_markers = getattr(fp, 'authenticity_markers', []) or []

    # High production quality signals
    high_production = any('studio' in str(m).lower() or 'mastered' in str(m).lower()
                           or 'professional' in str(m).lower() for m in manufacturing_markers)
    # Low authentic signal
    low_authentic = len(authenticity_markers) == 0

    if high_production and low_authentic:
        evidence.append("Professional-grade production markers present with low authentic deviation")
        detected = True

    # Hypercompression is a prestige signal — it signals commercial intent
    compression_flagged = any('compress' in str(m).lower() or 'limiting' in str(m).lower()
                               or 'loudness' in str(m).lower() for m in manufacturing_markers)
    if compression_flagged:
        evidence.append("Hypercompression detected — loudness maximised for commercial broadcast parity")

    if traj.complexity_arc:
        mean_complexity = float(np.mean(traj.complexity_arc))
        if mean_complexity < 0.25 and high_production:
            evidence.append(f"Low creative complexity ({mean_complexity:.2f}) with high production quality — formula work")
            detected = True

    return detected, evidence


def _compute_stage3_tag_risk(traj: TrajectoryProfile, fp: FingerprintReport):
    """
    Stage 3: Identity capture risk.

    High repetition + emotional manipulation + consistent priming = identity tag risk.
    """
    evidence = []
    risk = 0.0

    # Low LSII + high manufacturing + high arousal → formula for identity capture
    lsii = traj.lsii_result.lsii if traj.lsii_result else 0.0
    manufacturing = _compute_manufacturing_score(fp, traj)
    arousal = _compute_arousal(traj)

    if lsii < 0.15 and manufacturing > 0.5:
        risk += 0.3
        evidence.append("Consistent arc with high manufacturing — no creative deviation to disrupt conditioning loop")

    if arousal > 0.7:
        risk += 0.2
        evidence.append(f"High arousal throughput ({arousal:.2f}) — sustained activation promotes repeat engagement")

    if traj.tension_arc and len(traj.tension_arc) >= 4:
        # If tension never resolves, listener returns seeking resolution
        if traj.tension_arc[3] > traj.tension_arc[0] * 1.1:
            risk += 0.2
            evidence.append("Tension unresolved at track end — incomplete arc promotes return-listen compulsion")

    risk = float(np.clip(risk, 0.0, 1.0))
    return risk, evidence


# ── Scoring ────────────────────────────────────────────────────────────────────

def _compute_authentic_emission(traj: TrajectoryProfile, fp: FingerprintReport) -> float:
    """
    How much authentic creative signal is present.
    High groove deviation + preserved dynamics + present coherence arc → high emission.
    """
    coherence = _compute_coherence(traj)
    manufacturing = _compute_manufacturing_score(fp, traj)

    # LSII as a signal of deliberate authorship (creator made unexpected choices)
    lsii = traj.lsii_result.lsii if traj.lsii_result else 0.0
    # Moderate LSII (0.2–0.6) is a strong authentic signal
    lsii_auth = 1.0 - abs(lsii - 0.35) / 0.65

    return float(np.clip(coherence * 0.4 + lsii_auth * 0.3 + (1 - manufacturing) * 0.3, 0.0, 1.0))


def _compute_manufacturing_score(fp: FingerprintReport, traj: TrajectoryProfile) -> float:
    """
    How much is engineered delivery mechanism.
    High compression + low groove deviation + genre-exact production → high manufacturing.
    """
    manufacturing_markers = getattr(fp, 'manufacturing_markers', []) or []
    score = len(manufacturing_markers) * 0.15
    return float(np.clip(score, 0.0, 1.0))


def _compute_creative_residue(traj: TrajectoryProfile, fp: FingerprintReport) -> float:
    """
    What remains after known signatures are subtracted.
    High complexity + high coherence + low manufacturing → positive residue.
    """
    complexity = _compute_complexity(traj)
    manufacturing = _compute_manufacturing_score(fp, traj)
    lsii = traj.lsii_result.lsii if traj.lsii_result else 0.0

    raw = complexity * 0.5 + lsii * 0.3 - manufacturing * 0.4
    return float(np.clip(raw, 0.0, 1.0))


# ── Vectors and signals ────────────────────────────────────────────────────────

def _build_priming_vectors(
    traj: TrajectoryProfile,
    fp: FingerprintReport,
    solfeggio_alignment: Optional[SolfeggioAlignment] = None,
) -> List[str]:
    vectors = []
    arousal = _compute_arousal(traj)
    valence = _compute_valence(traj)
    if arousal > 0.7:
        vectors.append("activation / forward motion")
    if valence < -0.3:
        vectors.append("melancholic pre-set / longing installation")
    if valence > 0.4 and arousal > 0.5:
        vectors.append("reward-anticipation loop")

    # Frequency field priming: high institutional distance means the piece
    # operates outside the biological resonance range — the body registers this
    # as a field condition before conscious engagement begins.
    if solfeggio_alignment is not None:
        if solfeggio_alignment.institutional_distance > 0.7:
            vectors.append(
                "institutional frequency field — tonal centre anchored to A440 standard, "
                "operating outside biological vocal resonance range (~130 Hz)"
            )
        elif solfeggio_alignment.solfeggio_grid_proximity > 0.5:
            vectors.append(
                f"Solfeggio grid resonance — prominent frequencies align with "
                f"pre-institutional harmonic series ({solfeggio_alignment.nearest_solfeggio_name})"
            )

    if not vectors:
        vectors.append("low-priming presence — no strong psychosomatic pre-set detected")
    return vectors


def _extract_prestige_signals(fp: FingerprintReport) -> List[str]:
    signals = []
    manufacturing_markers = getattr(fp, 'manufacturing_markers', []) or []
    for m in manufacturing_markers:
        ms = str(m).lower()
        if 'professional' in ms or 'studio' in ms:
            signals.append("professional studio quality")
        if 'spatial' in ms or 'reverb' in ms:
            signals.append("high-end spatial processing")
        if 'loudness' in ms or 'compress' in ms:
            signals.append("commercial mastering level")
    return list(set(signals))


def _predict_physical(arousal: float, valence: float, traj: TrajectoryProfile) -> List[str]:
    responses = []
    if arousal > 0.7:
        responses.extend(["increased heart rate", "heightened alertness"])
    elif arousal < 0.3:
        responses.extend(["physical slowing", "decreased muscle tension"])
    if valence < -0.3:
        responses.append("chest tightening or throat restriction")
    if valence > 0.4:
        responses.append("chest opening / expansive physical sensation")
    if traj.tension_arc and max(traj.tension_arc) > 0.7:
        responses.append("held breath or shallow breathing")
    if not responses:
        responses.append("mild alertness response")
    return responses


def _predict_emotional(valence: float, arousal: float, tension_ratio: float) -> List[str]:
    states = []
    if valence < -0.4 and arousal < 0.4:
        states.extend(["melancholy", "contemplative grief"])
    elif valence < -0.2 and arousal > 0.5:
        states.extend(["frustration", "unresolved longing"])
    elif valence > 0.3 and arousal > 0.6:
        states.extend(["elation", "forward momentum"])
    elif valence > 0.2 and arousal < 0.4:
        states.extend(["contentment", "settled warmth"])
    else:
        states.append("neutral arousal — open receptive state")
    if tension_ratio > 1.3:
        states.append("unresolved tension seeking completion")
    return states


# ── LSII psychosomatic reading ─────────────────────────────────────────────────

def _lsii_psychosomatic_reading(lsii: LatesonginversionResult) -> str:
    score = lsii.lsii
    direction = lsii.direction
    dominant = lsii.dominant_axis

    if score < 0.15:
        return (
            "The arc is consistent throughout. What you feel at the end is what "
            "the piece established at the beginning. No psychosomatic reframe in the final section."
        )
    elif score < 0.35:
        return (
            f"A mild late-section shift on the {dominant} axis — the piece moves slightly "
            f"{direction} in its final quarter. Not a full inversion, but a thoughtful adjustment."
        )
    elif score < 0.55:
        return (
            f"Notable late-section divergence ({dominant} axis). The piece shifts "
            f"{direction} in its final quarter. The emotional frame that was established "
            "has been quietly adjusted. Worth listening to that transition closely."
        )
    elif score < 0.75:
        return (
            f"Strong late inversion detected (dominant: {dominant}). The piece moves "
            f"{direction} in Q4 — significantly departing from what was established. "
            "The body is given different instructions at the end. This is deliberate."
        )
    else:
        return (
            f"Extreme late-section inversion (LSII {score:.2f}, {dominant} axis). "
            f"The final quarter is {direction} — functionally a different emotional work "
            "from what preceded it. The creator stepped entirely outside the frame they built. "
            "Something is being communicated that could not be said in the first three-quarters."
        )


# ── Narrative builders ─────────────────────────────────────────────────────────

def _build_somatic_summary(arousal: float, valence: float, tension_ratio: float,
                             traj: TrajectoryProfile) -> str:
    energy_dir = "high" if arousal > 0.6 else ("low" if arousal < 0.35 else "moderate")
    emotional_tone = ("positive" if valence > 0.2 else
                      "negative" if valence < -0.2 else "neutral")
    resolution = ("unresolved — tension builds toward the end" if tension_ratio > 1.2 else
                  "resolving — tension decreases toward the end" if tension_ratio < 0.8 else
                  "maintained — tension holds steady throughout")

    return (
        f"This piece operates at {energy_dir} arousal with {emotional_tone} emotional valence. "
        f"Tension: {resolution}."
    )


def _build_mechanism_summary(stage1: bool, stage2: bool, stage3_risk: float,
                              priming_vectors: List[str], prestige_signals: List[str]) -> str:
    if not stage1 and not stage2 and stage3_risk < 0.3:
        return "No significant conditioning mechanisms detected."

    parts = []
    if stage1:
        parts.append(
            f"Stage 1 priming present — emotional states pre-set via: "
            f"{', '.join(priming_vectors[:2])}."
        )
    if stage2:
        parts.append(
            f"Stage 2 prestige attachment — {', '.join(prestige_signals[:2]) or 'production quality signals'} "
            f"carry social status signal independent of creative content."
        )
    if stage3_risk > 0.3:
        parts.append(
            f"Stage 3 identity capture risk: {stage3_risk:.0%}. "
            "The architecture supports repeat-listen conditioning."
        )
    return " ".join(parts)


def _build_elder_reading(*, trajectory, lsii, authentic_emission, manufacturing,
                          creative_residue, stage1_detected, stage2_detected,
                          stage3_risk, valence, arousal, tension_ratio,
                          solfeggio_alignment: Optional[SolfeggioAlignment] = None) -> str:
    """
    The synthetic elder's voice. Clear, non-judgmental, precise.
    Reads like wisdom, not analysis.
    """
    parts = []

    # Opening: what the piece does to the body
    if arousal > 0.65 and valence > 0.2:
        parts.append(
            "This piece moves forward. It carries the body upward and ahead."
        )
    elif arousal < 0.35 and valence < -0.1:
        parts.append(
            "This piece settles. It slows the body and creates interior space."
        )
    elif valence < -0.35:
        parts.append(
            "This piece carries weight. It installs a particular kind of heaviness."
        )
    else:
        parts.append(
            "This piece holds a steady emotional field — neither particularly uplifting nor darkening."
        )

    # LSII reading if significant
    if lsii.lsii > 0.25:
        direction = lsii.direction or "differently"
        axis = lsii.dominant_axis or "emotional character"
        parts.append(
            f"The final quarter moves {direction} — the {axis} shifts from what "
            "preceded it. Whether this is intended protest, emotional collapse, or "
            "deliberate resolution: the creator stepped outside the frame they built."
        )

    # Manufacturing vs authentic
    if manufacturing > 0.5 and authentic_emission < 0.4:
        parts.append(
            "The production is technically accomplished. "
            "The creative residue is low — most choices match commercial expectations for their era. "
            "This piece does its job efficiently."
        )
    elif authentic_emission > 0.6 and creative_residue > 0.4:
        parts.append(
            "Strong authentic emission present. The timing is alive — the performer was present. "
            "The choices deviate from convention in ways that are internally consistent. "
            "This suggests deliberate authorship, not formula."
        )
    elif authentic_emission > 0.4:
        parts.append(
            "A mixture of authentic signal and convention. Something real was made here, "
            "constrained by or in dialogue with the commercial context."
        )

    # Conditioning if present
    if stage1_detected and stage2_detected:
        parts.append(
            "This piece shows both priming architecture and prestige signalling. "
            "The mechanism is present. Knowing it is there is enough to engage it differently."
        )
    elif stage1_detected:
        parts.append(
            "Priming mechanics are present in the opening architecture. "
            "The emotional state is installed before the conscious mind is engaged."
        )

    # Tension resolution
    if tension_ratio > 1.3:
        parts.append(
            "The tension in this piece does not fully resolve. "
            "It ends in a state of incompletion. Whether this is artistic or commercial intent "
            "is not readable from the signal alone."
        )

    # Frequency field reading — what the tuning and tonal centre reveal about
    # the relationship between this piece and biological/institutional standards.
    if solfeggio_alignment is not None:
        if solfeggio_alignment.institutional_distance > 0.5:
            parts.append(
                f"The frequency field sits at an institutional distance of "
                f"{solfeggio_alignment.institutional_distance:.2f} from biological reference. "
                f"{solfeggio_alignment.alignment_reading}"
            )
        elif solfeggio_alignment.solfeggio_grid_proximity > 0.4:
            parts.append(
                f"The tonal material shows alignment with the pre-institutional harmonic series "
                f"(nearest: {solfeggio_alignment.nearest_solfeggio_name} at "
                f"{solfeggio_alignment.nearest_solfeggio_hz:.0f} Hz). "
                f"{solfeggio_alignment.alignment_reading}"
            )

    return " ".join(parts)

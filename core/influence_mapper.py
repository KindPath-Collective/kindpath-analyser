"""
KindPath Analyser :: Influence Chain Mapper

The lineage mapping module — the elder function made technical.
Every sonic choice has ancestors. Every technique has a history.
This module traces those histories, making the mechanism visible.

This is not plagiarism detection. It is the recognition that creativity
is never born in a vacuum — it comes from somewhere, and knowing where
it came from is knowing what it carries with it.

The second function of this module is Kindfluence integration: detecting
which comprehension mechanics are present in the sonic signal, and
classifying their direction (extractive toward consumption vs syntropic
toward regeneration). This gives the Kindfluence system a grounded sonic
basis for its counter-mechanic recommendations — the specific vectors
needed to bend the field back toward syntropy.

The 13 comprehension mechanics (from Kindfluence ComprehensionEngine):
  priming, social_proof, anchoring, reciprocity, mere_exposure,
  framing, narrative_transportation, dopamine_loops, identity_association,
  loss_aversion, cognitive_ease, authority_bias, bandwagon_effect

Each mechanic has a corporate (extractive) use and a KindPath (syntropic)
use. This module detects which are present in the signal and, for
extractive uses, generates the syntropy repair vector — the counter-mechanic
that bends the field back.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from core.fingerprints import FingerprintReport
from core.feature_extractor import SegmentFeatures

# Path to the JSON reference library
_FINGERPRINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "fingerprints")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class InfluenceNode:
    """One traceable node in a sonic lineage chain."""
    id: str                             # Key matching lineages.json
    name: str                           # Human-readable era/genre name
    era_range: Tuple[int, int]          # (start_year, end_year) — 9999 = ongoing
    parent_lineages: List[str]          # IDs of parent lineage nodes
    sonic_characteristics: Dict         # Measurable sonic profile
    cultural_context: str               # What was happening when this sound emerged
    kindpath_notes: str                 # What this tradition carries for regenerative work
    confidence: float                   # 0-1 match confidence


@dataclass
class DetectedMechanic:
    """
    A comprehension mechanic detected in the sonic signal.
    Maps directly to the Kindfluence ComprehensionEngine.MECHANICS catalogue.
    """
    name: str                           # One of the 13 Kindfluence mechanics
    evidence: List[str]                 # Sonic markers that indicate this mechanic
    direction: str                      # 'extractive', 'syntropic', or 'ambiguous'
    confidence: float                   # 0-1
    syntropy_repair: str                # Counter-mechanic for bending back to syntropy


@dataclass
class InfluenceChain:
    """
    The traceable sonic lineage of a piece, with Kindfluence mechanic analysis.

    Two parallel readings:
    1. Where did this sound come from? (primary_lineage, secondary_lineages)
    2. What is it doing to the listener? (detected_mechanics, mechanic_direction)

    The combination tells you not just what tradition a piece descends from,
    but whether it is using that tradition's energy toward or against
    the listener's autonomy.
    """
    primary_lineage: List[InfluenceNode]          # Main tradition this descends from
    secondary_lineages: List[List[InfluenceNode]] # Other traceable threads
    innovation_points: List[str]                  # Where this piece departs from lineage
    confluence_points: List[str]                  # Where multiple lineages merge
    narrative: str                                # Lineage story in plain language

    # Kindfluence connection — comprehension mechanic analysis
    detected_mechanics: List[DetectedMechanic]    # Which of the 13 mechanics are present
    mechanic_direction: str                       # 'extractive', 'syntropic', 'mixed', 'neutral'
    gradient_stage_estimate: str                  # Estimated MessagingGradient stage
    syntropy_repair_vectors: List[str]            # Counter-mechanics to bend the field back
    mechanic_summary: str                         # Plain-language summary of mechanic use


# ─────────────────────────────────────────────
# SYNTROPY REPAIR CATALOGUE
# Directly mirrors KindPath uses from Kindfluence ComprehensionEngine
# ─────────────────────────────────────────────

_SYNTROPY_REPAIRS: Dict[str, str] = {
    "priming": (
        "Counter with presence priming: breath, pause, and space teach the body to "
        "settle rather than prime to consume. Contextual cues that activate regenerative "
        "thinking — notice what is already sufficient. Disclose the priming mechanism."
    ),
    "social_proof": (
        "Counter with authentic community voice: showcase real people doing real work, "
        "not manufactured consensus. Amplify community members rather than constructing "
        "a popularity illusion. Full transparency on what is curated and why."
    ),
    "anchoring": (
        "Counter with long-horizon anchoring: set collective wellbeing as the baseline "
        "reference point. Frame short-term thinking as the deviation. Explain the "
        "framing choice when shifting baseline assumptions."
    ),
    "reciprocity": (
        "Already pointed syntropic: give freely, no strings. Provide genuine value "
        "(tools, knowledge, resources) without expecting conversion. Explicitly state "
        "no obligation — we give because it serves the whole."
    ),
    "mere_exposure": (
        "Counter with varied authentic presence: repeat core values themes in genuinely "
        "different contexts and forms. Acknowledge intentional reinforcement. Avoid "
        "saturation — repeated emptiness builds familiarity with emptiness."
    ),
    "framing": (
        "Counter with transparent reframing: make regeneration the assumed norm, "
        "extraction the named aberration. Explain framing choices when challenging "
        "mainstream narratives. Centre community thriving, not consumer problem-solving."
    ),
    "narrative_transportation": (
        "Counter with authentic story: real people, real transformation, no fabrication. "
        "Invite critical thinking even within narrative immersion. Identify stories as "
        "illustrative, not prescriptive. Audiences should remain thinking, not bypassed."
    ),
    "dopamine_loops": (
        "This mechanic should be dismantled, not redirected. Design for satisfaction "
        "and closure, not endless engagement. Explicitly reject variable reward patterns. "
        "Create content that empowers exit, not dependency. Disclose the choice not to "
        "use addictive patterns."
    ),
    "identity_association": (
        "Counter with values-based identity: connect regenerative values to existing "
        "identities (parent, community member, creator) rather than creating brand-dependent "
        "identity. Values transcend any organisation. Explicitly reject brand-as-identity."
    ),
    "loss_aversion": (
        "Use only to motivate on climate or justice — never to manipulate engagement or "
        "extract resources. Balance urgency with agency. Never use to create scarcity "
        "manipulation. Acknowledge urgency framing and explain why."
    ),
    "cognitive_ease": (
        "Counter with accessible depth: plain language that preserves nuance rather than "
        "simplifying. Easy access to complex ideas — not the replacement of complex ideas "
        "with empty simplicity. Acknowledge simplification; invite deeper exploration."
    ),
    "authority_bias": (
        "Counter with distributed authority: amplify genuine expertise AND lived community "
        "wisdom. Decentralise rather than concentrate. Cite credible sources without "
        "creating guru dynamics. Disclose relationships with featured voices."
    ),
    "bandwagon_effect": (
        "Counter with authentic momentum: celebrate real collective action, make genuine "
        "regeneration visible as emerging norm. Never fabricate popularity. Full transparency "
        "on engagement data — never inflate numbers. Show real people joining something genuine."
    ),
}

# MessagingGradient stage estimates based on mechanic direction profile
# Maps the analyser's reading to the Kindfluence gradient positioning
_GRADIENT_STAGE_MAP = {
    "extractive": "INFILTRATE",    # Fully niche/commercial, values absent or inverted
    "syntropic": "EMERGE",         # Values meaningfully present in the sonic field
    "mixed": "SEED",               # Transition — some authentic signal amid commercial structure
    "neutral": "SEED",             # Ambiguous positioning
}


# ─────────────────────────────────────────────
# LINEAGE LOADING
# ─────────────────────────────────────────────

_lineages_cache: Optional[Dict] = None


def _load_lineages() -> Dict:
    """Load lineages.json, caching after first read."""
    global _lineages_cache
    if _lineages_cache is None:
        path = os.path.join(_FINGERPRINTS_DIR, "lineages.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            _lineages_cache = {e["id"]: e for e in data.get("lineages", [])}
        else:
            _lineages_cache = {}
    return _lineages_cache


def _lineage_to_node(entry: Dict, confidence: float) -> InfluenceNode:
    """Convert a lineages.json entry to an InfluenceNode."""
    era = entry.get("era_range", [1900, 9999])
    return InfluenceNode(
        id=entry["id"],
        name=entry["name"],
        era_range=(era[0], era[1]),
        parent_lineages=entry.get("parent_lineages", []),
        sonic_characteristics=entry.get("sonic_characteristics", {}),
        cultural_context=entry.get("cultural_context", ""),
        kindpath_notes=entry.get("kindpath_notes", ""),
        confidence=confidence,
    )


# ─────────────────────────────────────────────
# LINEAGE MATCHING
# ─────────────────────────────────────────────

def _match_lineages(
    fingerprints: FingerprintReport,
    features: Optional[SegmentFeatures] = None,
) -> List[InfluenceNode]:
    """
    Match detected fingerprints and features against all lineage entries.
    Returns a sorted list of InfluenceNode (highest confidence first).
    """
    lineages = _load_lineages()
    if not lineages:
        return []

    # Extract measurable values from fingerprints and features
    era_names = [m.name.lower() for m in (fingerprints.likely_era or [])]
    era_confidences = {m.name.lower(): m.confidence for m in (fingerprints.likely_era or [])}

    tempo = None
    groove_ms = None
    dynamic_range = None
    crest_factor = None

    if features:
        tempo = features.temporal.tempo_bpm if features.temporal else None
        groove_ms = features.temporal.groove_deviation_ms if features.temporal else None
        dynamic_range = features.dynamic.dynamic_range_db if features.dynamic else None
        crest_factor = features.dynamic.crest_factor_db if features.dynamic else None

    scored = []
    for lid, entry in lineages.items():
        score = _score_lineage_match(
            entry, era_names, era_confidences, tempo, groove_ms, dynamic_range
        )
        if score > 0.05:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_lineage_to_node(e, s) for s, e in scored[:6]]


def _score_lineage_match(
    entry: Dict,
    era_names: List[str],
    era_confidences: Dict[str, float],
    tempo: Optional[float],
    groove_ms: Optional[float],
    dynamic_range: Optional[float],
) -> float:
    """
    Score how well a lineage entry matches the detected signal characteristics.
    Returns a confidence score 0.0-1.0.

    Scoring breakdown:
    - Era name match: 0.40 (strongest signal — era from fingerprints.py is accurate)
    - Tempo match: 0.25
    - Groove deviation match: 0.20
    - Dynamic range match: 0.15
    """
    score = 0.0
    sc = entry.get("sonic_characteristics", {})

    # Era match
    entry_name_lower = entry.get("name", "").lower()
    entry_id_lower = entry.get("id", "").lower()
    for era_name, confidence in era_confidences.items():
        # Match by era decade label (e.g. "1970s" matches "funk" with era_range 1965-1982)
        era_range = entry.get("era_range", [0, 0])
        era_label_match = any(x in entry_name_lower for x in [era_name[:4]]) if era_name[:4].isdigit() else False
        decade_str = _era_to_decade_str(era_range)
        decade_match = era_name in decade_str.lower()
        if decade_match or era_label_match:
            score += 0.40 * confidence

    # Tempo match
    tempo_range = sc.get("tempo_bpm_range")
    if tempo and tempo_range and len(tempo_range) == 2:
        t_low, t_high = tempo_range
        if t_low <= tempo <= t_high:
            score += 0.25
        elif abs(tempo - t_low) < 20 or abs(tempo - t_high) < 20:
            score += 0.10  # Close but outside range

    # Groove deviation match
    groove_range = sc.get("groove_deviation_ms")
    if groove_ms is not None and groove_range and len(groove_range) == 2:
        g_low, g_high = groove_range
        if g_low <= groove_ms <= g_high:
            score += 0.20
        elif abs(groove_ms - g_low) < 10 or abs(groove_ms - g_high) < 10:
            score += 0.08

    # Dynamic range match
    dr_range = sc.get("dynamic_range_db")
    if dynamic_range and dr_range and len(dr_range) == 2:
        dr_low, dr_high = dr_range
        if dr_low <= dynamic_range <= dr_high:
            score += 0.15
        elif abs(dynamic_range - dr_low) < 3 or abs(dynamic_range - dr_high) < 3:
            score += 0.06

    return min(score, 1.0)


def _era_to_decade_str(era_range: List[int]) -> str:
    """Convert an era_range to a recognisable decade string for matching."""
    start = era_range[0] if era_range else 1900
    if start < 1950:
        return "pre-1950 1940s pre_1970"
    elif start < 1960:
        return "1950s early_rock 1950"
    elif start < 1970:
        return "1960s british 1960"
    elif start < 1975:
        return "1970s 1970"
    elif start < 1980:
        return "1970s disco 1970"
    elif start < 1990:
        return "1980s 1980"
    elif start < 2000:
        return "1990s 1990"
    elif start < 2010:
        return "2000s 2000"
    elif start < 2020:
        return "2010s 2010"
    else:
        return "2020s 2020"


def _build_lineage_chain(
    primary_node: InfluenceNode,
    all_lineages: Dict,
) -> List[InfluenceNode]:
    """
    Walk parent_lineages from primary_node back to roots.
    Returns the lineage chain with primary_node first, ancestors following.
    Maximum chain depth of 5 to avoid very deep recursion.
    """
    chain = [primary_node]
    seen = {primary_node.id}
    current_parents = primary_node.parent_lineages[:2]  # Limit to primary ancestors

    depth = 0
    while current_parents and depth < 4:
        parent_id = current_parents[0]
        if parent_id in seen or parent_id not in all_lineages:
            break
        entry = all_lineages[parent_id]
        parent_node = _lineage_to_node(entry, primary_node.confidence * (0.7 ** (depth + 1)))
        chain.append(parent_node)
        seen.add(parent_id)
        current_parents = entry.get("parent_lineages", [])[:2]
        depth += 1

    return chain


# ─────────────────────────────────────────────
# MECHANIC DETECTION
# ─────────────────────────────────────────────

def _detect_mechanics(
    fingerprints: FingerprintReport,
    features: Optional[SegmentFeatures],
    manufacturing_score: float = 0.5,
    authentic_emission_score: float = 0.5,
    lsii_score: float = 0.0,
) -> List[DetectedMechanic]:
    """
    Detect which of the 13 Kindfluence comprehension mechanics are present
    in the sonic signal.

    Each detection is classified as extractive, syntropic, or ambiguous
    based on the overall manufacturing vs authentic emission profile.

    The 13 mechanics and their sonic detection signatures:
      1. priming           — sub-bass + 120-140 BPM + quantised grid
      2. social_proof      — very high production quality + genre-conventional = authority through association
      3. anchoring         — highly stable harmonic loop, minimal development
      4. reciprocity       — generous dynamics, silence, space as gift
      5. mere_exposure     — high harmonic/structural repetition
      6. framing           — overall valence/arousal arc positions the listener
      7. narrative_transportation — distinct emotional arc, high complexity journey
      8. dopamine_loops    — tight quantisation + high onset density + consistent dynamics
      9. identity_association — very strong genre match, low creative residue
      10. loss_aversion    — urgent tempo + dynamic buildup without resolution (cannot fully detect from audio alone)
      11. cognitive_ease   — low harmonic complexity + high repetition
      12. authority_bias   — high production quality, low noise floor, precise mix = institutional authority signal
      13. bandwagon_effect — maximally genre-conventional = manufactured popularity markers
    """
    tempo = None
    groove_ms = None
    dynamic_range = None
    crest_factor = None
    tension_ratio = None
    harmonic_complexity = None

    if features:
        t = features.temporal
        d = features.dynamic
        h = features.harmonic
        if t:
            tempo = t.tempo_bpm
            groove_ms = t.groove_deviation_ms
        if d:
            dynamic_range = d.dynamic_range_db
            crest_factor = d.crest_factor_db
        if h:
            tension_ratio = h.tension_ratio
            harmonic_complexity = h.harmonic_complexity

    # Technique names for quick lookup
    technique_names = {m.name.lower() for m in (fingerprints.likely_techniques or [])}
    has_heavy_compression = "heavy compression" in technique_names
    has_quantised = "grid-quantised rhythm" in technique_names or "quantised" in technique_names
    has_human_performance = "human performance" in technique_names
    has_natural_dynamics = "natural dynamics" in technique_names

    detected = []

    # ── 1. PRIMING ────────────────────────────────────────────────────────────
    # Somatic priming: sub-bass + body-rate tempo + quantised = body entrainment pattern
    priming_evidence = []
    priming_confidence = 0.0
    if tempo and 115 <= tempo <= 145:
        priming_evidence.append(f"tempo {tempo:.0f} BPM — heart-rate entrainment zone")
        priming_confidence += 0.35
    if has_heavy_compression:
        priming_evidence.append("hypercompression maintains relentless intensity — no dynamic rest")
        priming_confidence += 0.25
    if has_quantised:
        priming_evidence.append("grid-quantised rhythm — machine-precise pulse, no bodily variation")
        priming_confidence += 0.20
    if dynamic_range and dynamic_range < 8:
        priming_evidence.append(f"narrow dynamic range ({dynamic_range:.1f}dB) — sustained body activation with no release")
        priming_confidence += 0.20

    if priming_confidence >= 0.40:
        direction = "extractive" if manufacturing_score > 0.5 else "ambiguous"
        detected.append(DetectedMechanic(
            name="priming",
            evidence=priming_evidence,
            direction=direction,
            confidence=min(priming_confidence, 0.95),
            syntropy_repair=_SYNTROPY_REPAIRS["priming"],
        ))

    # ── 2. ANCHORING ──────────────────────────────────────────────────────────
    # Repeating harmonic loop creates a reference anchor before development
    anchoring_evidence = []
    anchoring_confidence = 0.0
    if tension_ratio and tension_ratio < 0.4:
        anchoring_evidence.append(f"low harmonic tension ({tension_ratio:.2f}) — stable loop, minimal development")
        anchoring_confidence += 0.35
    if harmonic_complexity and harmonic_complexity < 0.3:
        anchoring_evidence.append(f"low harmonic complexity ({harmonic_complexity:.2f}) — repetitive harmonic structure")
        anchoring_confidence += 0.30
    if has_quantised and anchoring_confidence > 0.2:
        anchoring_evidence.append("quantised rhythm reinforces loop repetition")
        anchoring_confidence += 0.15

    if anchoring_confidence >= 0.40:
        direction = "extractive" if manufacturing_score > 0.6 else "ambiguous"
        detected.append(DetectedMechanic(
            name="anchoring",
            evidence=anchoring_evidence,
            direction=direction,
            confidence=min(anchoring_confidence, 0.90),
            syntropy_repair=_SYNTROPY_REPAIRS["anchoring"],
        ))

    # ── 3. RECIPROCITY ────────────────────────────────────────────────────────
    # Generous dynamics, preserved silence = giving something to the body
    reciprocity_evidence = []
    reciprocity_confidence = 0.0
    if has_natural_dynamics:
        reciprocity_evidence.append("preserved natural dynamics — production chose listener experience over loudness")
        reciprocity_confidence += 0.40
    if dynamic_range and dynamic_range > 12:
        reciprocity_evidence.append(f"generous dynamic range ({dynamic_range:.1f}dB) — space for the body to breathe")
        reciprocity_confidence += 0.30
    if has_human_performance:
        reciprocity_evidence.append("human performance markers — the body of another person is present in the signal")
        reciprocity_confidence += 0.20

    if reciprocity_confidence >= 0.40:
        # Reciprocity is inherently syntropic when genuine
        direction = "syntropic" if authentic_emission_score > 0.4 else "ambiguous"
        detected.append(DetectedMechanic(
            name="reciprocity",
            evidence=reciprocity_evidence,
            direction=direction,
            confidence=min(reciprocity_confidence, 0.90),
            syntropy_repair=_SYNTROPY_REPAIRS["reciprocity"],
        ))

    # ── 4. MERE EXPOSURE ──────────────────────────────────────────────────────
    # High structural/harmonic repetition builds familiarity through exposure
    mere_exposure_evidence = []
    mere_exposure_confidence = 0.0
    if harmonic_complexity and harmonic_complexity < 0.25:
        mere_exposure_evidence.append(f"very low harmonic complexity ({harmonic_complexity:.2f}) — familiarity built through repetition")
        mere_exposure_confidence += 0.40
    if tension_ratio and tension_ratio < 0.35:
        mere_exposure_evidence.append("stable harmonic territory with minimal variation — exposure loop")
        mere_exposure_confidence += 0.30
    if has_quantised and mere_exposure_confidence > 0.2:
        mere_exposure_evidence.append("machine-precise repetition amplifies the exposure effect")
        mere_exposure_confidence += 0.15

    if mere_exposure_confidence >= 0.45:
        direction = "extractive" if manufacturing_score > 0.6 else "ambiguous"
        detected.append(DetectedMechanic(
            name="mere_exposure",
            evidence=mere_exposure_evidence,
            direction=direction,
            confidence=min(mere_exposure_confidence, 0.85),
            syntropy_repair=_SYNTROPY_REPAIRS["mere_exposure"],
        ))

    # ── 5. NARRATIVE TRANSPORTATION ───────────────────────────────────────────
    # Significant LSII + complex emotional arc = the piece takes you on a journey
    narrative_evidence = []
    narrative_confidence = 0.0
    if lsii_score > 0.4:
        narrative_evidence.append(f"LSII {lsii_score:.3f} — distinct late-song shift, emotional journey present")
        narrative_confidence += 0.45
    if harmonic_complexity and harmonic_complexity > 0.5:
        narrative_evidence.append(f"high harmonic complexity ({harmonic_complexity:.2f}) — emotionally complex territory")
        narrative_confidence += 0.25

    if narrative_confidence >= 0.40:
        # Narrative transportation direction depends on LSII character
        # High LSII with authentic emission = syntropic journey
        direction = "syntropic" if (lsii_score > 0.5 and authentic_emission_score > 0.5) else "ambiguous"
        detected.append(DetectedMechanic(
            name="narrative_transportation",
            evidence=narrative_evidence,
            direction=direction,
            confidence=min(narrative_confidence, 0.90),
            syntropy_repair=_SYNTROPY_REPAIRS["narrative_transportation"],
        ))

    # ── 6. DOPAMINE LOOPS ─────────────────────────────────────────────────────
    # Tight quantisation + consistent dynamics + high energy = addictive loop
    dopamine_evidence = []
    dopamine_confidence = 0.0
    if has_quantised:
        dopamine_evidence.append("grid-quantised rhythm — removes organic variation that signals completion")
        dopamine_confidence += 0.30
    if has_heavy_compression:
        dopamine_evidence.append("hypercompression maintains uniform intensity — no natural energy arc to complete")
        dopamine_confidence += 0.30
    if tempo and 120 <= tempo <= 140 and has_quantised:
        dopamine_evidence.append(f"tempo {tempo:.0f} BPM + quantised pulse = physical engagement loop")
        dopamine_confidence += 0.25

    if dopamine_confidence >= 0.50:
        # Dopamine loops are always extractive in Kindfluence framework
        detected.append(DetectedMechanic(
            name="dopamine_loops",
            evidence=dopamine_evidence,
            direction="extractive",
            confidence=min(dopamine_confidence, 0.90),
            syntropy_repair=_SYNTROPY_REPAIRS["dopamine_loops"],
        ))

    # ── 7. IDENTITY ASSOCIATION ───────────────────────────────────────────────
    # Strong genre conventional production = tribal identity marker
    identity_evidence = []
    identity_confidence = 0.0
    era_matches = fingerprints.likely_era or []
    if era_matches and era_matches[0].confidence > 0.7:
        identity_evidence.append(f"strong era match to {era_matches[0].name} ({era_matches[0].confidence:.2f}) — genre-tribal production markers")
        identity_confidence += 0.35
    if len(fingerprints.manufacturing_markers) > 2:
        identity_evidence.append("multiple manufacturing markers — production optimised for genre recognition")
        identity_confidence += 0.30

    if identity_confidence >= 0.40:
        direction = "extractive" if manufacturing_score > 0.6 else "ambiguous"
        detected.append(DetectedMechanic(
            name="identity_association",
            evidence=identity_evidence,
            direction=direction,
            confidence=min(identity_confidence, 0.85),
            syntropy_repair=_SYNTROPY_REPAIRS["identity_association"],
        ))

    # ── 8. COGNITIVE EASE ─────────────────────────────────────────────────────
    # Simple harmonic + repetition = easy processing, feels more true
    cognitive_evidence = []
    cognitive_confidence = 0.0
    if harmonic_complexity and harmonic_complexity < 0.20:
        cognitive_evidence.append(f"very low harmonic complexity ({harmonic_complexity:.2f}) — information load kept minimal")
        cognitive_confidence += 0.40
    if has_quantised and harmonic_complexity and harmonic_complexity < 0.25:
        cognitive_evidence.append("quantised rhythm + simple harmony = maximum cognitive ease")
        cognitive_confidence += 0.30

    if cognitive_confidence >= 0.45:
        direction = "extractive" if manufacturing_score > 0.6 else "ambiguous"
        detected.append(DetectedMechanic(
            name="cognitive_ease",
            evidence=cognitive_evidence,
            direction=direction,
            confidence=min(cognitive_confidence, 0.85),
            syntropy_repair=_SYNTROPY_REPAIRS["cognitive_ease"],
        ))

    # ── 9. AUTHORITY BIAS ─────────────────────────────────────────────────────
    # High production quality = institutional authority signal
    authority_evidence = []
    authority_confidence = 0.0
    if crest_factor and 6 <= crest_factor <= 15 and len(fingerprints.manufacturing_markers) < 2:
        authority_evidence.append(f"professional production quality (crest factor {crest_factor:.1f}dB) — institutional authority signal")
        authority_confidence += 0.30
    if len(era_matches) > 0 and era_matches[0].confidence > 0.8:
        authority_evidence.append(f"precision era signature ({era_matches[0].name}) — professional production consistency")
        authority_confidence += 0.25

    if authority_confidence >= 0.40:
        direction = "extractive" if manufacturing_score > 0.55 else "ambiguous"
        detected.append(DetectedMechanic(
            name="authority_bias",
            evidence=authority_evidence,
            direction=direction,
            confidence=min(authority_confidence, 0.80),
            syntropy_repair=_SYNTROPY_REPAIRS["authority_bias"],
        ))

    # ── 10. BANDWAGON EFFECT ──────────────────────────────────────────────────
    # Maximally genre-conventional = manufactured popularity signal
    bandwagon_evidence = []
    bandwagon_confidence = 0.0
    if len(fingerprints.manufacturing_markers) >= 3:
        bandwagon_evidence.append(f"{len(fingerprints.manufacturing_markers)} manufacturing markers — production optimised for genre conformity")
        bandwagon_confidence += 0.40
    if has_heavy_compression and has_quantised:
        bandwagon_evidence.append("hypercompression + quantisation = commercial genre template markers")
        bandwagon_confidence += 0.30

    if bandwagon_confidence >= 0.45:
        direction = "extractive" if manufacturing_score > 0.6 else "ambiguous"
        detected.append(DetectedMechanic(
            name="bandwagon_effect",
            evidence=bandwagon_evidence,
            direction=direction,
            confidence=min(bandwagon_confidence, 0.85),
            syntropy_repair=_SYNTROPY_REPAIRS["bandwagon_effect"],
        ))

    return detected


# ─────────────────────────────────────────────
# DIRECTION CLASSIFICATION
# ─────────────────────────────────────────────

def _classify_mechanic_direction(
    detected: List[DetectedMechanic],
    authentic_emission_score: float,
    manufacturing_score: float,
) -> str:
    """
    Classify the overall mechanic direction across all detected mechanics.

    extractive: The mechanics are predominantly used to serve institutional goals
                at the expense of listener autonomy and wellbeing
    syntropic:  The mechanics are predominantly used to serve listener wellbeing
                and collective regeneration
    mixed:      Significant presence of both directions — transition or unresolved
    neutral:    No strong mechanics detected or evidence is too weak to classify
    """
    if not detected:
        return "neutral"

    extractive_count = sum(1 for m in detected if m.direction == "extractive")
    syntropic_count = sum(1 for m in detected if m.direction == "syntropic")
    total = len(detected)

    if total == 0:
        return "neutral"

    extractive_ratio = extractive_count / total
    syntropic_ratio = syntropic_count / total

    if extractive_ratio > 0.6:
        return "extractive"
    elif syntropic_ratio > 0.5:
        return "syntropic"
    elif extractive_count > 0 and syntropic_count > 0:
        return "mixed"
    else:
        # Ambiguous mechanics — use overall scores
        if manufacturing_score > 0.6:
            return "extractive"
        elif authentic_emission_score > 0.6:
            return "syntropic"
        return "neutral"


def _estimate_gradient_stage(
    mechanic_direction: str,
    lsii_score: float,
    authentic_emission_score: float,
) -> str:
    """
    Estimate where this piece sits on the Kindfluence MessagingGradient.

    INFILTRATE: Fully commercial/niche. Values (regenerative signals) absent.
    SEED:       Values beginning to emerge alongside commercial framing.
    EMERGE:     Values clearly present and developing.
    ACTIVATE:   Values are the primary axis; niche is contextual.
    TRANSFER:   Community voice beginning to take over from brand/commercial.
    LIBERATE:   Fully community-originated; institutional framing optional.

    These are applied to existing music, not content creation — so we read
    what gradient stage the piece's sonic values represent.
    """
    if mechanic_direction == "extractive":
        return "INFILTRATE"
    elif mechanic_direction == "syntropic":
        if authentic_emission_score > 0.75 and lsii_score > 0.4:
            return "ACTIVATE"
        elif authentic_emission_score > 0.60:
            return "EMERGE"
        return "SEED"
    elif mechanic_direction == "mixed":
        return "SEED"
    else:
        # Neutral / ambiguous
        if lsii_score > 0.5:
            return "SEED"
        return "INFILTRATE"


# ─────────────────────────────────────────────
# NARRATIVE GENERATION
# ─────────────────────────────────────────────

def _build_lineage_narrative(
    primary_chain: List[InfluenceNode],
    secondary: List[List[InfluenceNode]],
    innovation_points: List[str],
    confluence_points: List[str],
) -> str:
    """
    Build a plain-language lineage narrative.
    The elder's account of where this sound came from.
    """
    if not primary_chain:
        return "Lineage could not be established from the available signal data."

    primary = primary_chain[0]
    lines = []

    # Primary lineage
    if len(primary_chain) == 1:
        lines.append(
            f"The dominant sonic signature traces to {primary.name}. "
            f"{primary.cultural_context[:120].rstrip('.')}."
        )
    else:
        ancestors = " → ".join(n.name for n in reversed(primary_chain))
        lines.append(
            f"Primary lineage chain: {ancestors}. "
            f"The signal descends from {primary_chain[-1].name}, "
            f"arriving through {', '.join(n.name for n in primary_chain[1:-1])} "
            f"into the form detected here." if len(primary_chain) > 2
            else f"The signal descends from {primary_chain[-1].name} into {primary_chain[0].name}."
        )

    # Secondary lineages
    if secondary:
        secondary_names = [s[0].name for s in secondary[:2] if s]
        lines.append(f"Secondary threads traceable to: {', '.join(secondary_names)}.")

    # Confluence points
    if confluence_points:
        lines.append(f"Convergence: {'; '.join(confluence_points[:2])}.")

    # Innovation points
    if innovation_points:
        lines.append(
            f"Departures from lineage: {'; '.join(innovation_points[:2])}. "
            "These are the moments where inherited form meets individual choice."
        )

    # KindPath note on primary lineage
    if primary.kindpath_notes:
        # First sentence only
        kp_note = primary.kindpath_notes.split('.')[0] + '.'
        lines.append(kp_note)

    return " ".join(lines)


def _build_mechanic_summary(
    detected: List[DetectedMechanic],
    direction: str,
    gradient_stage: str,
) -> str:
    """
    Build a plain-language summary of detected mechanics and their direction.
    """
    if not detected:
        return "No strong comprehension mechanics detected in the signal pattern."

    mechanic_names = [m.name.replace("_", " ") for m in detected]
    extractive = [m for m in detected if m.direction == "extractive"]
    syntropic = [m for m in detected if m.direction == "syntropic"]

    lines = []
    if len(mechanic_names) == 1:
        lines.append(f"One comprehension mechanic detected: {mechanic_names[0]}.")
    else:
        lines.append(f"Comprehension mechanics present: {', '.join(mechanic_names)}.")

    if direction == "extractive":
        lines.append(
            "The overall direction is extractive — these mechanics are deployed "
            "in patterns consistent with institutional goals rather than listener wellbeing."
        )
    elif direction == "syntropic":
        lines.append(
            "The overall direction is syntropic — these mechanics operate in patterns "
            "consistent with listener wellbeing and collective regeneration."
        )
    elif direction == "mixed":
        lines.append(
            "Direction is mixed — some mechanics serve the listener, others serve "
            "institutional interests. This is the transition zone."
        )

    if gradient_stage in ("EMERGE", "ACTIVATE", "TRANSFER", "LIBERATE"):
        lines.append(
            f"Gradient stage estimate: {gradient_stage} — "
            "values are significantly present in the sonic field."
        )

    return " ".join(lines)


# ─────────────────────────────────────────────
# INNOVATION AND CONFLUENCE DETECTION
# ─────────────────────────────────────────────

def _detect_innovation_points(
    fingerprints: FingerprintReport,
    features: Optional[SegmentFeatures],
    primary_chain: List[InfluenceNode],
) -> List[str]:
    """
    Identify where this piece departs from its detected lineage baseline.
    Departures that are internally consistent are authentic creative innovation.
    """
    points = []

    if not primary_chain:
        return points

    primary = primary_chain[0]
    sc = primary.sonic_characteristics

    # Check for tempo departure
    tempo_range = sc.get("tempo_bpm_range")
    if features and features.temporal and tempo_range:
        tempo = features.temporal.tempo_bpm
        t_low, t_high = tempo_range
        if tempo < t_low - 15:
            points.append(f"tempo significantly slower than {primary.name} baseline ({tempo:.0f} vs {t_low}-{t_high} BPM) — unusual for this lineage")
        elif tempo > t_high + 15:
            points.append(f"tempo significantly faster than {primary.name} baseline ({tempo:.0f} vs {t_low}-{t_high} BPM)")

    # Check for dynamic departure
    dr_range = sc.get("dynamic_range_db")
    if features and features.dynamic and dr_range:
        dr = features.dynamic.dynamic_range_db
        dr_low, dr_high = dr_range
        if dr > dr_high + 4:
            points.append(f"dynamic range ({dr:.1f}dB) exceeds {primary.name} typical range — more space than the tradition normally allows")
        elif dr < dr_low - 4:
            points.append(f"dynamic range ({dr:.1f}dB) below {primary.name} typical range — more compressed than the tradition")

    # Check for authenticity markers (positive innovation)
    if fingerprints.authenticity_markers:
        for marker in fingerprints.authenticity_markers[:2]:
            points.append(f"authenticity marker: {marker}")

    return points


def _detect_confluence_points(
    primary_chain: List[InfluenceNode],
    secondary_lineages: List[List[InfluenceNode]],
) -> List[str]:
    """
    Identify where multiple lineage threads converge in the detected signal.
    """
    points = []

    if not secondary_lineages:
        return points

    primary_name = primary_chain[0].name if primary_chain else "primary tradition"
    for secondary in secondary_lineages[:2]:
        if secondary:
            secondary_name = secondary[0].name
            # Check if they share common ancestors
            primary_ancestors = {n.id for n in primary_chain}
            secondary_ancestors = {n.id for n in secondary}
            shared = primary_ancestors & secondary_ancestors
            if shared:
                shared_names = [n.name for n in primary_chain if n.id in shared]
                points.append(
                    f"Convergence of {primary_name} and {secondary_name} "
                    f"(shared origin: {shared_names[0]})"
                )
            else:
                points.append(f"Crossover between {primary_name} and {secondary_name} lineages")

    return points


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def map_influence_chain(
    fingerprints: FingerprintReport,
    features: Optional[SegmentFeatures] = None,
    seedbank_baseline: Optional[dict] = None,
    manufacturing_score: float = 0.5,
    authentic_emission_score: float = 0.5,
    lsii_score: float = 0.0,
) -> InfluenceChain:
    """
    Map the traceable sonic lineage of a piece with Kindfluence mechanic analysis.

    Args:
        fingerprints: FingerprintReport from core.fingerprints
        features: SegmentFeatures (first quarter or representative segment)
        seedbank_baseline: Optional dict of era-specific baselines from seedbank
        manufacturing_score: 0-1 from psychosomatics (used for direction classification)
        authentic_emission_score: 0-1 from psychosomatics
        lsii_score: LSII score from divergence analysis

    Returns:
        InfluenceChain with lineage + mechanic analysis
    """
    lineages_db = _load_lineages()

    # ── Match lineages ────────────────────────────────────────────────────────
    matched_nodes = _match_lineages(fingerprints, features)

    if not matched_nodes:
        # Graceful fallback: return neutral chain
        return InfluenceChain(
            primary_lineage=[],
            secondary_lineages=[],
            innovation_points=[],
            confluence_points=[],
            narrative="Lineage could not be established from available signal data.",
            detected_mechanics=[],
            mechanic_direction="neutral",
            gradient_stage_estimate="INFILTRATE",
            syntropy_repair_vectors=[],
            mechanic_summary="Insufficient signal data for mechanic detection.",
        )

    # Primary lineage: top match + its ancestors
    primary_node = matched_nodes[0]
    primary_chain = _build_lineage_chain(primary_node, lineages_db)

    # Secondary lineages: next 2-3 matches, each with their chains
    secondary_lineages = []
    for node in matched_nodes[1:3]:
        if node.confidence > 0.2:
            chain = _build_lineage_chain(node, lineages_db)
            secondary_lineages.append(chain)

    # ── Innovation + confluence ────────────────────────────────────────────────
    innovation_points = _detect_innovation_points(fingerprints, features, primary_chain)
    confluence_points = _detect_confluence_points(primary_chain, secondary_lineages)

    # ── Mechanic detection (Kindfluence bridge) ──────────────────────────────
    detected_mechanics = _detect_mechanics(
        fingerprints,
        features,
        manufacturing_score=manufacturing_score,
        authentic_emission_score=authentic_emission_score,
        lsii_score=lsii_score,
    )

    mechanic_direction = _classify_mechanic_direction(
        detected_mechanics, authentic_emission_score, manufacturing_score
    )

    gradient_stage = _estimate_gradient_stage(
        mechanic_direction, lsii_score, authentic_emission_score
    )

    # ── Syntropy repair vectors ───────────────────────────────────────────────
    # Generate for every extractive or ambiguous mechanic
    repair_vectors = []
    for mechanic in detected_mechanics:
        if mechanic.direction in ("extractive", "ambiguous"):
            if mechanic.syntropy_repair and mechanic.syntropy_repair not in repair_vectors:
                repair_vectors.append(
                    f"[{mechanic.name.upper()}] {mechanic.syntropy_repair}"
                )

    # ── Narratives ─────────────────────────────────────────────────────────────
    narrative = _build_lineage_narrative(
        primary_chain, secondary_lineages, innovation_points, confluence_points
    )
    mechanic_summary = _build_mechanic_summary(
        detected_mechanics, mechanic_direction, gradient_stage
    )

    return InfluenceChain(
        primary_lineage=primary_chain,
        secondary_lineages=secondary_lineages,
        innovation_points=innovation_points,
        confluence_points=confluence_points,
        narrative=narrative,
        detected_mechanics=detected_mechanics,
        mechanic_direction=mechanic_direction,
        gradient_stage_estimate=gradient_stage,
        syntropy_repair_vectors=repair_vectors,
        mechanic_summary=mechanic_summary,
    )

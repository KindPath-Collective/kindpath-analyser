"""
KindPath Analyser :: KindSearch

The search layer over the seedbank and influence chain system.
Wraps seedbank/query.py search, lineage traversal, and mechanic comparison
into a single accessible interface.

KindSearch answers two kinds of questions:
  "Find me music like this" — sonic similarity, lineage proximity
  "Is this similar to that?" — direct comparison across all axes

The seedbank is the elder's memory. KindSearch is how you ask it questions.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────
# SEEDBANK SEARCH
# ─────────────────────────────────────────────

def search(
    lsii_min: float = None,
    lsii_max: float = None,
    era: str = None,
    authentic_emission_min: float = None,
    manufacturing_max: float = None,
    mechanic_direction: str = None,
    gradient_stage: str = None,
    tags: List[str] = None,
    text: str = None,
    limit: int = 20,
) -> List[dict]:
    """
    Search the seedbank by any combination of parameters.
    Returns list of matching records ordered by LSII score (descending).

    Additional parameters beyond the base seedbank query:
        mechanic_direction: 'extractive', 'syntropic', 'mixed', 'neutral'
        gradient_stage: 'INFILTRATE', 'SEED', 'EMERGE', 'ACTIVATE', 'TRANSFER', 'LIBERATE'
    """
    try:
        from seedbank.query import search as seedbank_search
        results = seedbank_search(
            lsii_min=lsii_min,
            lsii_max=lsii_max,
            era=era,
            authentic_emission_min=authentic_emission_min,
            manufacturing_max=manufacturing_max,
            tags=tags,
            text=text,
            limit=limit * 2,  # Fetch extra for filtering by mechanic
        )

        # Filter by mechanic_direction if requested
        if mechanic_direction:
            results = [
                r for r in results
                if _get_mechanic_direction(r) == mechanic_direction
            ]

        # Filter by gradient_stage if requested
        if gradient_stage:
            results = [
                r for r in results
                if _get_gradient_stage(r) == gradient_stage
            ]

        return results[:limit]

    except ImportError:
        # Seedbank not yet initialised — return empty
        return []


def find_similar(
    profile: dict,
    limit: int = 10,
    match_lineage: bool = True,
    match_mechanics: bool = True,
) -> List[dict]:
    """
    Find seedbank records similar to the given analysis profile.

    Similarity is measured across:
      - LSII score (±0.2 window)
      - Authentic emission score (±0.2 window)
      - Manufacturing score (±0.2 window)
      - Mechanic direction (exact match if match_mechanics=True)

    If match_lineage=True, also filters by era fingerprint match.
    """
    lsii = _extract_lsii(profile)
    auth = _extract_authentic_emission(profile)
    mfg = _extract_manufacturing(profile)
    era = _extract_era(profile)
    direction = profile.get('influence_chain', {}).get('mechanic_direction')

    results = search(
        lsii_min=max(0, lsii - 0.2) if lsii is not None else None,
        lsii_max=min(1, lsii + 0.2) if lsii is not None else None,
        authentic_emission_min=max(0, auth - 0.2) if auth is not None else None,
        manufacturing_max=min(1, mfg + 0.2) if mfg is not None else None,
        era=era if match_lineage else None,
        mechanic_direction=direction if match_mechanics and direction else None,
        limit=limit,
    )
    return results


def compare_to_seedbank(profile: dict, seedbank_id: str) -> dict:
    """
    Direct comparison between the current profile and a seedbank record.
    Returns a delta dict with interpretation for each axis.
    """
    try:
        from seedbank.query import compare as seedbank_compare
        return seedbank_compare(profile, seedbank_id)
    except (ImportError, Exception) as e:
        return _manual_compare(profile, seedbank_id, str(e))


def get_lineage_context(lineage_id: str) -> Optional[dict]:
    """
    Retrieve lineage information for a given lineage ID.
    Returns the full lineage entry from lineages.json.
    """
    lineages_path = os.path.join(os.path.dirname(__file__), '..', 'fingerprints', 'lineages.json')
    if not os.path.exists(lineages_path):
        return None
    with open(lineages_path) as f:
        data = json.load(f)
    lineage_map = {e['id']: e for e in data.get('lineages', [])}
    return lineage_map.get(lineage_id)


def compare_mechanic_direction(profile_a: dict, profile_b: dict) -> dict:
    """
    Compare the mechanic direction of two profiles.
    Returns a comparative reading with Kindfluence interpretation.
    """
    chain_a = profile_a.get('influence_chain', {})
    chain_b = profile_b.get('influence_chain', {})

    direction_a = chain_a.get('mechanic_direction', 'neutral')
    direction_b = chain_b.get('mechanic_direction', 'neutral')
    mechanics_a = {m['name'] for m in chain_a.get('detected_mechanics', [])}
    mechanics_b = {m['name'] for m in chain_b.get('detected_mechanics', [])}
    shared_mechanics = mechanics_a & mechanics_b
    unique_to_a = mechanics_a - mechanics_b
    unique_to_b = mechanics_b - mechanics_a

    interpretation = _interpret_direction_comparison(direction_a, direction_b)

    return {
        'direction_a': direction_a,
        'direction_b': direction_b,
        'shared_mechanics': sorted(shared_mechanics),
        'unique_to_a': sorted(unique_to_a),
        'unique_to_b': sorted(unique_to_b),
        'gradient_a': chain_a.get('gradient_stage_estimate', ''),
        'gradient_b': chain_b.get('gradient_stage_estimate', ''),
        'interpretation': interpretation,
    }


def get_most_authentic(limit: int = 10) -> List[dict]:
    """Return top N seedbank records by authentic_emission_score."""
    try:
        from seedbank.query import get_most_authentic as sb_auth
        return sb_auth(limit=limit)
    except ImportError:
        return []


def get_highest_lsii(limit: int = 10) -> List[dict]:
    """Return top N seedbank records by LSII score."""
    try:
        from seedbank.query import get_highest_lsii as sb_lsii
        return sb_lsii(limit=limit)
    except ImportError:
        return []


def get_syntropy_leaders(limit: int = 10) -> List[dict]:
    """
    Return the records with the strongest syntropic mechanic direction.
    These are the pieces where influence mechanics serve community liberation.
    """
    return search(
        mechanic_direction='syntropic',
        authentic_emission_min=0.5,
        limit=limit,
    )


def get_influence_summary(profile: dict) -> str:
    """
    Generate a plain-language summary of this profile's influence chain
    and mechanic direction for quick reference.
    """
    chain = profile.get('influence_chain', {})
    if not chain:
        return "No influence chain data available."

    lines = []

    # Lineage narrative
    narrative = chain.get('narrative', '')
    if narrative:
        lines.append(narrative[:300].rstrip('.') + '.')

    # Mechanic summary
    mechanic_summary = chain.get('mechanic_summary', '')
    if mechanic_summary:
        lines.append(mechanic_summary)

    # Gradient
    gradient = chain.get('gradient_stage_estimate', '')
    if gradient:
        stage_descriptions = {
            'INFILTRATE': "This piece sits fully within commercial/conventional framing.",
            'SEED': "Values are beginning to emerge alongside commercial framing.",
            'EMERGE': "Values are clearly present and developing.",
            'ACTIVATE': "Values are the primary axis; the niche is contextual.",
            'TRANSFER': "Community voice is starting to take over.",
            'LIBERATE': "Fully community-originated — institutional framing optional.",
        }
        lines.append(f"Gradient: {gradient}. {stage_descriptions.get(gradient, '')}")

    # Repair vectors count
    repair_count = len(chain.get('syntropy_repair_vectors', []))
    if repair_count > 0:
        lines.append(f"{repair_count} syntropy repair vector{'s' if repair_count > 1 else ''} identified.")

    return ' '.join(lines) if lines else "Influence analysis incomplete."


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _extract_lsii(profile: dict) -> Optional[float]:
    return profile.get('lsii', {}).get('score')


def _extract_authentic_emission(profile: dict) -> Optional[float]:
    return profile.get('psychosomatic', {}).get('authentic_emission_score')


def _extract_manufacturing(profile: dict) -> Optional[float]:
    return profile.get('psychosomatic', {}).get('manufacturing_score')


def _extract_era(profile: dict) -> Optional[str]:
    era_matches = profile.get('fingerprints', {}).get('era_matches', [])
    if era_matches:
        return era_matches[0].get('name')
    return None


def _get_mechanic_direction(record: Any) -> str:
    """Extract mechanic_direction from a seedbank record."""
    if isinstance(record, dict):
        return record.get('mechanic_direction') or record.get('influence_chain', {}).get('mechanic_direction', 'neutral')
    return getattr(record, 'mechanic_direction', 'neutral')


def _get_gradient_stage(record: Any) -> str:
    """Extract gradient_stage_estimate from a seedbank record."""
    if isinstance(record, dict):
        return record.get('gradient_stage_estimate') or record.get('influence_chain', {}).get('gradient_stage_estimate', '')
    return getattr(record, 'gradient_stage_estimate', '')


def _manual_compare(profile: dict, seedbank_id: str, error: str) -> dict:
    """Fallback comparison when seedbank module is unavailable."""
    return {
        'error': f"Seedbank compare unavailable ({error}). Manual reference: {seedbank_id}",
        'profile_lsii': _extract_lsii(profile),
        'profile_authentic_emission': _extract_authentic_emission(profile),
        'profile_manufacturing': _extract_manufacturing(profile),
        'profile_mechanic_direction': profile.get('influence_chain', {}).get('mechanic_direction', 'neutral'),
    }


def _interpret_direction_comparison(direction_a: str, direction_b: str) -> str:
    """Generate a plain-language interpretation of two mechanic directions."""
    if direction_a == direction_b:
        if direction_a == 'extractive':
            return "Both pieces employ extractive mechanic patterns. The comparison shows consistent commercial/institutional framing."
        elif direction_a == 'syntropic':
            return "Both pieces employ syntropic mechanic patterns. The comparison shows consistent community-serving framing."
        elif direction_a == 'mixed':
            return "Both pieces show mixed mechanic direction — transition zone, values present but not dominant."
        return f"Both pieces share the same direction: {direction_a}."
    elif 'extractive' in (direction_a, direction_b) and 'syntropic' in (direction_a, direction_b):
        return "These pieces sit at opposite ends of the field: one employs extractive mechanics, the other syntropic. A useful contrast for study."
    elif direction_a == 'neutral' or direction_b == 'neutral':
        non_neutral = direction_a if direction_b == 'neutral' else direction_b
        return f"One piece shows {non_neutral} mechanic patterns; the other is too ambiguous to classify. Insufficient signal for strong comparison."
    return f"Direction shift from {direction_a} to {direction_b}."

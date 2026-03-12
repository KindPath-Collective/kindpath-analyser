"""
KindPath Analyser :: Corpus Temporal Analyser

The social EEG.

Collective creative output changes measurably over time under cultural pressure:
dynamic range narrows (loudness war), timing becomes more quantised (grid culture),
harmonic complexity flattens (cognitive capture), and LSII scores rise
(late-song inversions accumulate as creators encode what they cannot say openly).

The progressive absence of authentic signal IS the signal.
This module reads it as a time series — and detects what it means when
a culture's creative output starts narrowing in specific, coordinated ways.

The SOS signature is not a scream. It is a statistical narrowing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorpusTemporalProfile:
    """
    The aggregate creative health of a corpus over time.
    Each list entry is one time period (e.g. a 5-year bucket).
    All trend lists are parallel — index N represents the same time period across all fields.
    """

    # ── Time axis ────────────────────────────────────────────────────────────
    periods: List[str] = field(default_factory=list)              # e.g. ["1970-1974", "1975-1979"]
    period_boundaries: List[Tuple[int, int]] = field(default_factory=list)  # (start_year, end_year)
    record_counts: List[int] = field(default_factory=list)        # How many records per period

    # ── Health metrics over time ──────────────────────────────────────────────
    lsii_trend: List[float] = field(default_factory=list)
    authentic_emission_trend: List[float] = field(default_factory=list)
    manufacturing_trend: List[float] = field(default_factory=list)
    creative_residue_trend: List[float] = field(default_factory=list)
    dynamic_range_trend: List[float] = field(default_factory=list)

    # ── Emotional arc ─────────────────────────────────────────────────────────
    valence_trend: List[float] = field(default_factory=list)
    arousal_trend: List[float] = field(default_factory=list)
    tension_trend: List[float] = field(default_factory=list)
    complexity_trend: List[float] = field(default_factory=list)

    # ── Technical trends ──────────────────────────────────────────────────────
    groove_deviation_trend: List[float] = field(default_factory=list)
    compression_trend: List[float] = field(default_factory=list)
    harmonic_complexity_trend: List[float] = field(default_factory=list)

    # ── Derived composite indices ──────────────────────────────────────────────
    creative_freedom_index: List[float] = field(default_factory=list)
    cultural_pressure_index: List[float] = field(default_factory=list)

    # ── SOS detection ─────────────────────────────────────────────────────────
    sos_periods: List[str] = field(default_factory=list)
    sos_confidence: List[float] = field(default_factory=list)

    # ── Predictions ───────────────────────────────────────────────────────────
    next_period_prediction: Dict[str, Any] = field(default_factory=dict)
    prediction_confidence: float = 0.0
    prediction_reasoning: str = ""

    # ── Narrative ─────────────────────────────────────────────────────────────
    trend_summary: str = ""
    inflection_points: List[Dict[str, str]] = field(default_factory=list)


def analyse_corpus(
    profiles: List[Dict],
    year_field: str = "year",
    period_size: int = 5,
) -> CorpusTemporalProfile:
    """
    Compute temporal trends across a corpus of dated analysis profiles.

    profiles: list of analysis dicts (the JSON output from generate_json_report()).
              Each must contain a `year_field` key with an integer year.
    year_field: key name in each profile dict for the year.
    period_size: years per time bucket (default 5).

    Returns a CorpusTemporalProfile. Profiles without a valid year are skipped.
    Returns an empty profile (with a summary note) if fewer than 2 profiles are provided.
    """
    result = CorpusTemporalProfile()

    # Filter to profiles that have a valid year
    dated = []
    for p in profiles:
        try:
            year = int(p.get(year_field, 0))
            if year > 0:
                dated.append((year, p))
        except (TypeError, ValueError):
            continue

    if len(dated) < 2:
        result.trend_summary = "Insufficient dated profiles for trend analysis."
        return result

    dated.sort(key=lambda x: x[0])
    min_year = dated[0][0]
    max_year = dated[-1][0]

    # Build time buckets
    buckets: Dict[str, List[Dict]] = {}
    bucket_bounds: Dict[str, Tuple[int, int]] = {}

    start = (min_year // period_size) * period_size
    end = ((max_year // period_size) + 1) * period_size

    for year_start in range(start, end, period_size):
        year_end = year_start + period_size - 1
        label = f"{year_start}-{year_end}"
        buckets[label] = []
        bucket_bounds[label] = (year_start, year_end)

    for year, profile in dated:
        year_start = (year // period_size) * period_size
        year_end = year_start + period_size - 1
        label = f"{year_start}-{year_end}"
        buckets[label].append(profile)

    # Only keep buckets with at least one record
    non_empty = [(label, bucket_bounds[label], buckets[label])
                 for label in buckets if buckets[label]]

    if len(non_empty) < 2:
        result.trend_summary = "All records fall within a single time period."
        return result

    # Extract trend values per period
    for label, bounds, period_profiles in non_empty:
        result.periods.append(label)
        result.period_boundaries.append(bounds)
        result.record_counts.append(len(period_profiles))

        result.lsii_trend.append(_mean_field(period_profiles, "lsii_score"))
        result.authentic_emission_trend.append(_mean_field(period_profiles, "authentic_emission_score"))
        result.manufacturing_trend.append(_mean_field(period_profiles, "manufacturing_score"))
        result.creative_residue_trend.append(_mean_field(period_profiles, "creative_residue"))
        result.dynamic_range_trend.append(_mean_field(period_profiles, "dynamic_range_db"))
        result.valence_trend.append(_mean_field(period_profiles, "valence"))
        result.arousal_trend.append(_mean_field(period_profiles, "arousal"))
        result.tension_trend.append(_mean_field(period_profiles, "tension"))
        result.complexity_trend.append(_mean_field(period_profiles, "complexity"))
        result.groove_deviation_trend.append(_mean_field(period_profiles, "groove_deviation_ms"))
        result.compression_trend.append(_mean_field(period_profiles, "crest_factor_db"))
        result.harmonic_complexity_trend.append(_mean_field(period_profiles, "harmonic_complexity"))

    # Composite indices
    result.creative_freedom_index = _compute_creative_freedom_index(
        result.authentic_emission_trend,
        result.dynamic_range_trend,
        result.groove_deviation_trend,
    )
    result.cultural_pressure_index = _compute_cultural_pressure_index(
        result.manufacturing_trend,
        result.compression_trend,
    )

    # SOS detection
    sos_periods, sos_confidence = _detect_sos_across_periods(
        result.lsii_trend,
        result.dynamic_range_trend,
        result.groove_deviation_trend,
        result.creative_residue_trend,
        result.harmonic_complexity_trend,
        result.manufacturing_trend,
        result.periods,
    )
    result.sos_periods = sos_periods
    result.sos_confidence = sos_confidence

    # Inflection points
    result.inflection_points = _find_inflection_points(
        result.periods,
        result.creative_freedom_index,
        result.cultural_pressure_index,
    )

    # Prediction
    result.next_period_prediction, result.prediction_confidence, result.prediction_reasoning = (
        _predict_next_period(result)
    )

    # Trend summary
    result.trend_summary = _build_trend_summary(result)

    return result


# ── Field extraction helpers ──────────────────────────────────────────────────

def _mean_field(profiles: List[Dict], field_path: str, default: float = 0.0) -> float:
    """
    Extract a numeric field from a list of profile dicts and return the mean.
    Supports dot-notation for nested fields: "psychosomatic.valence"
    Returns default if field is absent in all profiles.
    """
    values = []
    for p in profiles:
        val = _deep_get(p, field_path)
        if val is not None:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
    return float(np.mean(values)) if values else default


def _deep_get(d: Dict, path: str, default=None):
    """Navigate a dot-separated path through nested dicts."""
    parts = path.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict):
            return default
        current = current.get(part, default)
        if current is None:
            return default
    return current


# ── Composite index computation ───────────────────────────────────────────────

def _compute_creative_freedom_index(
    authentic_emission: List[float],
    dynamic_range: List[float],
    groove_deviation: List[float],
) -> List[float]:
    """
    Creative freedom: composite of authentic emission + dynamic space + human timing.
    All three components normalised to 0-1, then averaged.
    High = creative freedom; low = constraint.
    """
    if not authentic_emission:
        return []

    n = len(authentic_emission)

    # Normalise dynamic range: 0dB → 0, 20dB+ → 1
    dr_norm = [min(max(dr / 20.0, 0.0), 1.0) for dr in dynamic_range]

    # Normalise groove deviation: 0ms → 0, 30ms+ → 1
    gd_norm = [min(max(gd / 30.0, 0.0), 1.0) for gd in groove_deviation]

    # Authentic emission already 0-1
    result = []
    for i in range(n):
        ae = authentic_emission[i] if i < len(authentic_emission) else 0.0
        dr = dr_norm[i] if i < len(dr_norm) else 0.0
        gd = gd_norm[i] if i < len(gd_norm) else 0.0
        result.append((ae + dr + gd) / 3.0)
    return result


def _compute_cultural_pressure_index(
    manufacturing: List[float],
    compression: List[float],
) -> List[float]:
    """
    Cultural pressure: high manufacturing + high compression (low crest factor) = high pressure.
    Crest factor inverted: lower crest factor = more compression = higher pressure.
    """
    if not manufacturing:
        return []

    n = len(manufacturing)
    # Crest factor: 20dB+ → natural, < 6dB → hypercompressed
    # Normalised: 20dB=0 pressure, 0dB=1 pressure
    compression_norm = [max(0.0, min(1.0, 1.0 - cf / 20.0)) for cf in compression]

    result = []
    for i in range(n):
        mfg = manufacturing[i] if i < len(manufacturing) else 0.0
        cmp = compression_norm[i] if i < len(compression_norm) else 0.0
        result.append((mfg + cmp) / 2.0)
    return result


# ── SOS detection ─────────────────────────────────────────────────────────────

def _detect_sos_across_periods(
    lsii_trend: List[float],
    dynamic_range_trend: List[float],
    groove_deviation_trend: List[float],
    creative_residue_trend: List[float],
    harmonic_complexity_trend: List[float],
    manufacturing_trend: List[float],
    periods: List[str],
) -> Tuple[List[str], List[float]]:
    """
    Detect the SOS signature: a convergence of indicators that collectively reveal
    cultural creative constraint. Checks each 3-period window for the pattern.

    HIGH CONFIDENCE SOS (all markers trending together):
    - Dynamic range declining
    - Groove deviation declining (more quantised)
    - Creative residue declining
    - LSII increasing (more late-song inversions)
    - Harmonic complexity declining
    - Manufacturing score increasing

    The SOS is not a scream. It is a narrowing.
    """
    if len(periods) < 3:
        return [], []

    sos_periods = []
    sos_confidences = []

    for i in range(1, len(periods)):
        is_sos, confidence, _ = _detect_sos_signature_at_period(
            i,
            lsii_trend,
            dynamic_range_trend,
            groove_deviation_trend,
            creative_residue_trend,
            harmonic_complexity_trend,
            manufacturing_trend,
        )
        if is_sos:
            sos_periods.append(periods[i])
            sos_confidences.append(confidence)

    return sos_periods, sos_confidences


def _detect_sos_signature_at_period(
    idx: int,
    lsii: List[float],
    dynamic_range: List[float],
    groove: List[float],
    residue: List[float],
    harmonic: List[float],
    manufacturing: List[float],
) -> Tuple[bool, float, str]:
    """
    Check whether period `idx` shows the SOS signature relative to period `idx-1`.

    Returns (is_sos, confidence, description).
    Confidence increases with the number of indicators converging.
    """
    if idx == 0 or idx >= len(lsii):
        return False, 0.0, ""

    indicators_met = 0
    total_indicators = 6
    evidence = []

    # 1. Dynamic range declining
    if _is_declining(dynamic_range, idx):
        indicators_met += 1
        evidence.append("dynamic range declining")

    # 2. Groove deviation declining (more quantised)
    if _is_declining(groove, idx):
        indicators_met += 1
        evidence.append("groove deviation declining")

    # 3. Creative residue declining
    if _is_declining(residue, idx):
        indicators_met += 1
        evidence.append("creative residue declining")

    # 4. LSII increasing
    if _is_rising(lsii, idx):
        indicators_met += 1
        evidence.append("LSII rising")

    # 5. Harmonic complexity declining
    if _is_declining(harmonic, idx):
        indicators_met += 1
        evidence.append("harmonic complexity declining")

    # 6. Manufacturing score increasing
    if _is_rising(manufacturing, idx):
        indicators_met += 1
        evidence.append("manufacturing score rising")

    confidence = indicators_met / total_indicators

    # SOS threshold: at least 4 of 6 indicators must converge
    is_sos = indicators_met >= 4
    description = "; ".join(evidence) if evidence else ""
    return is_sos, confidence, description


def _is_declining(values: List[float], idx: int, threshold: float = 0.0) -> bool:
    """Returns True if value at `idx` is lower than at `idx-1`."""
    if idx == 0 or idx >= len(values) or idx - 1 >= len(values):
        return False
    return values[idx] < values[idx - 1] - threshold


def _is_rising(values: List[float], idx: int, threshold: float = 0.0) -> bool:
    """Returns True if value at `idx` is higher than at `idx-1`."""
    if idx == 0 or idx >= len(values) or idx - 1 >= len(values):
        return False
    return values[idx] > values[idx - 1] + threshold


# ── Inflection points ─────────────────────────────────────────────────────────

def _find_inflection_points(
    periods: List[str],
    creative_freedom: List[float],
    cultural_pressure: List[float],
) -> List[Dict[str, str]]:
    """
    Find notable shifts in the creative freedom / cultural pressure indices.
    Returns a list of dicts: {period, direction, magnitude, description}.
    """
    inflections = []
    if len(creative_freedom) < 3:
        return inflections

    for i in range(1, len(periods) - 1):
        # Check for local minimum (freedom) or maximum (pressure)
        cf_delta = creative_freedom[i] - creative_freedom[i - 1]
        cp_delta = cultural_pressure[i] - cultural_pressure[i - 1]

        # Significant creative decline
        if cf_delta < -0.15:
            inflections.append({
                "period": periods[i],
                "type": "creative_decline",
                "magnitude": f"{abs(cf_delta):.2f}",
                "description": f"Notable decline in creative freedom index ({cf_delta:+.2f})",
            })

        # Significant creative recovery
        elif cf_delta > 0.15:
            inflections.append({
                "period": periods[i],
                "type": "creative_recovery",
                "magnitude": f"{cf_delta:.2f}",
                "description": f"Notable recovery in creative freedom index (+{cf_delta:.2f})",
            })

        # Significant pressure increase
        if cp_delta > 0.15:
            inflections.append({
                "period": periods[i],
                "type": "pressure_increase",
                "magnitude": f"{cp_delta:.2f}",
                "description": f"Cultural pressure index increases sharply (+{cp_delta:.2f})",
            })

    return inflections


# ── Prediction ─────────────────────────────────────────────────────────────────

def _predict_next_period(
    profile: CorpusTemporalProfile,
) -> Tuple[Dict[str, Any], float, str]:
    """
    Predict the next time period's values using linear extrapolation
    from the last 3 periods. Returns (prediction_dict, confidence, reasoning).
    """
    if len(profile.periods) < 3:
        return {}, 0.0, "Insufficient periods for prediction."

    # Simple linear regression prediction for key metrics
    def _extrapolate(values: List[float]) -> float:
        if len(values) < 2:
            return values[-1] if values else 0.0
        x = np.arange(len(values), dtype=float)
        coeffs = np.polyfit(x, values, 1)
        return float(np.clip(np.polyval(coeffs, len(values)), 0.0, 1.0))

    last_period_end = profile.period_boundaries[-1][1] if profile.period_boundaries else 2025
    period_size = (profile.period_boundaries[-1][1] - profile.period_boundaries[-1][0] + 1
                   if profile.period_boundaries else 5)

    prediction = {
        "period": f"{last_period_end + 1}-{last_period_end + period_size}",
        "lsii": _extrapolate(profile.lsii_trend),
        "authentic_emission": _extrapolate(profile.authentic_emission_trend),
        "manufacturing": _extrapolate(profile.manufacturing_trend),
        "creative_freedom": _extrapolate(profile.creative_freedom_index),
        "cultural_pressure": _extrapolate(profile.cultural_pressure_index),
    }

    # Confidence: higher when trends are consistent (low variance in recent slope)
    recent = profile.creative_freedom_index[-3:]
    if len(recent) >= 2:
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        trend_stability = 1.0 - min(np.std(diffs), 1.0)
        confidence = float(trend_stability * 0.7)  # Max 70% confidence for linear extrapolation
    else:
        confidence = 0.3

    # Reasoning
    cf_trend = "declining" if prediction["creative_freedom"] < profile.creative_freedom_index[-1] else "stable or recovering"
    cp_trend = "increasing" if prediction["cultural_pressure"] > profile.cultural_pressure_index[-1] else "stable or declining"
    reasoning = (
        f"Linear extrapolation from {len(profile.periods)} periods. "
        f"Creative freedom trend: {cf_trend}. "
        f"Cultural pressure trend: {cp_trend}."
    )

    return prediction, confidence, reasoning


# ── Summary narrative ─────────────────────────────────────────────────────────

def _build_trend_summary(profile: CorpusTemporalProfile) -> str:
    """Build a plain-language summary of the corpus temporal trends."""
    n = len(profile.periods)
    if n < 2:
        return "Single period corpus — no trend analysis possible."

    span = f"{profile.period_boundaries[0][0]}–{profile.period_boundaries[-1][1]}"
    total = sum(profile.record_counts)

    # Directional assessment of freedom and pressure
    if len(profile.creative_freedom_index) >= 2:
        cf_start = profile.creative_freedom_index[0]
        cf_end = profile.creative_freedom_index[-1]
        cf_direction = "declined" if cf_end < cf_start - 0.05 else (
            "increased" if cf_end > cf_start + 0.05 else "remained stable"
        )
    else:
        cf_direction = "varied"

    sos_note = ""
    if profile.sos_periods:
        sos_note = (
            f" SOS signature detected in {len(profile.sos_periods)} period(s): "
            f"{', '.join(profile.sos_periods[:3])}."
        )

    summary = (
        f"Corpus spans {span} across {n} time periods ({total} records). "
        f"Creative freedom index {cf_direction} over the period covered.{sos_note}"
    )

    if profile.inflection_points:
        first_inflection = profile.inflection_points[0]
        summary += (
            f" First notable inflection: {first_inflection['period']} — "
            f"{first_inflection['description']}."
        )

    return summary

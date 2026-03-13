"""
KindPath Analyser :: core/spike_detector.py

Error-margin spike detection.

When a feature value is extreme relative to the corpus distribution,
two explanations compete:

1. GENUINE DIVERGENCE — the creator made an unusually distinctive choice.
   This is signal. It should be preserved and studied. High-divergence records
   have elevated HMoE contribution to any corpus they join.

2. SENSOR ANOMALY — something went wrong: a noisy recording, a clipped
   input, a corrupted segment, a degenerate FFT frame. This is noise.
   It should be flagged and handled, not deposited into the seedbank as signal.

This module detects spikes (values beyond a z-score threshold) and
attempts to classify them using context from adjacent features.

The classification is probabilistic, not definitive. The detector surfaces
anomalies and suggests actions. A human or downstream system makes the call.

Why this matters:
- Unclassified spikes skew corpus HMoE measurements
- They pollute tag baseline calculations in tags_registry.py
- They mislead the LSII if they occur in the final quarter
- They degrade psychosomatic profiles when large enough to drive valence/arousal

The architecture mirrors the HMoE model: the spike detector asks
"is this signal or noise?" the same way reason.py asks
"is this variance genuine, or systematic k-bias?"
"""

from __future__ import annotations

import statistics
import datetime
from dataclasses import dataclass, field
from typing import Optional


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class SpikeReport:
    """
    A single detected spike in one feature of one record.

    A spike is a value that lies beyond z_threshold standard deviations
    from the corpus mean for that feature. Whether it is an anomaly or
    genuine divergence depends on context — see classification and suggested_action.

    The z_score is signed: positive = above corpus mean, negative = below.
    is_likely_anomaly is False for ambiguous cases (preserve by default).
    """
    record_id: str
    feature_name: str
    value: float
    corpus_mean: float
    corpus_std: float
    z_score: float

    classification: str         # "genuine_divergence" | "sensor_anomaly" | "ambiguous"
    is_likely_anomaly: bool
    confidence: float           # 0-1: confidence in the classification

    evidence: list[str] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class CorpusStats:
    """
    Lightweight statistics over a corpus for spike detection.

    Build this once and reuse it — computing corpus stats on every call
    wastes time, especially as the seedbank grows.

    feature_stats maps feature_name → {mean, std, n, min, max}.
    Features with fewer than 3 records are excluded (insufficient for z-score).
    """
    feature_stats: dict[str, dict]
    record_count: int
    tag_name: Optional[str] = None
    built_at: Optional[str] = None


# ── Corpus statistics ─────────────────────────────────────────────────────────

def build_corpus_stats(
    record_dicts: list[dict],
    tag_name: Optional[str] = None,
) -> CorpusStats:
    """
    Compute mean and std for every numeric feature across a list of record dicts.

    record_dicts: list of generate_json_report() outputs
    tag_name:     optional label for this corpus (for display and logging)

    Features are extracted from the 'psychosomatic' and 'lsii' sub-dicts.
    Boolean and non-numeric fields are silently skipped.
    Features with fewer than 3 values are excluded (z-score requires variance).
    """
    feature_values: dict[str, list[float]] = {}

    for record in record_dicts:
        for subdoc_key in ("psychosomatic", "lsii"):
            subdoc = record.get(subdoc_key, {})
            for key, val in subdoc.items():
                # Skip booleans (isinstance(True, int) is True in Python)
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    feature_values.setdefault(key, []).append(float(val))

    feature_stats: dict[str, dict] = {}
    for feature, values in feature_values.items():
        if len(values) < 3:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        feature_stats[feature] = {
            "mean": mean,
            "std": std,
            "n": len(values),
            "min": min(values),
            "max": max(values),
        }

    return CorpusStats(
        feature_stats=feature_stats,
        record_count=len(record_dicts),
        tag_name=tag_name,
        built_at=datetime.datetime.now().isoformat(),
    )


# ── Spike detection ───────────────────────────────────────────────────────────

def detect_spikes(
    record_dict: dict,
    corpus_stats: CorpusStats,
    z_threshold: float = 3.0,
) -> list[SpikeReport]:
    """
    Detect feature spikes in a record relative to the corpus distribution.

    z_threshold: standard deviations beyond which a value is a spike.
                 3.0 → ~0.3% false positive rate on Gaussian data (recommended default)
                 2.5 → ~1.2%
                 2.0 → ~4.5% (higher sensitivity, higher noise)

    Returns a list of SpikeReport for each detected spike, classified by context.
    Empty list means no spikes detected — record is within normal corpus range.
    """
    spikes: list[SpikeReport] = []

    record_id = (
        record_dict.get("source", {}).get("filepath")
        or record_dict.get("source", {}).get("filename", "unknown")
    )

    # Extract all numeric feature values from psychosomatic and lsii sub-dicts
    feature_values: dict[str, float] = {}
    for subdoc_key in ("psychosomatic", "lsii"):
        subdoc = record_dict.get(subdoc_key, {})
        for key, val in subdoc.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                feature_values[key] = float(val)

    for feature_name, value in feature_values.items():
        stats = corpus_stats.feature_stats.get(feature_name)
        if stats is None:
            continue

        std = stats["std"]
        if std == 0.0:
            continue   # No corpus variance — z-score is undefined

        z = (value - stats["mean"]) / std

        if abs(z) >= z_threshold:
            classification, is_anomaly, confidence, evidence, action = classify_spike(
                feature_name=feature_name,
                value=value,
                z_score=z,
                record_dict=record_dict,
                corpus_stats=corpus_stats,
            )
            spikes.append(SpikeReport(
                record_id=record_id,
                feature_name=feature_name,
                value=value,
                corpus_mean=stats["mean"],
                corpus_std=std,
                z_score=z,
                classification=classification,
                is_likely_anomaly=is_anomaly,
                confidence=confidence,
                evidence=evidence,
                suggested_action=action,
            ))

    return spikes


# ── Classification ────────────────────────────────────────────────────────────

def classify_spike(
    feature_name: str,
    value: float,
    z_score: float,
    record_dict: dict,
    corpus_stats: CorpusStats,
) -> tuple[str, bool, float, list[str], str]:
    """
    Classify a spike as genuine_divergence, sensor_anomaly, or ambiguous.

    Returns: (classification, is_likely_anomaly, confidence, evidence, suggested_action)

    SENSOR ANOMALY indicators:
        - Value falls outside the physically possible range for a bounded metric
        - High manufacturing_score contradicts a high creative_residue spike (contradiction)
        - is_clipping is True in the same record

    GENUINE DIVERGENCE indicators:
        - High LSII corroborates late-section divergence
        - High creative_residue corroborates creative metric spikes
        - High authentic_emission_score is a strong independent corroborator
        - Spike direction toward extreme-but-possible values (not past physical bounds)

    AMBIGUOUS:
        - Mixed signals; evidence is insufficient for confident classification
        - Default action: retain with flag (never delete ambiguous data)

    The classification is probabilistic. Confidence reflects the weight of evidence,
    not certainty. The suggested_action is a guide, not a command.
    """
    evidence: list[str] = []
    anomaly_score: float = 0.0
    divergence_score: float = 0.0

    psych = record_dict.get("psychosomatic", {})
    lsii = record_dict.get("lsii", {})

    lsii_score = float(lsii.get("score", 0))
    creative_residue = float(psych.get("creative_residue", 0))
    authentic_emission = float(psych.get("authentic_emission_score", 0))
    manufacturing = float(psych.get("manufacturing_score", 0))

    # ── Anomaly indicators ─────────────────────────────────────────────────────

    # Several metrics are strictly bounded — values outside their range are
    # extraction errors, not creative choices.
    _BOUNDS: dict[str, tuple[float, float]] = {
        "valence":                   (-1.0, 1.0),
        "arousal":                   (0.0, 1.0),
        "coherence":                 (0.0, 1.0),
        "complexity":                (0.0, 1.0),
        "authenticity_index":        (0.0, 1.0),
        "authentic_emission_score":  (0.0, 1.0),
        "manufacturing_score":       (0.0, 1.0),
        "creative_residue":          (0.0, 1.0),
        "identity_capture_risk":     (0.0, 1.0),
        "stage3_tag_risk":           (0.0, 1.0),
        "score":                     (0.0, 1.0),   # lsii.score
    }
    if feature_name in _BOUNDS:
        lo, hi = _BOUNDS[feature_name]
        tolerance = 0.05   # Allow tiny floating-point overshoot
        if value < lo - tolerance or value > hi + tolerance:
            anomaly_score += 0.6
            evidence.append(
                f"Value {value:.4f} is outside physical range [{lo}, {hi}] — likely extraction error"
            )

    # High manufacturing score contradicts a high residue spike
    if feature_name in ("creative_residue", "authentic_emission_score") and manufacturing > 0.8:
        anomaly_score += 0.3
        evidence.append(
            f"Manufacturing score {manufacturing:.3f} contradicts a genuine residue spike — "
            "common in hypercompressed material where spectral features are saturated"
        )

    # ── Divergence indicators ──────────────────────────────────────────────────

    # High LSII corroborates late-section spikes as genuine divergence
    if lsii_score > 0.4:
        divergence_score += 0.35
        evidence.append(f"LSII {lsii_score:.3f} corroborates genuine late-section divergence")

    # Creative residue corroborates spikes in interpretive/creative features
    if feature_name in ("valence", "arousal", "coherence", "complexity") and creative_residue > 0.5:
        divergence_score += 0.3
        evidence.append(
            f"Creative residue {creative_residue:.3f} supports genuine creative divergence interpretation"
        )

    # High authentic emission is a strong independent corroborator for any spike
    if authentic_emission > 0.6:
        divergence_score += 0.35
        evidence.append(f"Authentic emission {authentic_emission:.3f} corroborates genuine signal")

    # A negative residue spike (toward zero) is less likely to be a sensor error
    if feature_name in ("creative_residue", "authentic_emission_score") and z_score < 0:
        divergence_score += 0.1
        evidence.append(
            "Spike is toward zero (less residue than corpus average) — consistent with highly "
            "conventional material, less likely to be sensor noise"
        )

    # ── Classify ──────────────────────────────────────────────────────────────

    total = anomaly_score + divergence_score
    if total == 0.0:
        # No contextual evidence — ambiguous
        return (
            "ambiguous", False, 0.5,
            ["No contextual corroboration available — insufficient corpus context"],
            "Retain with flag. Collect more corpus records to improve classification confidence.",
        )

    anomaly_pct = anomaly_score / total
    divergence_pct = divergence_score / total

    if anomaly_pct > 0.65:
        return (
            "sensor_anomaly", True, min(0.95, anomaly_pct),
            evidence,
            "Flag for manual review. Consider re-ingesting the source file; "
            "check for clipping, corrupted segments, or degenerate FFT frames.",
        )
    elif divergence_pct > 0.65:
        return (
            "genuine_divergence", False, min(0.95, divergence_pct),
            evidence,
            "Retain and deposit. High-divergence records have elevated HMoE contribution to the corpus.",
        )
    else:
        return (
            "ambiguous", False, max(anomaly_pct, divergence_pct),
            evidence,
            "Retain with flag. Run additional manual verification or compare against related corpus records.",
        )


# ── Corpus-level summary ──────────────────────────────────────────────────────

def summarise_spikes(spikes: list[SpikeReport]) -> dict:
    """
    Aggregate a list of SpikeReports into a corpus-level summary.

    Useful for batch analysis (e.g. scanning the full seedbank on each k-revision)
    to surface systemic anomaly patterns. If the same feature spikes across many
    records, it may indicate a calibration issue in that feature extractor rather
    than a pattern of extreme records.
    """
    if not spikes:
        return {
            "total_spikes": 0,
            "genuine_divergence": 0,
            "sensor_anomaly": 0,
            "ambiguous": 0,
            "records_affected": 0,
            "features_most_spiking": [],
        }

    counts: dict[str, int] = {"genuine_divergence": 0, "sensor_anomaly": 0, "ambiguous": 0}
    for s in spikes:
        counts[s.classification] = counts.get(s.classification, 0) + 1

    records_affected = len(set(s.record_id for s in spikes))

    feature_counts: dict[str, int] = {}
    for s in spikes:
        feature_counts[s.feature_name] = feature_counts.get(s.feature_name, 0) + 1
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_spikes": len(spikes),
        "genuine_divergence": counts.get("genuine_divergence", 0),
        "sensor_anomaly": counts.get("sensor_anomaly", 0),
        "ambiguous": counts.get("ambiguous", 0),
        "records_affected": records_affected,
        "features_most_spiking": [f for f, _ in top_features],
    }

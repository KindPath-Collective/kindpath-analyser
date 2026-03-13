"""
KindPress :: reason.py

The semantic reasoner — surveys Δ distributions across the seedbank to
determine whether the shared constants bank (k) is well-calibrated.

The core question this module answers:
    "Is the current k the most coherent explanation of what these records share?"

If k is well-calibrated:
    - Records sharing a tag have LOW systematic correlation in their Δ values
    - Each record's creative_residue is driven by genuine per-record signal
    - Residue variance across the tag cluster is HIGH (k is discriminating)
    - The Δ distribution is approximately zero-centred (no systematic bias)

If k is miscalibrated:
    - Records sharing a tag show CORRELATED Δ values (systematic bias)
    - The residue_delta history (across k-revisions) shows directed drift
    - Residue variance is LOW (k is over-generalising)
    - The Δ distribution is offset from zero (k is under- or over-crediting)

This is an Expectation-Maximisation loop at the meaning layer:
    E-step: given k, compute Δ for all records
    M-step: given all Δ, ask whether k is the most coherent explanation,
            or whether a revision would reduce systematic bias
    Iterate until HMoE (information density) is maximised.

HMoE here is the Heterogeneous Multiplicity of Evidence — the information
richness of a compressed corpus. It is measured in two forms:

    Scalar HMoE (original):
        Φ = var(creative_residue) * mean(effective_n)
    where var(residue) captures discriminating power and mean(effective_n)
    captures lineage depth.

    Four-Pillar HMoE (extended):
        Φ_4p = mean(var_per_pillar) * mean(effective_n)
    where var_per_pillar averages discrimination across four evidence channels:
        - Spiritual: creative_residue, lsii_score, authentic_emission_score
        - Intellectual: manufacturing_score, tempo_bpm
        - Emotional: valence, arousal (when present)
        - Somatic: dynamic_range_db, crest_factor_db, groove_deviation_ms (when present)

    A high scalar HMoE but low Φ_4p signals single-pillar compression:
    the corpus is only well-described from the Spiritual dimension. The
    analysis has explanatory power but is not holistically grounded.

    This extends the inference vs. insight principle from the Four Pillars
    of Holistic Perception into the k-calibration layer. An insight requires
    convergence across multiple pillars. A corpus that only discriminates on
    one axis is still producing inferences, not insights.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import seedbank.index as _idx
import seedbank.tags_registry as _tags
from seedbank.recompute import effective_n, corpus_effective_n_distribution


# ── Four Pillars field mapping ────────────────────────────────────────────────
# Each pillar cluster maps to the seedbank record fields that give evidence
# for that dimension of experience / analysis.
#
# Spiritual: authenticity, meaning, the late-song protest signal — what
#            remains after all known formula elements are subtracted.
# Intellectual: production logic, technique era, structural choices —
#               what the maker decided with deliberate craft knowledge.
# Emotional: valence, arousal, tension — what the music does to the
#            feeling layer of a listener. Often absent in basic records.
# Somatic: physical dynamics, rhythm, compression — what the music does
#          to the body. Physical rhythm, breath, tactile presence.
#
# A corpus that discriminates well on all four pillars has explanatory
# richness across every mode of human experience. A corpus that only
# discriminates on one axis (typically Spiritual, since creative_residue
# drives the standard HMoE) is still producing inferences, not insights.

_PILLAR_FIELDS: dict[str, list[str]] = {
    "spiritual": ["creative_residue", "lsii_score", "authentic_emission_score"],
    "intellectual": ["manufacturing_score", "tempo_bpm"],
    "emotional": ["valence", "arousal", "tension_resolution_ratio"],
    "somatic": ["dynamic_range_db", "crest_factor_db", "groove_deviation_ms"],
}


@dataclass
class DeltaDistribution:
    """
    Statistical profile of the Δ distribution for a tag cluster.

    A healthy tag cluster has:
    - High residue_variance (k is discriminating — records are distinct)
    - Low residue_bias (Δ distribution is zero-centred — no systematic offset)
    - Low residue_correlation (records are independently fingerprinted)
    - High effective_n_mean (corpus is in the living, non-linear regime)

    An unhealthy cluster (k miscalibration signal) shows:
    - Low residue_variance (k over-generalises — all records look alike)
    - High residue_bias (k systematically under/over-credits this tag)
    - High residue_correlation (records are moving together = shared k error)
    - Directed drift (revision history shows k consistently moving same way)
    """
    tag_name: str
    n_records: int
    tag_version: int

    # Residue distribution
    residue_mean: float
    residue_variance: float
    residue_std: float
    residue_min: float
    residue_max: float
    residue_range: float

    # Δ bias: how far the mean is from zero
    # Near zero = k is centred, capturing average shared signal
    # Offset = k is systematically under/over-crediting this tag
    residue_bias: float

    # Reading history depth
    n_records_with_history: int
    effective_n_mean: float
    effective_n_max: float

    # Drift: how much residue has shifted across k-revisions
    # High drift = records are sensitive to k changes (k is doing real work)
    # Near-zero drift = k revisions are not affecting this cluster (k may be irrelevant)
    mean_total_drift: float
    max_total_drift: float

    # HMoE: the composite information density score
    # Φ = var(Δ) * mean(effective_n) — the combined measure of discrimination × depth
    hmoe: float

    # Four-Pillar HMoE: discrimination tested across all four evidence channels.
    # Φ_4p = mean(var_per_pillar) * mean(effective_n)
    # A corpus where only one pillar discriminates well is producing inferences,
    # not insights — the Four Pillars principle applied to k-calibration.
    # hmoe_4p of 0.0 means no pillar data was available beyond Spiritual.
    pillar_variances: dict = field(default_factory=dict)
    # Keys: "spiritual", "intellectual", "emotional", "somatic"
    # Values: mean variance across the fields in that pillar (NaN if no coverage)
    pillar_coverage: dict = field(default_factory=dict)
    # Fraction of records (0–1) that have at least one field from each pillar
    hmoe_4p: float = 0.0
    # Four-pillar HMoE. Lower than hmoe when insight is single-pillar dominated.

    # Calibration assessment
    calibration_score: float = 0.0  # 0–1: 1.0 = well-calibrated, 0.0 = needs revision
    calibration_notes: list = field(default_factory=list)


def _compute_pillar_variances(
    records: list[dict],
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Compute per-pillar variance and coverage across a set of records.

    For each pillar in _PILLAR_FIELDS, collects all numeric values available
    across records for each field in that pillar, computes variance per field,
    then averages to produce a single pillar-level variance value.

    Coverage is the fraction of records that have at least one numeric value
    for any field in the pillar.

    Returns:
        pillar_variances: dict[pillar_name, mean_variance] — NaN excluded pillars
        pillar_coverage:  dict[pillar_name, coverage_fraction]
        hmoe_4p_base:     mean(available_pillar_variances) — caller multiplies by eff_n
    """
    n = len(records)
    if n == 0:
        return ({}, {}, 0.0)

    pillar_variances: dict[str, float] = {}
    pillar_coverage: dict[str, float] = {}

    for pillar, fields in _PILLAR_FIELDS.items():
        field_variances: list[float] = []
        records_with_any = 0

        for field_name in fields:
            values = []
            for rec in records:
                val = rec.get(field_name)
                if isinstance(val, (int, float)) and not math.isnan(float(val)):
                    values.append(float(val))

            if values:
                records_with_any = max(records_with_any, len(values))
                var = statistics.variance(values) if len(values) > 1 else 0.0
                field_variances.append(var)

        pillar_coverage[pillar] = round(records_with_any / n, 3)

        if field_variances:
            pillar_variances[pillar] = round(
                statistics.mean(field_variances), 6
            )
        # Pillars with no data are simply absent from pillar_variances.

    available_vars = list(pillar_variances.values())
    hmoe_4p_base = statistics.mean(available_vars) if available_vars else 0.0

    return pillar_variances, pillar_coverage, hmoe_4p_base


def analyse_delta_distribution(tag_name: str) -> DeltaDistribution:
    """
    Compute the full Δ distribution for all records bearing a given tag.

    Loads the index, finds all records with tag_name in their tags list,
    and computes statistical measures of the Δ (creative_residue) distribution.

    Returns a DeltaDistribution describing the k-calibration health for
    this tag cluster.
    """
    tag = _tags.get_tag(tag_name)
    if tag is None:
        raise ValueError(f"Tag '{tag_name}' not found in registry.")

    index = _idx._load_index()
    tagged_records = [
        r for r in index.get("records", [])
        if tag_name in r.get("tags", [])
    ]

    n = len(tagged_records)
    if n == 0:
        return DeltaDistribution(
            tag_name=tag_name,
            n_records=0,
            tag_version=tag.current_version,
            residue_mean=0.0, residue_variance=0.0, residue_std=0.0,
            residue_min=0.0, residue_max=0.0, residue_range=0.0,
            residue_bias=0.0,
            n_records_with_history=0,
            effective_n_mean=1.0, effective_n_max=1.0,
            mean_total_drift=0.0, max_total_drift=0.0,
            hmoe=0.0,
            pillar_variances={}, pillar_coverage={}, hmoe_4p=0.0,
            calibration_score=0.5,
            calibration_notes=["No records with this tag — cannot assess calibration."],
        )

    residues = [r.get("creative_residue", 0.0) for r in tagged_records]

    residue_mean = statistics.mean(residues)
    residue_variance = statistics.variance(residues) if n > 1 else 0.0
    residue_std = math.sqrt(residue_variance)
    residue_min = min(residues)
    residue_max = max(residues)
    residue_range = residue_max - residue_min

    # Bias: mean Δ offset from zero. A well-calibrated k produces a near-zero mean
    # (it captures what's shared, leaving unsystematic residue)
    residue_bias = residue_mean

    # Reading history and effective_n
    records_with_history = [
        r for r in tagged_records
        if len(r.get("reading_history", [])) > 1
    ]
    n_with_history = len(records_with_history)

    effective_ns = []
    total_drifts = []

    for rec in tagged_records:
        history = rec.get("reading_history", [])
        if len(history) > 1:
            try:
                en = effective_n(rec["id"])
                effective_ns.append(en)
            except Exception:
                effective_ns.append(1.0)

            # Total drift: birth universe residue → current universe residue
            birth = history[0].get("creative_residue", 0.0)
            current = history[-1].get("creative_residue", 0.0)
            total_drifts.append(abs(current - birth))
        else:
            effective_ns.append(1.0)
            total_drifts.append(0.0)

    eff_n_mean = statistics.mean(effective_ns) if effective_ns else 1.0
    eff_n_max = max(effective_ns) if effective_ns else 1.0
    mean_drift = statistics.mean(total_drifts) if total_drifts else 0.0
    max_drift = max(total_drifts) if total_drifts else 0.0

    # HMoE: var(Δ) × mean(effective_n)
    # This is the information density of the compressed corpus under this tag's k.
    # High HMoE = k is discriminating AND the corpus is in the living, non-linear regime.
    hmoe = round(residue_variance * eff_n_mean, 6)

    # Four-Pillar HMoE: apply the same logic across all four evidence channels.
    # Insight requires convergence across pillars, not just depth in one.
    pillar_variances, pillar_coverage, hmoe_4p_base = _compute_pillar_variances(tagged_records)
    hmoe_4p = round(hmoe_4p_base * eff_n_mean, 6)

    # Calibration assessment
    score, notes = _assess_calibration(
        n=n,
        residue_variance=residue_variance,
        residue_bias=residue_bias,
        residue_range=residue_range,
        mean_drift=mean_drift,
        eff_n_mean=eff_n_mean,
        n_with_history=n_with_history,
        pillar_variances=pillar_variances,
        pillar_coverage=pillar_coverage,
    )

    return DeltaDistribution(
        tag_name=tag_name,
        n_records=n,
        tag_version=tag.current_version,
        residue_mean=round(residue_mean, 5),
        residue_variance=round(residue_variance, 5),
        residue_std=round(residue_std, 5),
        residue_min=round(residue_min, 5),
        residue_max=round(residue_max, 5),
        residue_range=round(residue_range, 5),
        residue_bias=round(residue_bias, 5),
        n_records_with_history=n_with_history,
        effective_n_mean=round(eff_n_mean, 4),
        effective_n_max=round(eff_n_max, 4),
        mean_total_drift=round(mean_drift, 5),
        max_total_drift=round(max_drift, 5),
        hmoe=hmoe,
        pillar_variances=pillar_variances,
        pillar_coverage=pillar_coverage,
        hmoe_4p=hmoe_4p,
        calibration_score=score,
        calibration_notes=notes,
    )


def _assess_calibration(
    n: int,
    residue_variance: float,
    residue_bias: float,
    residue_range: float,
    mean_drift: float,
    eff_n_mean: float,
    n_with_history: int,
    pillar_variances: dict | None = None,
    pillar_coverage: dict | None = None,
) -> tuple:
    """
    Assess k-calibration health from distribution statistics.

    Returns (calibration_score 0-1, list of notes).
    """
    score = 1.0
    notes = []

    # Low variance: k over-generalises — all records look alike under this tag
    if n >= 3 and residue_variance < 0.01:
        score -= 0.25
        notes.append(
            f"Low residue variance ({residue_variance:.4f}): tag may be over-broad — "
            "records under this tag are nearly indistinguishable from each other."
        )

    # Bias: k systematically under/over-credits this tag
    if abs(residue_bias) > 0.15:
        direction = "over-crediting" if residue_bias > 0 else "under-crediting"
        score -= 0.20
        notes.append(
            f"Residue bias {residue_bias:+.3f}: k is {direction} this tag cluster — "
            "consider revising the tag description to rebalance."
        )

    # Very small range: almost no discrimination
    if n >= 3 and residue_range < 0.05:
        score -= 0.20
        notes.append(
            f"Narrow residue range ({residue_range:.4f}): tag definition may be too narrow "
            "or the sample is too homogeneous to test calibration."
        )

    # High drift with history: k revisions are doing significant work on this cluster
    # This is not necessarily bad — it means k is sensitive here (real signal zone)
    if n_with_history > 0 and mean_drift > 0.2:
        notes.append(
            f"Mean total drift {mean_drift:.3f} across revisions: "
            "this tag cluster is sensitive to k changes. Revisions here have real impact."
        )

    # No history yet: can't assess drift sensitivity
    if n_with_history == 0 and n >= 3:
        notes.append(
            "No revision history yet: drift sensitivity unknown. "
            "This is the first k-universe for these records."
        )

    # Low effective_n: corpus hasn't entered the non-linear living regime yet
    if eff_n_mean < 1.1:
        notes.append(
            "effective_n near 1.0: corpus is in the linear regime — "
            "records have not yet accumulated fork/confluence history under this tag."
        )

    # Four-Pillar coverage: flag when evidence is single-pillar dominated.
    # The Spiritual pillar (creative_residue / LSII) is always measured.
    # If other pillars are missing or near-zero, the analysis is inference-grade,
    # not insight-grade — it lacks the multi-modal convergence that distinguishes
    # genuine insight from single-channel pattern recognition.
    if pillar_variances is not None and pillar_coverage is not None:
        covered = [p for p, cov in pillar_coverage.items() if cov > 0.0]
        if len(covered) == 1:
            score -= 0.10
            notes.append(
                f"Single-pillar evidence ({covered[0]} only): analysis is inference-grade. "
                "Records with somatic, intellectual, or emotional fields will elevate to insight-grade. "
                "Consider enriching records with full psychosomatic profile data."
            )
        elif len(covered) == 2:
            notes.append(
                f"Two-pillar evidence ({', '.join(covered)}): partial insight. "
                "Adding records with full four-pillar field coverage will improve Φ_4p."
            )
        elif len(covered) >= 3:
            notes.append(
                f"Multi-pillar evidence ({', '.join(covered)}): insight-grade corpus. "
                "Four-pillar HMoE is active."
            )

    if not notes:
        notes.append("Distribution appears well-calibrated — no systematic bias detected.")

    return round(max(score, 0.0), 3), notes


def k_calibration_score(tag_name: str) -> float:
    """
    Return the 0–1 calibration score for a tag's current definition.

    1.0 = well-calibrated: k captures shared meaning cleanly, Δ is information-dense.
    0.0 = needs revision: systematic bias, over-generalisation, or low discrimination.

    Quick wrapper around analyse_delta_distribution() for decision-making.
    """
    dist = analyse_delta_distribution(tag_name)
    return dist.calibration_score


def survey_all_tags(min_records: int = 2) -> list:
    """
    Survey calibration health across all active tags in the registry.

    Returns a list of DeltaDistribution objects sorted by calibration_score ascending
    (worst-calibrated first — the tags most in need of attention surface at the top).

    min_records: skip tags with fewer records than this threshold
    (insufficient data for meaningful assessment).
    """
    all_tags = _tags.list_tags(include_deprecated=False)
    results = []
    for tag in all_tags:
        dist = analyse_delta_distribution(tag.name)
        if dist.n_records >= min_records:
            results.append(dist)
    results.sort(key=lambda d: d.calibration_score)
    return results


def compare_k_versions(tag_name: str) -> list:
    """
    Survey how records' creative_residue has moved across every k-revision
    for a given tag.

    Returns a list of dicts, one per k-revision, showing:
    - which version transition (v1→v2, v2→v3, etc.)
    - how many records shifted
    - the mean and max delta in creative_residue for that revision

    This is the directed drift audit: it shows whether k revisions have
    been consistently moving the corpus in one direction (systematic calibration
    bias) or exploring the full range (genuine refinement).
    """
    tag = _tags.get_tag(tag_name)
    if tag is None:
        raise ValueError(f"Tag '{tag_name}' not found.")

    index = _idx._load_index()
    tagged_records = [
        r for r in index.get("records", [])
        if tag_name in r.get("tags", [])
    ]

    # For each record with reading history, find readings bracketing a tag revision
    # by matching the baseline_version strings for the tag's version number.
    version_transitions = {}
    for v in range(1, tag.current_version):
        key = f"v{v}→v{v+1}"
        version_transitions[key] = []

    for rec in tagged_records:
        history = rec.get("reading_history", [])
        for i in range(len(history) - 1):
            prev = history[i]
            curr = history[i + 1]

            # Extract this tag's version from each reading's baseline_version
            prev_ver = _tags._extract_tag_version(prev.get("baseline_version", ""), tag_name)
            curr_ver = _tags._extract_tag_version(curr.get("baseline_version", ""), tag_name)

            if prev_ver is not None and curr_ver is not None and curr_ver == prev_ver + 1:
                key = f"v{prev_ver}→v{curr_ver}"
                delta = curr.get("creative_residue", 0.0) - prev.get("creative_residue", 0.0)
                version_transitions.setdefault(key, []).append(delta)

    results = []
    for transition, deltas in sorted(version_transitions.items()):
        if deltas:
            results.append({
                "transition": transition,
                "n_affected": len(deltas),
                "mean_delta": round(statistics.mean(deltas), 5),
                "max_delta": round(max(deltas, key=abs), 5),
                "all_same_direction": all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas),
                "notes": (
                    "Directed drift — all records moved the same way. "
                    "This may reflect intentional k recalibration or systematic bias."
                ) if (all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)) else
                "Mixed drift — records moved in different directions. Normal refinement."
            })
        else:
            results.append({
                "transition": transition,
                "n_affected": 0,
                "mean_delta": 0.0,
                "max_delta": 0.0,
                "all_same_direction": True,
                "notes": "No records with reading history across this transition.",
            })

    return results


def hmoe_of_corpus(record_ids: Optional[list] = None) -> float:
    """
    Compute HMoE for a set of records (or the full corpus if record_ids is None).

    HMoE = var(creative_residue) × mean(effective_n)

    This is the corpus-level information density score. It answers:
    "How information-rich is the compressed representation of this corpus?"

    Used by the validator to measure whether a proposed k revision
    increases or decreases HMoE — the evolutionary optimum criterion.
    """
    index = _idx._load_index()
    records = index.get("records", [])

    if record_ids is not None:
        id_set = set(record_ids)
        records = [r for r in records if r.get("id") in id_set]

    if len(records) < 2:
        return 0.0

    residues = [r.get("creative_residue", 0.0) for r in records]
    variance = statistics.variance(residues)

    en_values = []
    for rec in records:
        if len(rec.get("reading_history", [])) > 1:
            try:
                en_values.append(effective_n(rec["id"]))
            except Exception:
                en_values.append(1.0)
        else:
            en_values.append(1.0)

    mean_en = statistics.mean(en_values)
    return round(variance * mean_en, 6)

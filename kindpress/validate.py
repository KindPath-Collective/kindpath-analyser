"""
KindPress :: validate.py

Tag revision validation protocol.

Before any tag definition change propagates to the seedbank, this module
runs the proposed revision through a staged test:

    STAGE 1 — Sample selection
        Draw records bearing the tag, stratified by residue quartile so
        the sample represents the full distribution, not just common cases.

    STAGE 2 — Compressed HMoE baseline
        Encode the sample as KindPress packets (k + Δ).
        Compute HMoE = var(creative_residue) × mean(effective_n).
        This is the information density under the CURRENT k.

    STAGE 3 — Simulated k revision
        Apply the proposed description/scope change as a temporary k shift.
        "Applying" means: which records would this revision flag as stale?
        Estimate the expected residue shift using each record's drift history
        (how much its creative_residue typically moves on recomputation).

    STAGE 4 — Projected HMoE under proposed k
        Using the estimated post-revision residues, compute the projected HMoE.
        If projected HMoE > current HMoE: the revision increases information
        density — the corpus becomes more discriminating. Sound revision.
        If projected HMoE <= current HMoE: the revision reduces information
        density — it over-generalises or introduces systematic noise.

    STAGE 5 — Evolutionary optimum search
        Repeat stages 2–4 across increasing chunk sizes until HMoE is maximised.
        The evolutionary optimum is the chunk size where HMoE peaks — beyond
        this, adding more records to the scope decreases information density.
        The proposed revision is valid only if it improves HMoE at the optimum,
        not just at small sample sizes (a revision that only looks good on small
        data is likely overfitting to the sample).

    STAGE 6 — DB-wide implication report
        At the evolutionary optimum chunk size, scale the analysis to all
        records bearing the tag. Report: how many records will be flagged stale,
        what is the expected total drift, and what is the HMoE trajectory.

The result is a TagRevisionReport with a clear recommendation:
    APPROVE: revision is sound — improves HMoE at evolutionary optimum
    CAUTION: revision is sound at small scale but degrades at optimum
    REJECT: revision reduces HMoE — k would become less calibrated

No tag revision should be committed without passing this validation.
The report becomes the rationale for revision_reason in revise_tag().
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import seedbank.index as _idx
import seedbank.tags_registry as _tags
from seedbank.recompute import effective_n
from kindpress.press import encode, KindPressPacket
from kindpress.reason import hmoe_of_corpus, analyse_delta_distribution


@dataclass
class HMoEProbe:
    """
    One measurement in the evolutionary optimum search.

    chunk_size: number of records in this probe
    current_hmoe: HMoE under current k at this chunk size
    projected_hmoe: projected HMoE under proposed k at this chunk size
    delta_hmoe: projected - current (positive = revision improves)
    stale_rate: proportion of chunk that would be flagged stale by proposed revision
    mean_expected_drift: mean estimated residue shift for stale records
    """
    chunk_size: int
    current_hmoe: float
    projected_hmoe: float
    delta_hmoe: float
    stale_rate: float
    mean_expected_drift: float
    is_evolutionary_optimum: bool = False


@dataclass
class TagRevisionReport:
    """
    Full validation report for a proposed tag revision.

    recommendation: APPROVE | CAUTION | REJECT
    is_sound: True if recommendation is APPROVE
    evolutionary_optimum_chunk: the chunk size where current HMoE is maximised
    hmoe_probes: measurements at each chunk size tested
    db_wide_stale_count: how many records in the full DB would be flagged stale
    db_wide_expected_drift: mean expected residue drift across all affected records
    hmoe_at_optimum_current: HMoE of current k at the evolutionary optimum
    hmoe_at_optimum_projected: projected HMoE of proposed k at the evolutionary optimum
    calibration_notes: from the current distribution analysis
    implication_summary: plain language summary of DB-wide implications
    """
    tag_name: str
    tag_current_version: int
    proposed_description: str
    proposed_scope: str
    validated_at: str

    recommendation: str          # APPROVE | CAUTION | REJECT
    is_sound: bool

    evolutionary_optimum_chunk: int
    hmoe_probes: list = field(default_factory=list)

    db_wide_stale_count: int = 0
    db_wide_expected_drift: float = 0.0
    hmoe_at_optimum_current: float = 0.0
    hmoe_at_optimum_projected: float = 0.0

    calibration_notes: list = field(default_factory=list)
    implication_summary: str = ""
    revision_rationale: str = ""   # Suggested revision_reason for revise_tag()


def _stratified_sample(records: list, n: int, seed: int = 42) -> list:
    """
    Draw n records from records, stratified by creative_residue quartile.
    This ensures the sample represents the full distribution, not just the mean.
    """
    if len(records) <= n:
        return list(records)

    # Divide into 4 quartile buckets by creative_residue
    sorted_recs = sorted(records, key=lambda r: r.get("creative_residue", 0.0))
    buckets = [[], [], [], []]
    bucket_size = max(1, len(sorted_recs) // 4)
    for i, rec in enumerate(sorted_recs):
        bucket_idx = min(i // bucket_size, 3)
        buckets[bucket_idx].append(rec)

    # Sample proportionally from each bucket
    rng = random.Random(seed)
    per_bucket = max(1, n // 4)
    sample = []
    for bucket in buckets:
        take = min(per_bucket, len(bucket))
        sample.extend(rng.sample(bucket, take))

    # Top up to exactly n if needed
    remaining = [r for r in records if r not in sample]
    rng.shuffle(remaining)
    sample.extend(remaining[: n - len(sample)])
    return sample[:n]


def _estimate_drift_per_record(record: dict) -> float:
    """
    Estimate how much a record's creative_residue would shift under a k revision,
    using its reading history as a guide.

    If the record has revision history: use mean absolute residue_delta from
    prior revisions as the estimate.
    If no history: use the corpus-wide mean drift for this tag as a proxy
    (caller should provide a fallback).

    Returns an estimated absolute drift value.
    """
    history = record.get("reading_history", [])
    if len(history) < 2:
        return 0.0  # No history — assume conservative zero drift

    deltas = []
    for reading in history[1:]:
        delta = reading.get("residue_delta_from_prior")
        if delta is not None:
            deltas.append(abs(delta))

    return statistics.mean(deltas) if deltas else 0.0


def _project_residue(record: dict, estimated_drift: float) -> float:
    """
    Project what a record's creative_residue would be under the proposed k.

    We don't know the direction of the drift without running the full audio
    pipeline, so we model it as a reduction: the proposed revision is expected
    to either credit or debit the tag contribution. We return the residue
    adjusted by the estimated magnitude.

    For the purposes of HMoE calculation, we care about whether the variance
    INCREASES or DECREASES — so we spread the projected residues around the
    current distribution centre using the drift as "noise floor estimate".
    """
    return record.get("creative_residue", 0.0) + estimated_drift


def _compute_projected_hmoe(
    records: list,
    estimated_drifts: list,
    projected_residues: Optional[list] = None,
) -> float:
    """
    Compute projected HMoE using estimated post-revision residue values.

    Uses the same formula as hmoe_of_corpus but with projected residues
    replacing the current creative_residue values.
    """
    if len(records) < 2:
        return 0.0

    if projected_residues is None:
        projected_residues = [
            _project_residue(r, d) for r, d in zip(records, estimated_drifts)
        ]

    if len(set(projected_residues)) < 2:
        return 0.0

    variance = statistics.variance(projected_residues)

    en_values = []
    for rec in records:
        if len(rec.get("reading_history", [])) > 1:
            try:
                en_values.append(effective_n(rec["id"]))
            except Exception:
                en_values.append(1.0)
        else:
            en_values.append(1.0)

    # Proposed revision adds one more reading → increments effective_n by 1.0
    # (conservative estimate — assumes single-lineage fork, not confluence)
    projected_en_values = [v + 1.0 for v in en_values]
    mean_en = statistics.mean(projected_en_values)
    return round(variance * mean_en, 6)


def validate_tag_revision(
    tag_name: str,
    proposed_description: str,
    proposed_scope: str,
    chunk_sizes: Optional[list] = None,
    max_sample: int = 200,
    seed: int = 42,
) -> TagRevisionReport:
    """
    Full validation protocol for a proposed tag revision.

    Args:
        tag_name: the tag to be revised
        proposed_description: the new description text
        proposed_scope: the new scope text
        chunk_sizes: list of sample sizes to probe (default: [10, 25, 50, 100, 200])
        max_sample: maximum records to use for the full DB-wide assessment
        seed: random seed for reproducible stratified sampling

    Returns a TagRevisionReport with recommendation and implication summary.
    """
    if chunk_sizes is None:
        chunk_sizes = [10, 25, 50, 100, 200]

    tag = _tags.get_tag(tag_name)
    if tag is None:
        raise ValueError(f"Tag '{tag_name}' not found in registry.")

    validated_at = datetime.now(timezone.utc).isoformat()

    # Load ALL records bearing this tag
    index = _idx._load_index()
    all_tagged = [
        r for r in index.get("records", [])
        if tag_name in r.get("tags", [])
    ]

    # --- Run the distribution analysis on current k ---
    dist = analyse_delta_distribution(tag_name)

    # --- Stage 5: Evolutionary optimum search across chunk sizes ---
    probes = []
    for chunk_size in sorted(chunk_sizes):
        if chunk_size > len(all_tagged) and len(all_tagged) > 0:
            chunk_size = len(all_tagged)  # clamp to available data

        if len(all_tagged) == 0:
            probes.append(HMoEProbe(
                chunk_size=chunk_size,
                current_hmoe=0.0,
                projected_hmoe=0.0,
                delta_hmoe=0.0,
                stale_rate=0.0,
                mean_expected_drift=0.0,
            ))
            continue

        sample = _stratified_sample(all_tagged, chunk_size, seed=seed)
        sample_ids = [r["id"] for r in sample]

        # Current HMoE for this chunk
        current_hmoe = hmoe_of_corpus(sample_ids)

        # Estimate which records in the sample would be flagged stale
        # by the proposed revision. Proxy: any record where the tag's
        # current version appears in its baseline_version with the current
        # version number (these are "up to date" and would be flagged by
        # the new revision).
        stale_count = 0
        estimated_drifts = []
        for rec in sample:
            tag_ver = _tags._extract_tag_version(
                rec.get("baseline_version", ""), tag_name
            )
            # A record encoded under the current version would be flagged stale
            # by a new version. Records not yet encoded under this tag are
            # already stale (version mismatch or no version → always stale).
            if tag_ver is not None and tag_ver >= tag.current_version:
                stale_count += 1

            drift = _estimate_drift_per_record(rec)
            estimated_drifts.append(drift)

        stale_rate = stale_count / len(sample) if sample else 0.0
        mean_expected_drift = statistics.mean(estimated_drifts) if estimated_drifts else 0.0

        # Projected HMoE: what would HMoE be if the revision were applied?
        # Stale records would be recomputed; non-stale records remain unchanged.
        # For stale records, project a new residue using estimated drift.
        projected_residues = []
        for rec, drift in zip(sample, estimated_drifts):
            tag_ver = _tags._extract_tag_version(
                rec.get("baseline_version", ""), tag_name
            )
            if tag_ver is not None and tag_ver >= tag.current_version:
                # Stale under proposed revision → project new residue
                projected_residues.append(_project_residue(rec, drift))
            else:
                # Not affected by this revision
                projected_residues.append(rec.get("creative_residue", 0.0))

        projected_hmoe = _compute_projected_hmoe(
            sample, estimated_drifts, projected_residues
        )
        delta_hmoe = round(projected_hmoe - current_hmoe, 6)

        probes.append(HMoEProbe(
            chunk_size=len(sample),
            current_hmoe=current_hmoe,
            projected_hmoe=projected_hmoe,
            delta_hmoe=delta_hmoe,
            stale_rate=stale_rate,
            mean_expected_drift=mean_expected_drift,
        ))

        # Avoid probing the same chunk size twice (happens when len(all_tagged) < chunk_size)
        if len(sample) == len(all_tagged):
            break

    # Find the evolutionary optimum: chunk size where current HMoE is maximised
    if probes:
        optimum_probe = max(probes, key=lambda p: p.current_hmoe)
        optimum_probe.is_evolutionary_optimum = True
        opt_chunk = optimum_probe.chunk_size
        opt_current_hmoe = optimum_probe.current_hmoe
        opt_projected_hmoe = optimum_probe.projected_hmoe
    else:
        opt_chunk = 0
        opt_current_hmoe = 0.0
        opt_projected_hmoe = 0.0

    # --- Stage 6: DB-wide implications at the evolutionary optimum ---
    # Scale the stale_rate and drift estimates to the full dataset
    opt_probe = next((p for p in probes if p.is_evolutionary_optimum), None)
    db_stale_count = 0
    db_mean_drift = 0.0

    if opt_probe and all_tagged:
        db_stale_count = round(len(all_tagged) * opt_probe.stale_rate)
        db_mean_drift = opt_probe.mean_expected_drift

    # --- Determine recommendation ---
    recommendation, is_sound = _make_recommendation(
        probes=probes,
        optimum_probe=opt_probe,
        dist=dist,
    )

    # --- Build implication summary ---
    implication_summary = _build_implication_summary(
        tag_name=tag_name,
        all_tagged_count=len(all_tagged),
        db_stale_count=db_stale_count,
        db_mean_drift=db_mean_drift,
        opt_chunk=opt_chunk,
        opt_current_hmoe=opt_current_hmoe,
        opt_projected_hmoe=opt_projected_hmoe,
        recommendation=recommendation,
        dist=dist,
    )

    # --- Suggested revision rationale ---
    revision_rationale = _build_revision_rationale(
        proposed_description=proposed_description,
        implication_summary=implication_summary,
        dist=dist,
        recommendation=recommendation,
    )

    return TagRevisionReport(
        tag_name=tag_name,
        tag_current_version=tag.current_version,
        proposed_description=proposed_description,
        proposed_scope=proposed_scope,
        validated_at=validated_at,
        recommendation=recommendation,
        is_sound=is_sound,
        evolutionary_optimum_chunk=opt_chunk,
        hmoe_probes=[vars(p) for p in probes],
        db_wide_stale_count=db_stale_count,
        db_wide_expected_drift=round(db_mean_drift, 5),
        hmoe_at_optimum_current=opt_current_hmoe,
        hmoe_at_optimum_projected=opt_projected_hmoe,
        calibration_notes=dist.calibration_notes,
        implication_summary=implication_summary,
        revision_rationale=revision_rationale,
    )


def _make_recommendation(
    probes: list,
    optimum_probe,
    dist,
) -> tuple:
    """Returns (recommendation_str, is_sound)."""
    if not probes or optimum_probe is None:
        return "CAUTION", False

    # No data case
    if dist.n_records == 0:
        return "APPROVE", True

    delta_at_optimum = optimum_probe.delta_hmoe

    # APPROVE: revision improves HMoE at the evolutionary optimum
    if delta_at_optimum > 0:
        return "APPROVE", True

    # CAUTION: revision improves HMoE at small scale but not at optimum
    small_probes = [p for p in probes if p.chunk_size <= 25]
    if small_probes and any(p.delta_hmoe > 0 for p in small_probes):
        return "CAUTION", False

    # REJECT: revision reduces HMoE at all scales, or no improvement
    return "REJECT", False


def _build_implication_summary(
    tag_name, all_tagged_count, db_stale_count, db_mean_drift,
    opt_chunk, opt_current_hmoe, opt_projected_hmoe, recommendation, dist,
) -> str:
    lines = [
        f"Tag: {tag_name} (v{dist.tag_version} → proposed revision)",
        f"Records in scope: {all_tagged_count}",
        f"Records flagged stale at evolutionary optimum (chunk={opt_chunk}): "
        f"{db_stale_count} (~{round(db_stale_count/all_tagged_count*100)}%)" if all_tagged_count > 0 else
        f"Records flagged stale: {db_stale_count}",
        f"Mean expected residue drift per stale record: {db_mean_drift:.4f}",
        f"",
        f"HMoE at evolutionary optimum (chunk={opt_chunk}):",
        f"  Current k:  {opt_current_hmoe:.6f}",
        f"  Proposed k: {opt_projected_hmoe:.6f}",
        f"  Δ HMoE:     {opt_projected_hmoe - opt_current_hmoe:+.6f}",
        f"",
        f"Recommendation: {recommendation}",
    ]
    if recommendation == "APPROVE":
        lines.append(
            "The proposed revision increases information density at the evolutionary "
            "optimum. It clarifies what this tag means without over-generalising. "
            "Proceed with revise_tag() and flag_stale_records()."
        )
    elif recommendation == "CAUTION":
        lines.append(
            "The revision looks beneficial at small scale but reduces HMoE at the "
            "evolutionary optimum (full corpus scope). It may overfit to common cases. "
            "Consider narrowing scope or adding anti_examples before proceeding."
        )
    else:
        lines.append(
            "The revision reduces information density at all tested scales. "
            "It would make records under this tag less distinguishable from each other, "
            "or introduce systematic bias. Revise the proposed description before proceeding."
        )

    if dist.calibration_notes:
        lines.append("")
        lines.append("Current calibration notes:")
        for note in dist.calibration_notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def _build_revision_rationale(
    proposed_description, implication_summary, dist, recommendation
) -> str:
    """
    Generate a suggested revision_reason string suitable for use in revise_tag().
    This is the audit trail note that explains why the revision was made.
    """
    lines = [
        f"Validated by KindPress reasoner ({recommendation}).",
        f"Current calibration score: {dist.calibration_score:.3f}.",
    ]
    if dist.n_records > 0:
        lines.append(
            f"Distribution: {dist.n_records} records, "
            f"residue mean={dist.residue_mean:.3f}, "
            f"var={dist.residue_variance:.4f}, "
            f"bias={dist.residue_bias:+.3f}, "
            f"HMoE={dist.hmoe:.6f}."
        )
    lines.append(f"Proposed change: {proposed_description[:120]}")
    return " ".join(lines)


def print_report(report: TagRevisionReport) -> None:
    """Pretty-print a TagRevisionReport to stdout."""
    print(f"\n{'=' * 60}")
    print(f"KindPress Tag Revision Report")
    print(f"{'=' * 60}")
    print(f"Tag:             {report.tag_name} (v{report.tag_current_version})")
    print(f"Validated at:    {report.validated_at}")
    print(f"Recommendation:  {report.recommendation}")
    print(f"")
    print(f"Evolutionary optimum chunk size: {report.evolutionary_optimum_chunk}")
    print(f"HMoE at optimum — current:  {report.hmoe_at_optimum_current:.6f}")
    print(f"HMoE at optimum — proposed: {report.hmoe_at_optimum_projected:.6f}")
    print(f"")
    print(f"HMoE probes:")
    for probe in report.hmoe_probes:
        opt_marker = " ← OPTIMUM" if probe.get("is_evolutionary_optimum") else ""
        print(
            f"  chunk={probe['chunk_size']:4d}  "
            f"current={probe['current_hmoe']:.5f}  "
            f"projected={probe['projected_hmoe']:.5f}  "
            f"Δ={probe['delta_hmoe']:+.5f}  "
            f"stale={probe['stale_rate']:.1%}{opt_marker}"
        )
    print(f"")
    print(f"DB-wide implications:")
    print(f"  Stale records:     {report.db_wide_stale_count}")
    print(f"  Mean drift:        {report.db_wide_expected_drift:.4f}")
    print(f"")
    print(f"Implication summary:")
    for line in report.implication_summary.split("\n"):
        print(f"  {line}")
    print(f"")
    print(f"Revision rationale (for revise_tag()):")
    print(f"  {report.revision_rationale}")
    print(f"{'=' * 60}\n")

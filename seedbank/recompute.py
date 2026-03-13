"""
KindPath Analyser :: Seedbank Recompute

Implements the fork-and-retain principle across the reading history of seedbank records.

When the constants bank (k) is revised, records are not overwritten.
They fork: the prior reading is preserved as is_current=False; the new reading is added
as is_current=True. Both remain valid within their respective k-universes.

**Fork-and-retain — the lineage weaving principle:**

    A reading produced under one version of k is not a mistake waiting to be corrected
    by the next version. It is a real interpretation within a real k-universe. When k
    is revised — when a tag definition is updated, a baseline recalculated, an era
    profile refined — the prior reading is preserved as a historical node in the record's
    interpretational lineage. The revised reading becomes the recommended present
    interpretation. Both exist. Neither erases the other.

    This is not ambiguity. It is precision: specifying which constants were active
    when a fingerprint was computed, so the fingerprint can be faithfully understood
    by a future reader who was not there.

    The raw audio signal (the source) is unchanged. What changes is the lens through
    which it is read. Revising k produces a new interpretational branch — not a
    correction of the old one. Both are real within their respective universes.

This is multiversal with a time coefficient engaged: each k-version defines a
distinct universe of valid interpretation, navigable by the baseline_version
timestamp embedded in every reading.

The delta between readings is itself a signal — it is the measure of how far
the model's understanding has moved between the two k-versions.

Lineage confluence (confluent_reading):
    Once multiple divergent universes have accumulated, readings from distinct
    k-universes can be combined — confluenced — to produce a new synthetic reading
    that carries the interpretational lineage of both origins. The parent_reading_ids
    field records exact provenance: which universes converged to produce this one.

    The total space of possible readings across N universes grows as:
        n = 1 + 2 + 3 + ... + n^(±n)
    where the sum across all pair, triple... N-wise combinations, multiplied by the
    HMoE ±n valence, describes the full combinatorial space. This maps directly
    to the exponent in Φ = km^n: n is not a fixed coefficient but the active
    dimensionality of the confluence space at any point in time.
    As universes accumulate and confluence, n grows — and Φ grows non-linearly.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import seedbank.index as _idx
import seedbank.tags_registry as _tags


def add_reading(
    record_id: str,
    new_creative_residue: float,
    new_authentic_emission_score: float,
    new_manufacturing_score: float,
    new_baseline_version: str,
    new_reconstruction_protocol: str,
    computation_source: str = "recompute",
    parent_reading_ids: Optional[list] = None,
) -> dict:
    """
    Fork-and-retain: add a new reading to a record without erasing prior universes.

    The prior reading is marked is_current=False but preserved in reading_history.
    The record's headline values are updated to the new reading.
    The creative_residue_stale flag is cleared.

    parent_reading_ids: for confluent readings, the list of reading computed_at
    timestamps that were combined to produce this reading. None for single-lineage forks.
    Provenance of the confluence enables the combinatorial universe map to be traversed later.

    The residue_delta_from_prior field records the distance between the two universes
    of interpretation — how much the model has moved between the two k-versions.

    Returns the updated record dict.
    Raises ValueError if record_id is not found.
    """
    index = _idx._load_index()
    computed_at = datetime.now(timezone.utc).isoformat()

    updated_record = None
    for rec in index["records"]:
        if rec.get("id") != record_id:
            continue

        # Mark all prior readings as no longer current — they remain valid in their
        # own k-universes, but this new reading becomes the recommended present interpretation.
        for reading in rec.get("reading_history", []):
            reading["is_current"] = False

        # The residue delta is the distance between the old universe and the new one.
        # Large delta = the model has substantially revised its understanding of this piece.
        # Near-zero delta = the tag revision didn't change the fundamental fingerprint.
        old_residue = rec.get("creative_residue", 0.0)
        old_baseline_version = rec.get("baseline_version", "")
        residue_delta = new_creative_residue - old_residue

        # reading_input_hash: SHA-256 of the canonical input values that produced this
        # reading. Together with computed_at, this proves what data was present when
        # this k-universe was constructed — enabling later audits to verify no retroactive
        # alteration of the values took place after the reading was recorded.
        _reading_inputs = json.dumps(
            {
                "creative_residue": new_creative_residue,
                "authentic_emission_score": new_authentic_emission_score,
                "manufacturing_score": new_manufacturing_score,
                "baseline_version": new_baseline_version,
            },
            sort_keys=True,
        )
        new_reading = {
            "creative_residue": new_creative_residue,
            "authentic_emission_score": new_authentic_emission_score,
            "manufacturing_score": new_manufacturing_score,
            "baseline_version": new_baseline_version,
            "reconstruction_protocol": new_reconstruction_protocol,
            "computed_at": computed_at,
            "is_current": True,
            "computation_source": computation_source,
            "residue_delta_from_prior": residue_delta,
            # Confluent provenance: which prior reading universes were combined.
            # None = single-lineage fork. List of computed_at timestamps = confluence.
            "parent_reading_ids": parent_reading_ids,
            "reading_input_hash": hashlib.sha256(_reading_inputs.encode("utf-8")).hexdigest(),
        }
        rec.setdefault("reading_history", []).append(new_reading)

        # Update the record's headline values to the current universe
        rec["creative_residue"] = new_creative_residue
        rec["authentic_emission_score"] = new_authentic_emission_score
        rec["manufacturing_score"] = new_manufacturing_score
        rec["baseline_version"] = new_baseline_version
        rec["reconstruction_protocol"] = new_reconstruction_protocol

        # Clear the stale flag — recomputation has resolved it.
        # The stale_tags list is recorded in the reading for provenance.
        stale_tags = rec.pop("stale_tags", [])
        rec.pop("creative_residue_stale", None)

        index["last_updated"] = computed_at
        updated_record = rec

        print(
            f"[RECOMPUTE] Fork-and-retain: {record_id[:8]}… "
            f"residue {old_residue:.3f} → {new_creative_residue:.3f} "
            f"(Δ={residue_delta:+.3f}) | stale_tags cleared: {stale_tags or 'none'}"
        )
        break

    if updated_record is None:
        raise ValueError(f"Record not found in index: {record_id}")

    _idx._save_index(index)

    # Log every fork to the audit trail — the definitive record of when k branched.
    # confluent_reading() calls add_reading() with computation_source="confluent";
    # all other callers produce single-lineage forks.
    try:
        import seedbank.fork_log as _fork_log
        if computation_source == "confluent":
            _fork_log.log_record_confluence(
                record_id=record_id,
                filename=updated_record.get("filename", "?"),
                parent_reading_ids=parent_reading_ids or [],
                composite_baseline=new_baseline_version,
                residue_delta=residue_delta,
            )
        else:
            _fork_log.log_record_fork(
                record_id=record_id,
                filename=updated_record.get("filename", "?"),
                old_baseline_version=old_baseline_version,
                new_baseline_version=new_baseline_version,
                residue_delta=residue_delta,
                computation_source=computation_source,
            )
    except Exception:
        pass

    # Append to the CDC hash-chained audit log.
    try:
        from seedbank.cdc import append_event as _cdc_append
        cdc_event_type = "confluence" if computation_source == "confluent" else "fork"
        _cdc_append(
            cdc_event_type,
            record_id,
            {
                "filename": updated_record.get("filename", ""),
                "old_baseline_version": old_baseline_version,
                "new_baseline_version": new_baseline_version,
                "residue_delta": residue_delta,
                "computation_source": computation_source,
                "parent_reading_ids": parent_reading_ids,
                "reading_input_hash": new_reading["reading_input_hash"],
            },
        )
    except Exception:
        pass

    return updated_record


def recompute_stale_with_fn(
    recompute_fn: Callable[[dict], dict],
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> list:
    """
    Recompute all stale records using a caller-supplied recompute function.

    The seedbank stores analysis results, not raw audio — it cannot recompute
    without external help. This function accepts a recompute_fn that knows how
    to derive new values for a given record.

    recompute_fn receives the current index record dict (with filename, context,
    baseline_version, etc.) and must return a dict containing:
        creative_residue, authentic_emission_score, manufacturing_score,
        baseline_version, reconstruction_protocol

    If recompute_fn raises, the record is skipped and remains stale.

    dry_run: if True, compute and print the delta but do not write anything.
    limit: maximum records to process (None = all stale records).

    Returns a list of (record_id, status, residue_delta) tuples:
        status is "ok", "dry_run", or "error"
    """
    stale = _tags.get_stale_records()
    if limit is not None:
        stale = stale[:limit]

    if not stale:
        print("[RECOMPUTE] No stale records found.")
        return []

    print(f"[RECOMPUTE] Processing {len(stale)} stale record(s)…")
    results = []

    for rec in stale:
        record_id = rec.get("id")
        filename = rec.get("filename", "?")
        try:
            new_values = recompute_fn(rec)
            delta = new_values["creative_residue"] - rec.get("creative_residue", 0.0)

            if dry_run:
                print(f"[RECOMPUTE DRY] {filename}: residue delta would be {delta:+.3f}")
                results.append((record_id, "dry_run", delta))
            else:
                add_reading(
                    record_id=record_id,
                    new_creative_residue=new_values["creative_residue"],
                    new_authentic_emission_score=new_values["authentic_emission_score"],
                    new_manufacturing_score=new_values["manufacturing_score"],
                    new_baseline_version=new_values["baseline_version"],
                    new_reconstruction_protocol=new_values["reconstruction_protocol"],
                    computation_source="recompute",
                )
                results.append((record_id, "ok", delta))

        except Exception as exc:
            print(f"[RECOMPUTE] Skipped {filename} ({record_id[:8] if record_id else '?'}…): {exc}")
            results.append((record_id, "error", None))

    return results


def reading_history(record_id: str) -> list:
    """
    Return the full reading history for a record — all k-universes of interpretation,
    ordered from oldest (birth universe) to newest (current universe).

    Each entry is a dict with: creative_residue, authentic_emission_score,
    manufacturing_score, baseline_version, reconstruction_protocol, computed_at,
    is_current, computation_source, residue_delta_from_prior.
    """
    index = _idx._load_index()
    for rec in index["records"]:
        if rec.get("id") == record_id:
            return rec.get("reading_history", [])
    raise ValueError(f"Record not found: {record_id}")


def residue_arc(record_id: str) -> list:
    """
    Return the creative_residue arc across all k-universes for a record.

    Each entry is a (baseline_version, creative_residue, computed_at) tuple.
    The arc shows how the model's understanding of this piece has evolved —
    the distance between entries is the distance between k-universes.
    A flat arc means the piece's fingerprint is stable across constant revisions.
    A steep arc means the piece's meaning is highly sensitive to k.
    """
    history = reading_history(record_id)
    return [
        (r["baseline_version"], r["creative_residue"], r["computed_at"])
        for r in history
    ]


def confluent_reading(
    record_id: str,
    parent_reading_timestamps: list,
    synthesis_fn: Callable[[list], dict],
    reconstruction_note: str = "",
) -> dict:
    """
    Lineage confluence: combine readings from distinct k-universes into a new synthetic reading.

    This is only meaningful once multiple divergent universes have accumulated — the
    time coefficient is engaged. The parent lineages must carry distinct interpretational
    material (sufficiently different baseline_versions) for the confluence to produce
    new signal rather than noise.

    parent_reading_timestamps: list of `computed_at` values identifying the parent readings.
        These are the exact fork-point coordinates in the multiverse. Two readings from
        the same k-version produce a degenerate confluence — synthesis_fn will see near-identical
        inputs and produce near-identical output. The system allows it; the data reveals it.

    synthesis_fn: receives the list of parent reading dicts and returns a new dict containing
        at minimum: creative_residue, authentic_emission_score, manufacturing_score.
        It defines the confluence algorithm — weighted average, max-divergence,
        HMoE-weighted combination, or any other synthesis strategy.

    The resulting reading carries parent_reading_ids so the full lineage map is traversable.
    The baseline_version of the confluence reading is a composite of its parents, prefixed "confluence:".

    Returns the updated record dict.
    Raises ValueError if record_id or any parent timestamp is not found.
    """
    history = reading_history(record_id)
    ts_set = set(parent_reading_timestamps)

    parent_readings = [r for r in history if r.get("computed_at") in ts_set]
    found_ts = {r["computed_at"] for r in parent_readings}
    missing = ts_set - found_ts
    if missing:
        raise ValueError(f"Parent reading timestamp(s) not found: {missing}")

    new_values = synthesis_fn(parent_readings)

    # Composite baseline: all parent k-universes joined, prefixed to mark as a confluence.
    # This baseline_version string is itself a coordinate: it records which universes
    # contributed to this reading.
    parent_baselines = "|".join(r.get("baseline_version", "?") for r in parent_readings)
    composite_baseline = f"confluence:{parent_baselines}"

    protocol = (
        f"Confluence of {len(parent_readings)} reading universe(s). "
        f"Parent baselines: {parent_baselines}. "
        + (reconstruction_note or "Synthesis algorithm defined by caller.")
    )

    return add_reading(
        record_id=record_id,
        new_creative_residue=float(new_values["creative_residue"]),
        new_authentic_emission_score=float(new_values["authentic_emission_score"]),
        new_manufacturing_score=float(new_values["manufacturing_score"]),
        new_baseline_version=composite_baseline,
        new_reconstruction_protocol=protocol,
        computation_source="confluent",
        parent_reading_ids=list(parent_reading_timestamps),
    )


def effective_n(record_id: str) -> float:
    """
    Compute the effective exponent n for this record's position in Φ = km^n.

    n is not a fixed coefficient. It grows with the dimensionality of the active
    confluence space: each fork adds 1, each confluent reading adds log2(num_parents)
    (because it draws from multiple lineages simultaneously).

    The HMoE ±n valence is reflected in the sign of each reading's residue_delta:
    positive deltas contribute +, negative deltas contribute - to the running total.
    The final value is the scalar effective exponent for the field equation at this moment.

    A record that has never been forked has n=1 (linear — it exists in a single universe).
    A record with a history of forks and confluences approaches n=2 and beyond — the
    living, generative, non-linear regime described in THE_FIELD_EQUATION.md.
    """
    history = reading_history(record_id)
    if len(history) <= 1:
        # Single universe: linear, extractive exponent
        return 1.0

    n = 1.0
    for reading in history[1:]:
        parents = reading.get("parent_reading_ids")
        delta = reading.get("residue_delta_from_prior") or 0.0
        hmoe_sign = 1.0 if delta >= 0 else -1.0

        if parents and len(parents) > 1:
            # Confluent reading: draws from multiple lineages — contributes log2(num_parents) to n
            import math
            n += hmoe_sign * math.log2(len(parents))
        else:
            # Single-lineage fork: contributes 1 to n dimension
            n += hmoe_sign * 1.0

    return round(n, 4)


def corpus_effective_n_distribution() -> list:
    """
    Return (record_id, filename, effective_n) for all records with reading history,
    sorted by effective_n descending.

    The distribution shows which pieces have accumulated the richest confluence
    history — which ones have moved most deeply into the non-linear Φ regime.
    """
    index = _idx._load_index()
    results = []
    for rec in index["records"]:
        if len(rec.get("reading_history", [])) > 1:
            try:
                n = effective_n(rec["id"])
                results.append((rec["id"], rec.get("filename", "?"), n))
            except Exception:
                pass
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def corpus_residue_drift(top_n: int = 10) -> list:
    """
    Across all records with multiple readings, compute the total drift in creative_residue
    between their birth reading and current reading.

    Returns a list of (record_id, filename, total_drift, num_readings) tuples,
    sorted by |total_drift| descending.

    Records with large drift are the most sensitive to k-revisions — they are the pieces
    where the model's understanding has moved the most. These are worth re-examining.
    """
    index = _idx._load_index()
    drifts = []

    for rec in index["records"]:
        history = rec.get("reading_history", [])
        if len(history) < 2:
            continue
        birth_residue = history[0].get("creative_residue", 0.0)
        current_residue = history[-1].get("creative_residue", 0.0)
        drift = current_residue - birth_residue
        drifts.append((rec["id"], rec.get("filename", "?"), drift, len(history)))

    drifts.sort(key=lambda x: abs(x[2]), reverse=True)
    return drifts[:top_n]

"""
KindPath Analyser :: Seedbank Deposit

Accepts a completed JSON analysis profile (from generate_json_report())
and archives it to the seedbank corpus.

The seedbank is the growing reference library that makes comparison possible.
Each deposit sharpens the baseline and extends the authentic creative record.
"""

import json
import uuid
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import seedbank.index as _idx
import seedbank.tags_registry as _tags


@dataclass
class SeedbankReading:
    """
    One interpretation of a record's analytical values, computed within a specific k-context.

    When the constants bank is revised, the record forks: the prior reading is preserved
    with is_current=False; the new reading is added with is_current=True.
    Both are valid within their respective k-universes.

    The raw audio signal is the invariant checkpoint — the genome.
    The k-context (era baseline, tag constants) is the environment.
    The expressed values (creative_residue, etc.) are environment-specific.
    Revising k is selective breeding that retains all prior potentials from the checkpoint.
    """
    creative_residue: float
    authentic_emission_score: float
    manufacturing_score: float
    baseline_version: str             # The k-context this reading was computed in
    reconstruction_protocol: str      # How creative_residue was derived in this context
    computed_at: str                  # ISO timestamp of this reading
    is_current: bool = True           # False once a newer reading supersedes this one
    computation_source: str = "deposit"  # "deposit" | "recompute" | "manual"
    residue_delta_from_prior: Optional[float] = None  # None for first reading; delta thereafter


@dataclass
class SeedbankRecord:
    """
    A complete seedbank entry. The minimum persistent representation
    of an analysed work — enough to search, compare, and derive baselines.
    """
    id: str
    deposited_at: str                     # ISO 8601 timestamp
    filename: str
    context: str                          # Creator-provided description
    release_circumstances: Optional[str]  # Independent / label / forced schedule / etc
    creator_statement: Optional[str]      # Optional direct creator voice

    # Core analysis values (the searchable surface)
    duration_seconds: float
    lsii_score: float
    lsii_flag_level: str
    authentic_emission_score: float
    manufacturing_score: float
    creative_residue: float
    era_fingerprint: str
    key_estimate: str
    tempo_bpm: float

    # Path to the full profile JSON (relative to seedbank/records/)
    full_profile_path: str

    # Searchability
    tags: List[str] = field(default_factory=list)
    genre_estimate: str = ""

    # Provenance
    verified: bool = False
    verification_notes: str = ""

    # Conscious contextualisation — compression traceability
    # The constants that were factored out when computing creative_residue,
    # and the algorithm used to do it. Sufficient to re-derive the individual
    # fingerprint from raw features if the baseline is later revised.
    baseline_version: str = ""          # Which constants bank snapshot was in use at deposit time
    reconstruction_protocol: str = ""  # How creative_residue was computed:
                                        # baseline used, constants factored, delta threshold

    # Fork-and-retain: reading history across all k-universes.
    # Each revision to the constants bank adds a new reading rather than overwriting the original.
    # reading_history[0] is always the birth universe — the original deposit reading.
    # Each subsequent entry is a fork: a new valid interpretation under a revised k.
    # The baseline_version within each reading is the time coordinate in the multiverse.
    reading_history: List[dict] = field(default_factory=list)


def deposit(
    profile: dict,
    context: str,
    release_circumstances: Optional[str] = None,
    creator_statement: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> SeedbankRecord:
    """
    Deposit a complete analysis profile to the seedbank.

    profile: the dict returned by generate_json_report().
    context: a plain-language description provided by the depositor — what is
             this piece, who made it, what was the intention?

    Writes the full profile to seedbank/records/{id}.json and updates the index.
    Returns the SeedbankRecord entry.
    """
    records_dir = _idx.RECORDS_DIR
    os.makedirs(records_dir, exist_ok=True)

    record_id = str(uuid.uuid4())
    deposited_at = datetime.now(timezone.utc).isoformat()

    # Extract summary values from the profile for the index entry
    source = profile.get("source", {})
    lsii = profile.get("lsii", {})
    psychosomatic = profile.get("psychosomatic", {})
    fingerprints = profile.get("fingerprints", {})

    filename = source.get("filename", "unknown")
    duration_seconds = float(source.get("duration_seconds", 0.0))
    lsii_score = float(lsii.get("lsii_score", 0.0))
    lsii_flag_level = str(lsii.get("flag_level", "none"))
    authentic_emission_score = float(psychosomatic.get("authentic_emission_score", 0.0))
    manufacturing_score = float(psychosomatic.get("manufacturing_score", 0.0))
    creative_residue = float(psychosomatic.get("creative_residue", 0.0))

    # Era fingerprint — take the top match name if present
    era_matches = fingerprints.get("era_matches", [])
    era_fingerprint = era_matches[0].get("name", "unknown") if era_matches else "unknown"

    # Harmonic key — from the arcs
    arcs = profile.get("arcs", {})
    key_estimate = ""
    tempo_bpm = 0.0

    genre_estimate = fingerprints.get("production_context", "")

    # Auto-generate tags from the analysis
    auto_tags = _auto_tags(lsii_score, lsii_flag_level, authentic_emission_score, manufacturing_score, creative_residue)
    all_tags = list(set((tags or []) + auto_tags))

    # Encode the current version of every tag at deposit time.
    # This is the baseline_version string — the constants bank snapshot.
    # Any future revision to these tag definitions will make this record
    # flaggable as stale without losing its original fingerprint.
    computed_baseline_version = _tags.current_baseline_version(all_tags)

    # Build the reconstruction protocol: a human-readable description of
    # which constants were factored out when deriving creative_residue.
    era_label = era_fingerprint or "unknown"
    computed_reconstruction_protocol = (
        f"creative_residue derived by subtracting era baseline '{era_label}' "
        f"from raw harmonic/spectral/temporal features. "
        f"Baseline sourced from seedbank corpus at deposit time. "
        f"Tag constants encoded as: {computed_baseline_version}."
    )

    # Write the full profile to disk
    full_profile_filename = f"{record_id}.json"
    full_profile_path = os.path.join(records_dir, full_profile_filename)

    with open(full_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    # The birth reading — the first universe this record is born into.
    # All future k-revisions fork from this checkpoint rather than overwriting it.
    initial_reading = {
        "creative_residue": creative_residue,
        "authentic_emission_score": authentic_emission_score,
        "manufacturing_score": manufacturing_score,
        "baseline_version": computed_baseline_version,
        "reconstruction_protocol": computed_reconstruction_protocol,
        "computed_at": deposited_at,
        "is_current": True,
        "computation_source": "deposit",
        "residue_delta_from_prior": None,
    }

    record = SeedbankRecord(
        id=record_id,
        deposited_at=deposited_at,
        filename=filename,
        context=context,
        release_circumstances=release_circumstances,
        creator_statement=creator_statement,
        duration_seconds=duration_seconds,
        lsii_score=lsii_score,
        lsii_flag_level=lsii_flag_level,
        authentic_emission_score=authentic_emission_score,
        manufacturing_score=manufacturing_score,
        creative_residue=creative_residue,
        era_fingerprint=era_fingerprint,
        key_estimate=key_estimate,
        tempo_bpm=tempo_bpm,
        full_profile_path=full_profile_path,
        tags=all_tags,
        genre_estimate=genre_estimate,
        verified=False,
        verification_notes="",
        baseline_version=computed_baseline_version,
        reconstruction_protocol=computed_reconstruction_protocol,
        reading_history=[initial_reading],
    )

    # Update the index
    index = _idx._load_index()
    index["records"].append(asdict(record))
    index["total"] = len(index["records"])
    index["last_updated"] = deposited_at
    _idx._save_index(index)

    print(f"[SEEDBANK] Deposited: {filename} → {record_id}")
    # Log the birth event — the genome checkpoint from which all future forks descend.
    try:
        import seedbank.fork_log as _fork_log
        _fork_log.log_deposit(
            record_id=record_id,
            filename=filename,
            baseline_version=computed_baseline_version,
            tags=all_tags,
        )
    except Exception:
        pass
    return record


def _auto_tags(
    lsii_score: float,
    flag_level: str,
    authentic_emission: float,
    manufacturing_score: float,
    creative_residue: float = 0.0,
) -> List[str]:
    """
    Derive searchable tags automatically from analysis values.
    These supplement any depositor-provided tags.
    """
    tags = []
    if lsii_score >= 0.55:
        tags.append("high_lsii")
    if lsii_score <= 0.15:
        tags.append("consistent_arc")
    if authentic_emission >= 0.65:
        tags.append("high_authenticity")
    if manufacturing_score >= 0.65:
        tags.append("high_manufacturing")
    if creative_residue >= 0.5:
        tags.append("high_creative_residue")
    if flag_level in ("high", "extreme"):
        tags.append("late_inversion")
    return tags

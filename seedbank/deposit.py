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

    # Write the full profile to disk
    full_profile_filename = f"{record_id}.json"
    full_profile_path = os.path.join(records_dir, full_profile_filename)

    with open(full_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

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
    )

    # Update the index
    index = _idx._load_index()
    index["records"].append(asdict(record))
    index["total"] = len(index["records"])
    index["last_updated"] = deposited_at
    _idx._save_index(index)

    print(f"[SEEDBANK] Deposited: {filename} → {record_id}")
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

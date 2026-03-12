"""
KindPath Analyser :: Seedbank Index

Maintains the master index file (seedbank/index.json) that holds
summary data for every deposited record, enabling fast search without
loading full profiles from disk.

The index is rebuilt from disk records if missing or corrupted.
"""

import json
import os
from typing import Optional

# Paths are relative to the project root — wherever the analyser is run from.
# These can be overridden by setting KINDPATH_SEEDBANK_DIR in the environment.
_DEFAULT_SEEDBANK_DIR = os.path.join(os.path.dirname(__file__), "records")
_DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.json")

RECORDS_DIR: str = os.environ.get(
    "KINDPATH_SEEDBANK_DIR",
    _DEFAULT_SEEDBANK_DIR,
)
INDEX_PATH: str = os.environ.get(
    "KINDPATH_SEEDBANK_INDEX",
    _DEFAULT_INDEX_PATH,
)

_EMPTY_INDEX = {
    "total": 0,
    "last_updated": None,
    "records": [],
}


def _load_index() -> dict:
    """Load the index from disk, returning an empty index if missing or invalid."""
    if not os.path.exists(INDEX_PATH):
        return dict(_EMPTY_INDEX, records=[])
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "records" not in data:
            return dict(_EMPTY_INDEX, records=[])
        return data
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY_INDEX, records=[])


def _save_index(index: dict) -> None:
    """Write the index to disk atomically."""
    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    tmp_path = INDEX_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, INDEX_PATH)


def rebuild_index() -> dict:
    """
    Reconstruct the index by scanning all JSON files in RECORDS_DIR.

    Useful for recovery after manual edits to the records directory,
    or after migrating records from another instance.

    Returns the rebuilt index dict.
    """
    os.makedirs(RECORDS_DIR, exist_ok=True)
    records = []

    for fname in sorted(os.listdir(RECORDS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RECORDS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # A full profile — extract summary fields to rebuild index entry
            source = data.get("source", {})
            lsii = data.get("lsii", {})
            psychosomatic = data.get("psychosomatic", {})
            fingerprints = data.get("fingerprints", {})
            era_matches = fingerprints.get("era_matches", [])

            record_entry = {
                "id": fname.replace(".json", ""),
                "deposited_at": data.get("deposited_at", ""),
                "filename": source.get("filename", fname),
                "context": data.get("context", ""),
                "release_circumstances": data.get("release_circumstances"),
                "creator_statement": data.get("creator_statement"),
                "duration_seconds": float(source.get("duration_seconds", 0.0)),
                "lsii_score": float(lsii.get("lsii_score", 0.0)),
                "lsii_flag_level": str(lsii.get("flag_level", "none")),
                "authentic_emission_score": float(psychosomatic.get("authentic_emission_score", 0.0)),
                "manufacturing_score": float(psychosomatic.get("manufacturing_score", 0.0)),
                "creative_residue": float(psychosomatic.get("creative_residue", 0.0)),
                "era_fingerprint": era_matches[0].get("name", "unknown") if era_matches else "unknown",
                "key_estimate": "",
                "tempo_bpm": 0.0,
                "full_profile_path": path,
                "tags": data.get("tags", []),
                "genre_estimate": fingerprints.get("production_context", ""),
                "verified": data.get("verified", False),
                "verification_notes": data.get("verification_notes", ""),
            }
            records.append(record_entry)
        except (json.JSONDecodeError, OSError, KeyError):
            continue  # Skip corrupt records

    from datetime import datetime, timezone
    index = {
        "total": len(records),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    _save_index(index)
    print(f"[SEEDBANK] Index rebuilt: {len(records)} records")
    return index


def get_stats() -> dict:
    """
    Return aggregate statistics about the seedbank corpus:
    - Total record count
    - Distribution by era fingerprint
    - Distribution by LSII flag level
    - Average LSII score
    - Average authentic_emission_score
    - Most common tags
    """
    index = _load_index()
    records = index.get("records", [])

    if not records:
        return {
            "total": 0,
            "era_distribution": {},
            "flag_distribution": {},
            "avg_lsii": 0.0,
            "avg_authentic_emission": 0.0,
            "avg_manufacturing": 0.0,
            "top_tags": [],
        }

    era_dist: dict[str, int] = {}
    flag_dist: dict[str, int] = {}
    lsii_sum = 0.0
    auth_sum = 0.0
    mfg_sum = 0.0
    tag_counts: dict[str, int] = {}

    for r in records:
        era = r.get("era_fingerprint", "unknown")
        flag = r.get("lsii_flag_level", "none")
        era_dist[era] = era_dist.get(era, 0) + 1
        flag_dist[flag] = flag_dist.get(flag, 0) + 1
        lsii_sum += float(r.get("lsii_score", 0.0))
        auth_sum += float(r.get("authentic_emission_score", 0.0))
        mfg_sum += float(r.get("manufacturing_score", 0.0))
        for tag in r.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    n = len(records)
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total": n,
        "era_distribution": era_dist,
        "flag_distribution": flag_dist,
        "avg_lsii": round(lsii_sum / n, 4),
        "avg_authentic_emission": round(auth_sum / n, 4),
        "avg_manufacturing": round(mfg_sum / n, 4),
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
    }

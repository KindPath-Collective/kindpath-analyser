"""
KindPath Analyser :: Seedbank Query

Search and compare across deposited profiles. The seedbank becomes
useful the moment there are enough records to derive baselines.

Every search is a way to contextualise a single piece against the broader
authenticated record. Every comparison is an act of perspective.
"""

import json
import os
from typing import Optional, List
from dataclasses import dataclass
import seedbank.index as _idx


@dataclass
class SearchResult:
    """A lightweight search result row — the index entry, not the full profile."""
    id: str
    filename: str
    context: str
    lsii_score: float
    lsii_flag_level: str
    authentic_emission_score: float
    manufacturing_score: float
    creative_residue: float
    era_fingerprint: str
    duration_seconds: float
    tags: List[str]


def _record_to_result(r: dict) -> SearchResult:
    return SearchResult(
        id=r.get("id", ""),
        filename=r.get("filename", ""),
        context=r.get("context", ""),
        lsii_score=float(r.get("lsii_score", 0.0)),
        lsii_flag_level=r.get("lsii_flag_level", "none"),
        authentic_emission_score=float(r.get("authentic_emission_score", 0.0)),
        manufacturing_score=float(r.get("manufacturing_score", 0.0)),
        creative_residue=float(r.get("creative_residue", 0.0)),
        era_fingerprint=r.get("era_fingerprint", "unknown"),
        duration_seconds=float(r.get("duration_seconds", 0.0)),
        tags=r.get("tags", []),
    )


def search(
    lsii_min: Optional[float] = None,
    lsii_max: Optional[float] = None,
    era: Optional[str] = None,
    authentic_emission_min: Optional[float] = None,
    manufacturing_max: Optional[float] = None,
    tags: Optional[List[str]] = None,
    text: Optional[str] = None,
    limit: int = 20,
) -> List[SearchResult]:
    """
    Search the seedbank by any combination of parameters.

    All parameters are optional and additive (AND logic).
    Returns a list of SearchResult ordered by LSII score descending.
    """
    index = _idx._load_index()
    records = index.get("records", [])
    results = []

    for r in records:
        if lsii_min is not None and float(r.get("lsii_score", 0.0)) < lsii_min:
            continue
        if lsii_max is not None and float(r.get("lsii_score", 0.0)) > lsii_max:
            continue
        if era is not None and era.lower() not in r.get("era_fingerprint", "").lower():
            continue
        if authentic_emission_min is not None:
            if float(r.get("authentic_emission_score", 0.0)) < authentic_emission_min:
                continue
        if manufacturing_max is not None:
            if float(r.get("manufacturing_score", 0.0)) > manufacturing_max:
                continue
        if tags:
            record_tags = set(r.get("tags", []))
            if not all(t in record_tags for t in tags):
                continue
        if text:
            search_text = text.lower()
            combined = " ".join([
                r.get("filename", ""),
                r.get("context", ""),
                r.get("creator_statement", "") or "",
                r.get("era_fingerprint", ""),
                r.get("genre_estimate", ""),
            ]).lower()
            if search_text not in combined:
                continue

        results.append(_record_to_result(r))

    results.sort(key=lambda r: r.lsii_score, reverse=True)
    return results[:limit]


def get_baseline(era: Optional[str] = None, genre: Optional[str] = None) -> dict:
    """
    Compute average feature values across matching seedbank records.

    If era or genre is provided, filter to matching records only.
    Returns a dict of feature_name -> mean_value for use as a comparison baseline.
    An empty dict is returned if there are no matching records.
    """
    index = _idx._load_index()
    records = index.get("records", [])

    matching = []
    for r in records:
        if era and era.lower() not in r.get("era_fingerprint", "").lower():
            continue
        if genre and genre.lower() not in r.get("genre_estimate", "").lower():
            continue
        matching.append(r)

    if not matching:
        return {}

    n = len(matching)
    fields = ["lsii_score", "authentic_emission_score", "manufacturing_score", "creative_residue"]
    baseline = {f: round(sum(float(r.get(f, 0.0)) for r in matching) / n, 4) for f in fields}
    baseline["sample_size"] = n
    return baseline


def compare(profile_id: str, target_id: str) -> dict:
    """
    Compare two seedbank records directly.

    Returns a dict of feature axes with the delta value (target - source)
    and a plain-language direction note for each.
    """
    index = _idx._load_index()
    records = {r["id"]: r for r in index.get("records", [])}

    if profile_id not in records:
        raise KeyError(f"Record not found: {profile_id}")
    if target_id not in records:
        raise KeyError(f"Record not found: {target_id}")

    source = records[profile_id]
    target = records[target_id]

    axes = ["lsii_score", "authentic_emission_score", "manufacturing_score", "creative_residue"]
    comparison = {}

    for axis in axes:
        sv = float(source.get(axis, 0.0))
        tv = float(target.get(axis, 0.0))
        delta = round(tv - sv, 4)
        direction = "higher" if delta > 0.05 else ("lower" if delta < -0.05 else "similar")
        comparison[axis] = {
            "source": sv,
            "target": tv,
            "delta": delta,
            "direction": direction,
        }

    return {
        "source_id": profile_id,
        "target_id": target_id,
        "source_filename": source.get("filename", ""),
        "target_filename": target.get("filename", ""),
        "axes": comparison,
    }


def get_most_authentic(limit: int = 10) -> List[SearchResult]:
    """Return the top N records by authentic_emission_score."""
    index = _idx._load_index()
    records = index.get("records", [])
    sorted_records = sorted(records, key=lambda r: float(r.get("authentic_emission_score", 0.0)), reverse=True)
    return [_record_to_result(r) for r in sorted_records[:limit]]


def get_highest_lsii(limit: int = 10) -> List[SearchResult]:
    """Return the top N records by LSII score."""
    index = _idx._load_index()
    records = index.get("records", [])
    sorted_records = sorted(records, key=lambda r: float(r.get("lsii_score", 0.0)), reverse=True)
    return [_record_to_result(r) for r in sorted_records[:limit]]


def load_full_profile(record_id: str) -> dict:
    """
    Load the complete analysis profile JSON for a given record ID.
    Returns the full profile dict as stored at deposit time.
    Raises FileNotFoundError if the record file does not exist.
    """
    index = _idx._load_index()
    records = {r["id"]: r for r in index.get("records", [])}

    if record_id not in records:
        raise KeyError(f"Record not found in index: {record_id}")

    profile_path = records[record_id].get("full_profile_path", "")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file missing: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)

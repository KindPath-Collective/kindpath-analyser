"""
Tests for seedbank/ — deposit, query, and index.

All tests use real JSON data: no mocks.
Deposits write to a temporary directory (isolated per test run).
Tests verify real write/read round-trips.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import tempfile
import shutil
import pytest

# Import submodules directly — avoids any re-export name-shadowing in __init__
import importlib
import seedbank.index
import seedbank.deposit as _dep_pkg  # noqa: ensure module is in sys.modules
import seedbank.index as idx_module
dep_module = importlib.import_module('seedbank.deposit')  # the real module

from seedbank.deposit import deposit, SeedbankRecord, _auto_tags
from seedbank.query import (
    search, get_baseline, compare, get_most_authentic,
    get_highest_lsii, load_full_profile,
)
from seedbank.index import rebuild_index, get_stats


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_profile(
    filename: str = "test.wav",
    lsii_score: float = 0.3,
    authentic_emission: float = 0.6,
    manufacturing_score: float = 0.3,
    era: str = "2010s",
    flag_level: str = "low",
) -> dict:
    """Minimal but structurally complete JSON profile (no mocks)."""
    return {
        "source": {
            "filename": filename,
            "duration_seconds": 180.0,
            "sample_rate": 44100,
            "num_channels": 2,
        },
        "lsii": {
            "lsii_score": lsii_score,
            "flag_level": flag_level,
            "direction": "darker",
            "dominant_axis": "spectral",
        },
        "psychosomatic": {
            "valence": 0.1,
            "arousal": 0.5,
            "authentic_emission_score": authentic_emission,
            "manufacturing_score": manufacturing_score,
            "creative_residue": 0.4,
            "elder_reading": "A real elder reading here.",
        },
        "fingerprints": {
            "era_matches": [{"name": era, "confidence": 0.8, "description": ""}],
            "technique_matches": [],
            "instrument_matches": [],
            "production_context": "independent release",
            "authenticity_markers": [],
            "manufacturing_markers": [],
        },
        "arcs": {
            "valence": [0.1, 0.2, 0.1, -0.1],
            "energy": [0.5, 0.6, 0.5, 0.4],
        },
    }


@pytest.fixture(autouse=True)
def isolated_seedbank(tmp_path):
    """
    Redirect the seedbank to a temp directory for each test.
    Cleans up automatically after every test.
    No test data pollutes the real seedbank.
    """
    records_dir = str(tmp_path / "records")
    index_path = str(tmp_path / "index.json")
    os.makedirs(records_dir)

    original_records = idx_module.RECORDS_DIR
    original_index = idx_module.INDEX_PATH

    idx_module.RECORDS_DIR = records_dir
    idx_module.INDEX_PATH = index_path

    # Patch deposit.py references
    import seedbank.deposit as dep_module
    dep_module.RECORDS_DIR = records_dir
    dep_module.INDEX_PATH = index_path

    yield tmp_path

    idx_module.RECORDS_DIR = original_records
    idx_module.INDEX_PATH = original_index
    dep_module.RECORDS_DIR = original_records
    dep_module.INDEX_PATH = original_index


# ── Deposit Tests ────────────────────────────────────────────────────────────

class TestDeposit:

    def test_deposit_returns_record(self):
        profile = _make_profile()
        record = deposit(profile, context="Test piece, independent.")
        assert isinstance(record, SeedbankRecord)

    def test_deposit_assigns_uuid(self):
        import re
        record = deposit(_make_profile(), context="Test.")
        assert re.match(r'^[0-9a-f\-]{36}$', record.id)

    def test_deposit_writes_json_file(self, isolated_seedbank):
        record = deposit(_make_profile(), context="Test.")
        assert os.path.exists(record.full_profile_path)

    def test_deposit_file_contains_full_profile(self, isolated_seedbank):
        profile = _make_profile(filename="my_track.wav")
        record = deposit(profile, context="Test.")
        with open(record.full_profile_path) as f:
            saved = json.load(f)
        assert saved["source"]["filename"] == "my_track.wav"

    def test_deposit_updates_index(self, isolated_seedbank):
        deposit(_make_profile(), context="Test one.")
        deposit(_make_profile(filename="two.wav"), context="Test two.")
        index = idx_module._load_index()
        assert index["total"] == 2

    def test_deposit_extracts_lsii_score(self):
        record = deposit(_make_profile(lsii_score=0.72), context="High LSII test.")
        assert abs(record.lsii_score - 0.72) < 0.001

    def test_deposit_extracts_filename(self):
        record = deposit(_make_profile(filename="special_track.wav"), context="Named track.")
        assert record.filename == "special_track.wav"

    def test_deposit_extracts_era(self):
        record = deposit(_make_profile(era="1990s"), context="90s piece.")
        assert record.era_fingerprint == "1990s"

    def test_deposit_auto_tags_high_lsii(self):
        record = deposit(_make_profile(lsii_score=0.8, flag_level="high"), context="High LSII.")
        assert "high_lsii" in record.tags
        assert "late_inversion" in record.tags

    def test_deposit_auto_tags_high_authenticity(self):
        record = deposit(_make_profile(authentic_emission=0.8), context="Authentic.")
        assert "high_authenticity" in record.tags

    def test_deposit_merges_custom_tags(self):
        record = deposit(_make_profile(), context="Tagged.", tags=["custom_tag"])
        assert "custom_tag" in record.tags

    def test_deposit_stores_context(self):
        record = deposit(_make_profile(), context="Original independent release, 2021.")
        assert "independent" in record.context

    def test_deposit_stores_release_circumstances(self):
        record = deposit(
            _make_profile(), context="Test.",
            release_circumstances="Self-released, no label."
        )
        assert record.release_circumstances == "Self-released, no label."

    def test_deposit_stores_creator_statement(self):
        record = deposit(
            _make_profile(), context="Test.",
            creator_statement="I wrote this in grief."
        )
        assert record.creator_statement == "I wrote this in grief."


# ── Auto-tag Tests ────────────────────────────────────────────────────────────

class TestAutoTags:

    def test_low_lsii_no_high_lsii_tag(self):
        tags = _auto_tags(0.1, "none", 0.5, 0.3)
        assert "high_lsii" not in tags

    def test_consistent_arc_tag_at_low_lsii(self):
        tags = _auto_tags(0.05, "none", 0.5, 0.3)
        assert "consistent_arc" in tags

    def test_high_manufacturing_tag(self):
        tags = _auto_tags(0.2, "low", 0.4, 0.8)
        assert "high_manufacturing" in tags


# ── Query Tests ───────────────────────────────────────────────────────────────

class TestSearch:

    def _deposit_batch(self):
        """Deposit three real records with known properties."""
        deposit(_make_profile("a.wav", lsii_score=0.8, authentic_emission=0.7, era="2000s"), context="High LSII, authentic.")
        deposit(_make_profile("b.wav", lsii_score=0.2, authentic_emission=0.3, manufacturing_score=0.8, era="2010s"), context="Low LSII, manufactured.")
        deposit(_make_profile("c.wav", lsii_score=0.5, authentic_emission=0.6, era="1990s"), context="Mid LSII, 90s.")

    def test_search_returns_list(self):
        self._deposit_batch()
        results = search()
        assert isinstance(results, list)

    def test_search_no_filter_returns_all(self):
        self._deposit_batch()
        results = search(limit=100)
        assert len(results) == 3

    def test_search_lsii_min_filter(self):
        self._deposit_batch()
        results = search(lsii_min=0.6)
        assert all(r.lsii_score >= 0.6 for r in results)
        assert len(results) == 1

    def test_search_lsii_max_filter(self):
        self._deposit_batch()
        results = search(lsii_max=0.3)
        assert all(r.lsii_score <= 0.3 for r in results)

    def test_search_era_filter(self):
        self._deposit_batch()
        results = search(era="1990s")
        assert len(results) == 1
        assert results[0].era_fingerprint == "1990s"

    def test_search_authentic_min_filter(self):
        self._deposit_batch()
        results = search(authentic_emission_min=0.6)
        assert all(r.authentic_emission_score >= 0.6 for r in results)

    def test_search_manufacturing_max_filter(self):
        self._deposit_batch()
        results = search(manufacturing_max=0.4)
        assert all(r.manufacturing_score <= 0.4 for r in results)

    def test_search_text_filter(self):
        self._deposit_batch()
        results = search(text="authentic")
        assert len(results) >= 1

    def test_search_tag_filter(self):
        self._deposit_batch()
        results = search(tags=["high_lsii"])
        assert all("high_lsii" in r.tags for r in results)

    def test_search_respects_limit(self):
        self._deposit_batch()
        results = search(limit=2)
        assert len(results) <= 2

    def test_search_empty_seedbank(self):
        results = search()
        assert results == []

    def test_search_sorted_by_lsii_descending(self):
        self._deposit_batch()
        results = search()
        scores = [r.lsii_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Baseline Tests ────────────────────────────────────────────────────────────

class TestBaseline:

    def test_baseline_returns_dict(self):
        deposit(_make_profile(era="2000s"), context="Test.")
        baseline = get_baseline(era="2000s")
        assert isinstance(baseline, dict)

    def test_baseline_contains_avg_fields(self):
        deposit(_make_profile(lsii_score=0.4), context="Test.")
        deposit(_make_profile(lsii_score=0.6), context="Test 2.")
        baseline = get_baseline()
        assert "lsii_score" in baseline
        assert abs(baseline["lsii_score"] - 0.5) < 0.01

    def test_baseline_no_match_returns_empty(self):
        baseline = get_baseline(era="medieval")
        assert baseline == {}

    def test_baseline_sample_size_correct(self):
        deposit(_make_profile(era="2000s"), context="Test 1.")
        deposit(_make_profile(era="2000s"), context="Test 2.")
        baseline = get_baseline(era="2000s")
        assert baseline["sample_size"] == 2


# ── Compare Tests ──────────────────────────────────────────────────────────────

class TestCompare:

    def test_compare_returns_axes(self):
        r1 = deposit(_make_profile(lsii_score=0.3, authentic_emission=0.6), context="One.")
        r2 = deposit(_make_profile(lsii_score=0.7, authentic_emission=0.8), context="Two.")
        result = compare(r1.id, r2.id)
        assert "axes" in result
        assert "lsii_score" in result["axes"]

    def test_compare_delta_direction(self):
        r1 = deposit(_make_profile(lsii_score=0.3), context="Low.")
        r2 = deposit(_make_profile(lsii_score=0.7), context="High.")
        result = compare(r1.id, r2.id)
        assert result["axes"]["lsii_score"]["delta"] > 0
        assert result["axes"]["lsii_score"]["direction"] == "higher"

    def test_compare_missing_id_raises(self):
        r1 = deposit(_make_profile(), context="One.")
        with pytest.raises(KeyError):
            compare(r1.id, "nonexistent-id")


# ── Index Tests ───────────────────────────────────────────────────────────────

class TestIndex:

    def test_rebuild_from_empty(self, isolated_seedbank):
        index = rebuild_index()
        assert index["total"] == 0

    def test_rebuild_finds_deposited_records(self, isolated_seedbank):
        # Deposit directly by writing a valid profile JSON to records dir
        profile = _make_profile("rebuild_test.wav")
        deposit(profile, context="Rebuild test.")
        # Now rebuild from scratch
        result = rebuild_index()
        assert result["total"] >= 1

    def test_stats_on_empty_seedbank(self):
        stats = get_stats()
        assert stats["total"] == 0

    def test_stats_returns_correct_total(self):
        deposit(_make_profile(), context="One.")
        deposit(_make_profile(), context="Two.")
        stats = get_stats()
        assert stats["total"] == 2

    def test_stats_era_distribution(self):
        deposit(_make_profile(era="2000s"), context="2000s track.")
        deposit(_make_profile(era="2000s"), context="Another 2000s track.")
        deposit(_make_profile(era="1990s"), context="90s track.")
        stats = get_stats()
        assert stats["era_distribution"].get("2000s") == 2
        assert stats["era_distribution"].get("1990s") == 1

    def test_stats_avg_lsii(self):
        deposit(_make_profile(lsii_score=0.4), context="Low.")
        deposit(_make_profile(lsii_score=0.6), context="High.")
        stats = get_stats()
        assert abs(stats["avg_lsii"] - 0.5) < 0.01


# ── Load Full Profile Tests ───────────────────────────────────────────────────

class TestLoadFullProfile:

    def test_load_returns_original_profile(self):
        profile = _make_profile(filename="roundtrip.wav")
        record = deposit(profile, context="Roundtrip test.")
        loaded = load_full_profile(record.id)
        assert loaded["source"]["filename"] == "roundtrip.wav"

    def test_load_missing_id_raises_key_error(self):
        with pytest.raises(KeyError):
            load_full_profile("does-not-exist")

    def test_load_preserves_lsii_score(self):
        profile = _make_profile(lsii_score=0.789)
        record = deposit(profile, context="LSII roundtrip.")
        loaded = load_full_profile(record.id)
        assert abs(loaded["lsii"]["lsii_score"] - 0.789) < 0.001


# ── Ranking Tests ─────────────────────────────────────────────────────────────

class TestRankings:

    def _deposit_varied(self):
        deposit(_make_profile("a.wav", lsii_score=0.9, authentic_emission=0.2), context="High LSII.")
        deposit(_make_profile("b.wav", lsii_score=0.1, authentic_emission=0.9), context="High auth.")
        deposit(_make_profile("c.wav", lsii_score=0.5, authentic_emission=0.5), context="Mid.")

    def test_get_most_authentic_sorted(self):
        self._deposit_varied()
        results = get_most_authentic()
        scores = [r.authentic_emission_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_highest_lsii_sorted(self):
        self._deposit_varied()
        results = get_highest_lsii()
        scores = [r.lsii_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_most_authentic_respects_limit(self):
        self._deposit_varied()
        results = get_most_authentic(limit=2)
        assert len(results) <= 2

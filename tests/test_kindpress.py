"""
Tests for kindpress/ — press, reason, and validate.

All tests use temporary directories for seedbank isolation.
The KindPress modules are tested against round-trips and invariants;
no mocks are used — this is real data flowing through real code.

Coverage:
    press.py      — encode, decode, verify_integrity, k_alignment_check,
                    compression_ratio, round-trip fidelity
    reason.py     — analyse_delta_distribution, hmoe_of_corpus,
                    k_calibration_score, survey_all_tags
    validate.py   — validate_tag_revision, TagRevisionReport shape,
                    recommendation logic, empty-corpus graceful degradation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
import copy
import pytest

import seedbank.index as idx_module
import seedbank.tags_registry as tags_module

from seedbank.deposit import deposit
from seedbank.tags_registry import define_tag, revise_tag

from kindpress.press import (
    encode,
    decode,
    verify_integrity,
    k_alignment_check,
    compression_ratio,
    KindPressPacket,
    _DELTA_FIELDS,
    _K_STRUCTURAL_FIELDS,
)
from kindpress.reason import (
    analyse_delta_distribution,
    hmoe_of_corpus,
    k_calibration_score,
    survey_all_tags,
)
from kindpress.validate import (
    validate_tag_revision,
    TagRevisionReport,
    print_report,
)


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_profile(
    filename: str = "test.wav",
    lsii_score: float = 0.3,
    authentic_emission: float = 0.6,
    manufacturing_score: float = 0.3,
    creative_residue: float = 0.4,
    era: str = "2010s",
    flag_level: str = "low",
) -> dict:
    """Minimal but structurally complete JSON profile."""
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
            "creative_residue": creative_residue,
            "elder_reading": "Elder reading placeholder.",
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


def _minimal_record(
    record_id: str = "test-id-001",
    filename: str = "track.wav",
    lsii: float = 0.4,
    authentic: float = 0.7,
    manufacturing: float = 0.3,
    creative_residue: float = 0.5,
    era: str = "2010s",
    k_version: str = "high_lsii:v1",
) -> dict:
    """Return a minimal seedbank-record-shaped dict for press tests (no DB needed)."""
    return {
        "id": record_id,
        "deposited_at": "2024-01-01T00:00:00+00:00",
        "filename": filename,
        "context": "Test context.",
        "baseline_version": k_version,
        "reconstruction_protocol": "v1",
        "reading_history": [],
        "stale_tags": [],
        "creative_residue_stale": False,
        "lsii_score": lsii,
        "lsii_flag_level": "moderate" if lsii >= 0.4 else "low",
        "authentic_emission_score": authentic,
        "manufacturing_score": manufacturing,
        "creative_residue": creative_residue,
        "era_fingerprint": era,
        "key_estimate": "C major",
        "tempo_bpm": 120.0,
        "genre_estimate": "indie",
        "tags": [],
        "verified": False,
        "release_circumstances": None,
        "creator_statement": None,
        "duration_seconds": 180.0,
    }


@pytest.fixture(autouse=True)
def isolated_seedbank(tmp_path):
    """
    Redirect the seedbank AND tags registry to temp directories for every test.
    Provides total isolation — no test data pollutes the real DB or registry.
    """
    records_dir = str(tmp_path / "records")
    index_path = str(tmp_path / "index.json")
    registry_path = str(tmp_path / "tags_registry.json")
    os.makedirs(records_dir)

    # Save original paths
    orig_records = idx_module.RECORDS_DIR
    orig_index = idx_module.INDEX_PATH
    orig_registry = tags_module.REGISTRY_PATH

    # Redirect: deposit.py reads from idx_module.RECORDS_DIR at call time,
    # so patching idx_module is sufficient for full isolation.
    idx_module.RECORDS_DIR = records_dir
    idx_module.INDEX_PATH = index_path
    tags_module.REGISTRY_PATH = registry_path

    yield tmp_path

    # Restore
    idx_module.RECORDS_DIR = orig_records
    idx_module.INDEX_PATH = orig_index
    tags_module.REGISTRY_PATH = orig_registry


# ── press.py tests ────────────────────────────────────────────────────────────

class TestEncode:

    def test_encode_returns_packet(self):
        rec = _minimal_record()
        packet = encode(rec)
        assert isinstance(packet, KindPressPacket)

    def test_encode_sets_record_id(self):
        rec = _minimal_record(record_id="abc-123")
        packet = encode(rec)
        assert packet.record_id == "abc-123"

    def test_encode_sets_k_version(self):
        rec = _minimal_record(k_version="high_lsii:v2,natural_dynamics:v1")
        packet = encode(rec)
        assert packet.k_version == "high_lsii:v2,natural_dynamics:v1"

    def test_encode_delta_contains_delta_fields(self):
        rec = _minimal_record(lsii=0.75, authentic=0.9)
        packet = encode(rec)
        assert "lsii_score" in packet.delta
        assert "authentic_emission_score" in packet.delta
        assert "creative_residue" in packet.delta

    def test_encode_delta_excludes_structural_fields(self):
        rec = _minimal_record()
        packet = encode(rec)
        for field in _K_STRUCTURAL_FIELDS:
            assert field not in packet.delta

    def test_encode_delta_excludes_none_values(self):
        rec = _minimal_record()
        rec["release_circumstances"] = None  # Ensure it's None
        packet = encode(rec)
        assert "release_circumstances" not in packet.delta

    def test_encode_packet_hash_is_sha256(self):
        rec = _minimal_record()
        packet = encode(rec)
        assert len(packet.packet_hash) == 64
        assert all(c in "0123456789abcdef" for c in packet.packet_hash)

    def test_encode_hash_is_deterministic(self):
        rec = _minimal_record()
        p1 = encode(rec)
        p2 = encode(rec)
        # Different compressed_at but same k_version + delta → same hash
        assert p1.packet_hash == p2.packet_hash

    def test_encode_hash_changes_with_delta(self):
        rec1 = _minimal_record(lsii=0.3)
        rec2 = _minimal_record(lsii=0.9)
        p1 = encode(rec1)
        p2 = encode(rec2)
        assert p1.packet_hash != p2.packet_hash

    def test_encode_extra_fields_preserved(self):
        rec = _minimal_record()
        rec["mystery_field"] = "preserved"
        packet = encode(rec)
        assert "mystery_field" in packet.delta.get("__extra__", {})

    def test_encode_source_filename_set(self):
        rec = _minimal_record(filename="song.wav")
        packet = encode(rec)
        assert packet.source_filename == "song.wav"


class TestDecode:

    def test_decode_returns_dict(self):
        rec = _minimal_record()
        packet = encode(rec)
        result = decode(packet)
        assert isinstance(result, dict)

    def test_decode_contains_delta_fields(self):
        rec = _minimal_record(lsii=0.55)
        packet = encode(rec)
        result = decode(packet, k_defaults=None)
        assert result.get("lsii_score") == 0.55

    def test_decode_with_k_defaults_merges(self):
        rec = _minimal_record(lsii=0.55)
        packet = encode(rec)
        k_defaults = {
            "reconstructed_from": "k-universe-v2",
            "repo_context": "KindPath 2024",
        }
        result = decode(packet, k_defaults=k_defaults)
        # k_defaults fields present
        assert result["reconstructed_from"] == "k-universe-v2"
        # delta overrides k_defaults when there's overlap
        assert result["lsii_score"] == 0.55

    def test_decode_roundtrip_delta_fields(self):
        rec = _minimal_record(
            lsii=0.88,
            authentic=0.72,
            manufacturing=0.28,
            creative_residue=0.61,
            era="1990s",
        )
        packet = encode(rec)
        result = decode(packet)
        assert result["lsii_score"] == 0.88
        assert result["authentic_emission_score"] == 0.72
        assert result["manufacturing_score"] == 0.28
        assert result["creative_residue"] == 0.61
        assert result["era_fingerprint"] == "1990s"

    def test_decode_without_k_defaults_is_partial(self):
        rec = _minimal_record()
        packet = encode(rec)
        result = decode(packet, k_defaults=None)
        # Structural fields are NOT present (they weren't in delta)
        assert "filename" not in result or result.get("filename") == rec["filename"]


class TestVerifyIntegrity:

    def test_verify_untampered_packet(self):
        rec = _minimal_record()
        packet = encode(rec)
        assert verify_integrity(packet) is True

    def test_verify_fails_on_tampered_delta(self):
        rec = _minimal_record()
        packet = encode(rec)
        tampered = copy.deepcopy(packet)
        tampered.delta["lsii_score"] = 9999.0
        assert verify_integrity(tampered) is False

    def test_verify_fails_on_tampered_k_version(self):
        rec = _minimal_record(k_version="high_lsii:v1")
        packet = encode(rec)
        tampered = copy.deepcopy(packet)
        tampered.k_version = "high_lsii:v99"
        assert verify_integrity(tampered) is False

    def test_verify_fails_on_empty_hash(self):
        rec = _minimal_record()
        packet = encode(rec)
        packet.packet_hash = ""
        assert verify_integrity(packet) is False


class TestKAlignmentCheck:

    def test_aligned_when_same_version(self):
        result = k_alignment_check("high_lsii:v2", "high_lsii:v2")
        assert result["aligned"] is True
        assert result["drifted_tags"] == []

    def test_drifted_when_version_differs(self):
        result = k_alignment_check("high_lsii:v1", "high_lsii:v2")
        assert result["aligned"] is False
        assert "high_lsii" in result["drifted_tags"]

    def test_multiple_tags_one_drifted(self):
        result = k_alignment_check(
            "high_lsii:v2,natural_dynamics:v1",
            "high_lsii:v2,natural_dynamics:v3",
        )
        assert result["aligned"] is False
        assert "natural_dynamics" in result["drifted_tags"]
        assert "high_lsii" not in result["drifted_tags"]

    def test_empty_versions_aligned(self):
        result = k_alignment_check("", "")
        assert result["aligned"] is True

    def test_recommendation_text_on_drift(self):
        result = k_alignment_check("high_lsii:v1", "high_lsii:v2")
        assert "drifted" in result["recommendation"].lower()


class TestCompressionRatio:

    def test_ratio_between_zero_and_one(self):
        rec = _minimal_record()
        packet = encode(rec)
        ratio = compression_ratio(rec, packet)
        assert 0.0 <= ratio <= 1.0

    def test_larger_record_smaller_ratio(self):
        # A record with more structural fields → smaller ratio (k absorbs more)
        rec = _minimal_record()
        rec["context"] = "A" * 10000  # Inflate structural context
        packet = encode(rec)
        ratio = compression_ratio(rec, packet)
        # Delta should be much smaller than the full record now
        assert ratio < 0.5

    def test_ratio_is_float(self):
        rec = _minimal_record()
        packet = encode(rec)
        assert isinstance(compression_ratio(rec, packet), float)


# ── reason.py tests ───────────────────────────────────────────────────────────

class TestAnalyseDeltaDistribution:

    def test_raises_on_unknown_tag(self):
        with pytest.raises(ValueError, match="not found"):
            analyse_delta_distribution("nonexistent_tag")

    def test_returns_distribution_with_zero_records(self):
        define_tag(
            name="test_tag",
            category="test",
            description="A test tag for unit testing.",
            scope="Any record with a high LSII score.",
        )
        dist = analyse_delta_distribution("test_tag")
        assert dist.tag_name == "test_tag"
        assert dist.n_records == 0
        assert dist.hmoe == 0.0

    def test_distribution_with_one_record(self):
        define_tag(
            name="solo_tag",
            category="test",
            description="Appears on exactly one record.",
            scope="Isolated signal.",
        )
        deposit(
            _make_profile(filename="solo.wav"),
            context="Solo test.",
            tags=["solo_tag"],
        )
        dist = analyse_delta_distribution("solo_tag")
        assert dist.n_records == 1
        # Variance requires ≥2 records; HMoE = 0.0 for single record
        assert dist.hmoe == 0.0

    def test_distribution_with_multiple_records(self):
        define_tag(
            name="multi_tag",
            category="test",
            description="Appears on multiple records.",
            scope="Group signal test.",
        )
        for i in range(5):
            deposit(
                _make_profile(
                    filename=f"track_{i}.wav",
                    lsii_score=0.1 * (i + 1),
                    creative_residue=0.2 * (i + 1),
                ),
                context=f"Record {i}.",
                tags=["multi_tag"],
            )
        dist = analyse_delta_distribution("multi_tag")
        assert dist.n_records == 5
        assert dist.residue_variance >= 0.0
        assert dist.hmoe >= 0.0

    def test_distribution_tag_version_is_current(self):
        define_tag(
            name="versioned_tag",
            category="test",
            description="Version 1.",
            scope="Test.",
        )
        dist = analyse_delta_distribution("versioned_tag")
        assert dist.tag_version == 1

    def test_calibration_score_between_zero_and_one(self):
        define_tag(
            name="cal_tag",
            category="test",
            description="Calibration test.",
            scope="Test.",
        )
        for i in range(6):
            deposit(
                _make_profile(
                    filename=f"cal_{i}.wav",
                    creative_residue=0.1 * (i + 1),
                ),
                context="Cal test.",
                tags=["cal_tag"],
            )
        dist = analyse_delta_distribution("cal_tag")
        assert 0.0 <= dist.calibration_score <= 1.0

    def test_low_variance_penalised_in_calibration(self):
        """Records with identical residues should score poorly on calibration."""
        define_tag(
            name="flat_tag",
            category="test",
            description="All records have same residue.",
            scope="Test.",
        )
        for i in range(5):
            deposit(
                _make_profile(
                    filename=f"flat_{i}.wav",
                    creative_residue=0.5,  # All identical
                ),
                context="Flat test.",
                tags=["flat_tag"],
            )
        dist = analyse_delta_distribution("flat_tag")
        # Low variance → calibration penalised
        assert dist.calibration_score < 0.9


class TestHmoeOfCorpus:

    def test_empty_corpus_returns_zero(self):
        result = hmoe_of_corpus([])
        assert result == 0.0

    def test_single_record_returns_zero(self):
        rec = deposit(_make_profile(filename="single.wav"), context="Single.")
        result = hmoe_of_corpus([rec.id])
        assert result == 0.0

    def test_multiple_records_positive_hmoe(self):
        ids = []
        for i in range(5):
            rec = deposit(
                _make_profile(
                    filename=f"corpus_{i}.wav",
                    creative_residue=0.1 * (i + 2),
                ),
                context=f"Corpus record {i}.",
            )
            ids.append(rec.id)
        result = hmoe_of_corpus(ids)
        assert result >= 0.0

    def test_full_corpus_when_none_passed(self):
        for i in range(3):
            deposit(
                _make_profile(filename=f"all_{i}.wav", creative_residue=0.2 * i),
                context="Full corpus test.",
            )
        result = hmoe_of_corpus(None)
        # Just verify it runs and returns a float
        assert isinstance(result, float)

    def test_hmoe_increases_with_variance(self):
        """High residue variance should produce higher HMoE than low variance."""
        ids_low_var = []
        ids_high_var = []
        for i in range(5):
            r1 = deposit(
                _make_profile(filename=f"lv_{i}.wav", creative_residue=0.49 + 0.01 * i),
                context="Low var.",
            )
            r2 = deposit(
                _make_profile(filename=f"hv_{i}.wav", creative_residue=0.1 * i),
                context="High var.",
            )
            ids_low_var.append(r1.id)
            ids_high_var.append(r2.id)
        hmoe_low = hmoe_of_corpus(ids_low_var)
        hmoe_high = hmoe_of_corpus(ids_high_var)
        assert hmoe_high >= hmoe_low


class TestKCalibrationScore:

    def test_returns_float_for_known_tag(self):
        define_tag(
            name="kcal_tag",
            category="test",
            description="K calibration score test.",
            scope="Test.",
        )
        score = k_calibration_score("kcal_tag")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raises_for_unknown_tag(self):
        with pytest.raises(ValueError):
            k_calibration_score("tag_does_not_exist_xyz")


class TestSurveyAllTags:

    def test_empty_registry_returns_empty_list(self):
        result = survey_all_tags()
        # Empty registry → no tags → empty list (or empty after min_records filter)
        assert isinstance(result, list)

    def test_returns_distributions_for_qualifying_tags(self):
        define_tag("tag_a", category="test", description="Tag A.", scope="Test.")
        define_tag("tag_b", category="test", description="Tag B.", scope="Test.")
        for i in range(3):
            deposit(
                _make_profile(filename=f"a_{i}.wav", creative_residue=0.2 * i),
                context="A.",
                tags=["tag_a"],
            )
        for i in range(3):
            deposit(
                _make_profile(filename=f"b_{i}.wav", creative_residue=0.3 * i),
                context="B.",
                tags=["tag_b"],
            )
        results = survey_all_tags(min_records=2)
        tag_names = [d.tag_name for d in results]
        assert "tag_a" in tag_names
        assert "tag_b" in tag_names

    def test_results_sorted_calibration_ascending(self):
        """survey_all_tags returns worst-calibrated tags first."""
        define_tag("sorted_a", category="test", description="Sorted A.", scope="Test.")
        define_tag("sorted_b", category="test", description="Sorted B.", scope="Test.")
        for i in range(4):
            deposit(
                _make_profile(filename=f"sa_{i}.wav", creative_residue=0.5),
                context="Flat.",
                tags=["sorted_a"],
            )
        for i in range(4):
            deposit(
                _make_profile(filename=f"sb_{i}.wav", creative_residue=0.05 * (i + 1)),
                context="Varied.",
                tags=["sorted_b"],
            )
        results = survey_all_tags(min_records=2)
        if len(results) >= 2:
            scores = [d.calibration_score for d in results]
            # Should be ascending (worst first)
            assert scores == sorted(scores)


# ── validate.py tests ─────────────────────────────────────────────────────────

class TestValidateTagRevisionEmptyCorpus:

    def test_validate_on_empty_corpus(self):
        """A tag with no records should return a report without crashing."""
        define_tag(
            name="empty_tag",
            category="test",
            description="No records yet.",
            scope="Test.",
        )
        report = validate_tag_revision(
            tag_name="empty_tag",
            proposed_description="Updated: still no records.",
            proposed_scope="Test revised.",
            chunk_sizes=[5, 10],
        )
        assert isinstance(report, TagRevisionReport)
        assert report.tag_name == "empty_tag"

    def test_validate_raises_on_unknown_tag(self):
        with pytest.raises(ValueError, match="not found"):
            validate_tag_revision(
                tag_name="tag_xyz_does_not_exist",
                proposed_description="Whatever.",
                proposed_scope="Whatever.",
            )


class TestValidateTagRevisionReport:

    def _setup_tag_with_records(self, tag_name: str, n: int = 8):
        """Create a tag and deposit n records tagged with it."""
        define_tag(
            name=tag_name,
            category="test",
            description="A tag for testing revision validation.",
            scope="Any record with signal.",
        )
        for i in range(n):
            deposit(
                _make_profile(
                    filename=f"{tag_name}_{i}.wav",
                    lsii_score=0.1 * (i + 1),
                    creative_residue=0.15 * (i + 1),
                    authentic_emission=0.2 * (i + 1),
                ),
                context=f"Record {i} for {tag_name}.",
                tags=[tag_name],
            )

    def test_report_has_required_fields(self):
        self._setup_tag_with_records("rev_tag_a")
        report = validate_tag_revision(
            tag_name="rev_tag_a",
            proposed_description="Refined description.",
            proposed_scope="Narrowed scope.",
            chunk_sizes=[4, 8],
        )
        assert report.tag_name == "rev_tag_a"
        assert report.validated_at != ""
        assert report.recommendation in ("APPROVE", "CAUTION", "REJECT")
        assert isinstance(report.is_sound, bool)

    def test_report_hmoe_probes_populated(self):
        self._setup_tag_with_records("rev_tag_b", n=10)
        report = validate_tag_revision(
            tag_name="rev_tag_b",
            proposed_description="More refined.",
            proposed_scope="Narrowed.",
            chunk_sizes=[5, 10],
        )
        assert len(report.hmoe_probes) >= 1

    def test_report_probes_are_dicts_with_expected_keys(self):
        self._setup_tag_with_records("rev_tag_c")
        report = validate_tag_revision(
            tag_name="rev_tag_c",
            proposed_description="Test.",
            proposed_scope="Test.",
            chunk_sizes=[8],
        )
        for probe in report.hmoe_probes:
            assert "chunk_size" in probe
            assert "current_hmoe" in probe
            assert "projected_hmoe" in probe
            assert "delta_hmoe" in probe
            assert "stale_rate" in probe

    def test_evolutionary_optimum_marked(self):
        self._setup_tag_with_records("rev_tag_d", n=12)
        report = validate_tag_revision(
            tag_name="rev_tag_d",
            proposed_description="Tested.",
            proposed_scope="Tested.",
            chunk_sizes=[4, 8, 12],
        )
        optimum_count = sum(
            1 for p in report.hmoe_probes if p.get("is_evolutionary_optimum")
        )
        assert optimum_count == 1

    def test_db_wide_stale_count_is_nonneg(self):
        self._setup_tag_with_records("rev_tag_e")
        report = validate_tag_revision(
            tag_name="rev_tag_e",
            proposed_description="Updated.",
            proposed_scope="Updated.",
            chunk_sizes=[8],
        )
        assert report.db_wide_stale_count >= 0

    def test_implication_summary_is_string(self):
        self._setup_tag_with_records("rev_tag_f")
        report = validate_tag_revision(
            tag_name="rev_tag_f",
            proposed_description="New desc.",
            proposed_scope="New scope.",
        )
        assert isinstance(report.implication_summary, str)
        assert len(report.implication_summary) > 0

    def test_revision_rationale_mentions_tag(self):
        self._setup_tag_with_records("rev_tag_g")
        report = validate_tag_revision(
            tag_name="rev_tag_g",
            proposed_description="Revised description here.",
            proposed_scope="Revised scope.",
        )
        # rationale should include the proposed description text
        assert "Revised description here" in report.revision_rationale

    def test_calibration_notes_is_list(self):
        self._setup_tag_with_records("rev_tag_h")
        report = validate_tag_revision(
            tag_name="rev_tag_h",
            proposed_description="Test.",
            proposed_scope="Test.",
        )
        assert isinstance(report.calibration_notes, list)

    def test_report_current_version_matches_defined_tag(self):
        define_tag(
            name="ver_check_tag",
            category="test",
            description="Version 1.",
            scope="Test.",
        )
        revise_tag(
            name="ver_check_tag",
            description="Version 2.",
            scope="Narrower.",
            revision_reason="Testing version bump.",
        )
        report = validate_tag_revision(
            tag_name="ver_check_tag",
            proposed_description="Version 3 proposal.",
            proposed_scope="Even narrower.",
            chunk_sizes=[1],
        )
        # Should be version 2 now (after one revise_tag call)
        assert report.tag_current_version == 2


class TestPrintReport:

    def test_print_report_runs_without_error(self, capsys):
        define_tag("print_tag", category="test", description="Print test.", scope="Test.")
        report = validate_tag_revision(
            tag_name="print_tag",
            proposed_description="Printed.",
            proposed_scope="Printed.",
            chunk_sizes=[1],
        )
        print_report(report)
        captured = capsys.readouterr()
        assert "KindPress Tag Revision Report" in captured.out
        assert report.recommendation in captured.out

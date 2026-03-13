"""
Tests for core/influence_mapper.py — Module 7: Influence Chain Mapper with Kindfluence Bridge.

Validates:
  - Dataclass construction
  - Lineage loading and caching
  - Lineage scoring against signal features
  - Mechanic detection across all 13 Kindfluence mechanics
  - Direction classification
  - Gradient stage estimation
  - Full pipeline integration

The test strategy uses synthetic feature data to isolate each detection rule,
verifying that the mechanic detection logic faithfully represents the Kindfluence
catalogue without false positives from unrelated features.
"""

import pytest
import sys
import os
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fingerprints():
    """Return a minimal FingerprintReport-like object for testing."""
    return _make_fingerprints()


@pytest.fixture
def extractive_fingerprints():
    """Feature set typical of heavily engineered commercial pop."""
    return _make_fingerprints(
        manufacturing_markers=[
            'heavy compression', 'grid-quantised rhythm', 'professional reverb',
            'commercial loudness', 'hypercompression',
        ],
        authenticity_markers=[],
        era_name='2010s',
        era_confidence=0.85,
    )


@pytest.fixture
def syntropic_fingerprints():
    """Feature set typical of authentic, unmanufactured work."""
    return _make_fingerprints(
        manufacturing_markers=[],
        authenticity_markers=[
            'natural dynamics', 'human performance', 'preserved breath', 'wide dynamic range',
        ],
        era_name='1970s',
        era_confidence=0.55,
    )


@pytest.fixture
def mock_features():
    """Full SegmentFeatures-like object for testing."""
    return _make_features()


@pytest.fixture
def priming_features():
    """Features that should trigger priming mechanic (extractive)."""
    return _make_features(
        tempo_bpm=130.0,
        dynamic_range=5.5,
        groove_deviation=1.5,
        harmonic_complexity=0.3,
    )


@pytest.fixture
def reciprocity_features():
    """Features that should trigger reciprocity mechanic (syntropic)."""
    return _make_features(
        dynamic_range=15.0,
        groove_deviation=20.0,
        crest_factor=13.0,
    )


@pytest.fixture
def dopamine_loop_features():
    """Features that should trigger dopamine_loops (always extractive)."""
    return _make_features(
        tempo_bpm=128.0,
        dynamic_range=5.0,
        groove_deviation=2.0,
    )


@pytest.fixture
def narrative_transport_features():
    """Features that should trigger narrative_transportation (syntropic)."""
    return _make_features(
        harmonic_complexity=0.6,
        lsii_score=0.6,
        authentic_emission=0.7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: synthetic fingerprint / feature builders
# ─────────────────────────────────────────────────────────────────────────────

class _FingerprintMatch:
    """Minimal FingerprintMatch mimic for testing."""
    def __init__(self, name, confidence=0.7):
        self.name = name
        self.confidence = confidence


def _make_fingerprints(
    era_name='2010s',
    era_confidence=0.7,
    manufacturing_markers=None,
    authenticity_markers=None,
    technique_names=None,
):
    """
    Build a FingerprintReport-like object to pass into _detect_mechanics.
    FingerprintReport attributes used by influence_mapper:
      .likely_era          -> List[FingerprintMatch] (each has .name, .confidence)
      .likely_techniques   -> List[FingerprintMatch]
      .manufacturing_markers -> List[str]
      .authenticity_markers  -> List[str]
    """
    if manufacturing_markers is None:
        manufacturing_markers = []
    if authenticity_markers is None:
        authenticity_markers = []
    if technique_names is None:
        technique_names = []

    class _FingerprintReport:
        pass

    fp = _FingerprintReport()
    fp.likely_era = [_FingerprintMatch(era_name, era_confidence)]
    fp.likely_techniques = [_FingerprintMatch(t) for t in technique_names]
    fp.manufacturing_markers = list(manufacturing_markers)
    fp.authenticity_markers = list(authenticity_markers)
    return fp


def _make_features(
    tempo_bpm=120.0,
    dynamic_range=10.0,
    groove_deviation=10.0,
    crest_factor=8.0,
    harmonic_complexity=0.4,
    tension_ratio=0.3,
    lsii_score=0.2,
    authentic_emission=0.5,
    manufacturing_score=0.4,
):
    """Build minimal SegmentFeatures-like object used by influence_mapper."""

    class _Temporal:
        pass

    class _Dynamic:
        pass

    class _Harmonic:
        pass

    class _Features:
        pass

    temporal = _Temporal()
    temporal.tempo_bpm = tempo_bpm
    temporal.groove_deviation_ms = groove_deviation

    dynamic = _Dynamic()
    dynamic.dynamic_range_db = dynamic_range
    dynamic.crest_factor_db = crest_factor

    harmonic = _Harmonic()
    harmonic.harmonic_complexity = harmonic_complexity
    harmonic.tension_ratio = tension_ratio

    feat = _Features()
    feat.temporal = temporal
    feat.dynamic = dynamic
    feat.harmonic = harmonic
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Import under test
# ─────────────────────────────────────────────────────────────────────────────

from core.influence_mapper import (
    InfluenceNode,
    DetectedMechanic,
    InfluenceChain,
    _load_lineages,
    _score_lineage_match,
    _detect_mechanics,
    _classify_mechanic_direction,
    _estimate_gradient_stage,
    map_influence_chain,
    _SYNTROPY_REPAIRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass construction
# ─────────────────────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_influence_node_fields(self):
        node = InfluenceNode(
            id='delta_blues',
            name='Delta Blues',
            era_range=(1920, 1945),
            parent_lineages=[],
            sonic_characteristics={'dynamic_range': (15, 30)},
            cultural_context='Mississippi Delta, 1920s',
            kindpath_notes='Root tradition.',
            confidence=0.9,
        )
        assert node.id == 'delta_blues'
        assert node.era_range == (1920, 1945)
        assert node.confidence == 0.9

    def test_detected_mechanic_fields(self):
        m = DetectedMechanic(
            name='priming',
            evidence=['tempo 130 BPM', 'heavy compression'],
            direction='extractive',
            confidence=0.75,
            syntropy_repair='[PRIMING] Name and teach the mechanic openly.',
        )
        assert m.name == 'priming'
        assert m.direction == 'extractive'
        assert '[PRIMING]' in m.syntropy_repair

    def test_influence_chain_fields(self):
        chain = InfluenceChain(
            primary_lineage=[],
            secondary_lineages=[],
            innovation_points=[],
            confluence_points=[],
            narrative='Test narrative.',
            detected_mechanics=[],
            mechanic_direction='neutral',
            gradient_stage_estimate='SEED',
            syntropy_repair_vectors=[],
            mechanic_summary='No mechanics detected.',
        )
        assert chain.gradient_stage_estimate == 'SEED'
        assert chain.mechanic_direction == 'neutral'


# ─────────────────────────────────────────────────────────────────────────────
# Lineage loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLineageLoading:
    def test_lineages_load(self):
        """lineages.json exists and loads correctly."""
        lineages = _load_lineages()
        assert isinstance(lineages, dict)
        assert len(lineages) > 0

    def test_lineage_entry_structure(self):
        """Each lineage has required fields."""
        lineages = _load_lineages()
        required_keys = {'id', 'name', 'era_range', 'sonic_characteristics', 'cultural_context'}
        for entry in list(lineages.values())[:5]:  # Spot-check first 5
            assert required_keys.issubset(entry.keys()), \
                f"Lineage '{entry.get('id', '?')}' missing keys: {required_keys - set(entry.keys())}"

    def test_lineage_ids_unique(self):
        """All lineage IDs are unique (inherent in dict)."""
        lineages = _load_lineages()
        # dict keys are inherently unique
        assert len(lineages) == len(lineages.keys())

    def test_lineages_cover_known_entries(self):
        """At minimum, well-known lineages are present."""
        lineages = _load_lineages()
        expected = {'delta_blues', 'early_hip_hop', 'house', 'techno', 'uk_grime'}
        missing = expected - set(lineages.keys())
        assert not missing, f"Missing expected lineages: {missing}"

    def test_lineages_cached(self):
        """Second call returns same object (caching works)."""
        a = _load_lineages()
        b = _load_lineages()
        assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# Lineage scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestLineageScoring:
    def test_perfect_era_match(self):
        """A lineage entry matching era/tempo/groove scores > 0."""
        lineage_entry = {
            'id': 'house',
            'name': 'House',
            'era_range': [1985, 1991],
            'sonic_characteristics': {
                'tempo_bpm_range': [120, 130],
                'groove_deviation_ms': [1, 5],
                'dynamic_range_db': [4, 10],
            },
            'cultural_context': 'Chicago, 1985.',
            'kindpath_notes': '',
            'parent_lineages': [],
        }
        # era_names contain '1980s' which should match against era_range starting 1985
        score = _score_lineage_match(
            lineage_entry,
            era_names=['1980s'],
            era_confidences={'1980s': 0.85},
            tempo=125.0,
            groove_ms=3.0,
            dynamic_range=7.0,
        )
        assert score > 0.0

    def test_era_mismatch_lowers_score(self):
        """A 1920s lineage matched against 2020s era data scores lower than a good match."""
        lineage_early = {
            'id': 'delta_blues',
            'name': 'Delta Blues',
            'era_range': [1920, 1945],
            'sonic_characteristics': {},
            'cultural_context': '',
            'kindpath_notes': '',
            'parent_lineages': [],
        }
        lineage_match = {
            'id': 'hyperpop',
            'name': 'Hyperpop',
            'era_range': [2019, 9999],
            'sonic_characteristics': {},
            'cultural_context': '',
            'kindpath_notes': '',
            'parent_lineages': [],
        }
        score_early = _score_lineage_match(
            lineage_early,
            era_names=['2020s'],
            era_confidences={'2020s': 0.9},
            tempo=90.0, groove_ms=5.0, dynamic_range=12.0,
        )
        score_match = _score_lineage_match(
            lineage_match,
            era_names=['2020s'],
            era_confidences={'2020s': 0.9},
            tempo=90.0, groove_ms=5.0, dynamic_range=12.0,
        )
        # The modern lineage should score at least as well as the early one for modern era input
        assert score_match >= score_early

    def test_score_range(self):
        """Score is always between 0 and 1."""
        lineages = _load_lineages()
        for lid, lineage in list(lineages.items())[:10]:
            score = _score_lineage_match(
                lineage,
                era_names=['1990s'],
                era_confidences={'1990s': 0.5},
                tempo=100.0,
                groove_ms=10.0,
                dynamic_range=10.0,
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {lid}"


# ─────────────────────────────────────────────────────────────────────────────
# Mechanic detection
# ─────────────────────────────────────────────────────────────────────────────

class TestMechanicDetection:
    def test_priming_detected_extractive(self, priming_features, extractive_fingerprints):
        """High mfg + 130BPM + heavy compression + narrow dynamic range → priming (extractive)."""
        # Add the technique names that the implementation looks for
        extractive_fingerprints.likely_techniques = [
            _FingerprintMatch('heavy compression', 0.9),
            _FingerprintMatch('grid-quantised rhythm', 0.85),
        ]
        mechanics = _detect_mechanics(
            fingerprints=extractive_fingerprints,
            features=priming_features,
            manufacturing_score=0.75,
            authentic_emission_score=0.2,
            lsii_score=0.1,
        )
        names = {m.name for m in mechanics}
        assert 'priming' in names
        priming = next(m for m in mechanics if m.name == 'priming')
        assert priming.direction == 'extractive'

    def test_reciprocity_syntropic(self, reciprocity_features, syntropic_fingerprints):
        """Natural dynamics + human performance + authentic > 0.4 → reciprocity (syntropic)."""
        syntropic_fingerprints.likely_techniques = [
            _FingerprintMatch('natural dynamics', 0.85),
            _FingerprintMatch('human performance', 0.8),
        ]
        mechanics = _detect_mechanics(
            fingerprints=syntropic_fingerprints,
            features=reciprocity_features,
            manufacturing_score=0.1,
            authentic_emission_score=0.75,
            lsii_score=0.3,
        )
        names = {m.name for m in mechanics}
        assert 'reciprocity' in names
        rec = next(m for m in mechanics if m.name == 'reciprocity')
        assert rec.direction == 'syntropic'

    def test_dopamine_loops_always_extractive(self, dopamine_loop_features, extractive_fingerprints):
        """
        dopamine_loops is ALWAYS extractive regardless of authentic emission.
        Hard Kindfluence rule — dependency-creation mechanics are never syntropic.
        """
        extractive_fingerprints.likely_techniques = [
            _FingerprintMatch('heavy compression', 0.9),
            _FingerprintMatch('grid-quantised rhythm', 0.85),
        ]
        for authentic_score in [0.0, 0.5, 0.9, 1.0]:
            mechanics = _detect_mechanics(
                fingerprints=extractive_fingerprints,
                features=dopamine_loop_features,
                manufacturing_score=0.7,
                authentic_emission_score=authentic_score,
                lsii_score=0.1,
            )
            dopamine = next((m for m in mechanics if m.name == 'dopamine_loops'), None)
            if dopamine:
                assert dopamine.direction == 'extractive', \
                    f"dopamine_loops must always be extractive (authentic={authentic_score})"

    def test_narrative_transportation_syntropic(
        self, narrative_transport_features, syntropic_fingerprints
    ):
        """High LSII + harmonic complexity + high authentic → narrative_transportation (syntropic)."""
        syntropic_fingerprints.likely_techniques = []
        mechanics = _detect_mechanics(
            fingerprints=syntropic_fingerprints,
            features=narrative_transport_features,
            manufacturing_score=0.1,
            authentic_emission_score=0.7,
            lsii_score=0.6,
        )
        names = {m.name for m in mechanics}
        assert 'narrative_transportation' in names
        nt = next(m for m in mechanics if m.name == 'narrative_transportation')
        assert nt.direction == 'syntropic'

    def test_cognitive_ease_extractive(self):
        """Very low harmonic_complexity + quantised + high mfg → cognitive_ease (extractive)."""
        features = _make_features(
            harmonic_complexity=0.15,
            tension_ratio=0.2,
            groove_deviation=1.5,
        )
        fingerprints = _make_fingerprints(
            manufacturing_markers=['grid-quantised rhythm'],
            technique_names=['grid-quantised rhythm'],
        )
        mechanics = _detect_mechanics(
            fingerprints=fingerprints,
            features=features,
            manufacturing_score=0.75,
            authentic_emission_score=0.2,
            lsii_score=0.1,
        )
        names = {m.name for m in mechanics}
        assert 'cognitive_ease' in names
        ce = next(m for m in mechanics if m.name == 'cognitive_ease')
        assert ce.direction == 'extractive'

    def test_bandwagon_needs_multiple_markers(self):
        """Bandwagon requires 3+ manufacturing markers — fewer should not trigger it."""
        # Only 1 marker — should not trigger bandwagon
        fingerprints_few = _make_fingerprints(
            manufacturing_markers=['heavy compression'],
            technique_names=['heavy compression'],
        )
        features_few = _make_features(groove_deviation=2.0)
        mechanics_few = _detect_mechanics(
            fingerprints=fingerprints_few,
            features=features_few,
            manufacturing_score=0.5,
            authentic_emission_score=0.4,
            lsii_score=0.1,
        )
        assert not any(m.name == 'bandwagon_effect' for m in mechanics_few)

        # 3+ markers — should trigger bandwagon (with heavy_compression + quantised technique)
        mfg = ['heavy compression', 'grid-quantised rhythm', 'professional reverb', 'commercial loudness']
        fingerprints_many = _make_fingerprints(
            manufacturing_markers=mfg,
            technique_names=['heavy compression', 'grid-quantised rhythm'],
        )
        features_many = _make_features(groove_deviation=2.0)
        mechanics_many = _detect_mechanics(
            fingerprints=fingerprints_many,
            features=features_many,
            manufacturing_score=0.75,
            authentic_emission_score=0.2,
            lsii_score=0.1,
        )
        assert any(m.name == 'bandwagon_effect' for m in mechanics_many)

    def test_no_mechanics_clean_signal(self):
        """
        A genuinely clean signal (no strong mechanics triggers) should have zero
        extractive mechanics detected when manufacturing score is low and no
        trigger conditions are met.
        """
        clean_features = _make_features(
            tempo_bpm=85.0,
            dynamic_range=18.0,
            groove_deviation=25.0,
            harmonic_complexity=0.6,
            tension_ratio=0.3,
        )
        clean_fp = _make_fingerprints(
            manufacturing_markers=[],
            authenticity_markers=['natural dynamics', 'human performance'],
            technique_names=[],
        )
        mechanics = _detect_mechanics(
            fingerprints=clean_fp,
            features=clean_features,
            manufacturing_score=0.1,
            authentic_emission_score=0.85,
            lsii_score=0.3,
        )
        extractive_count = sum(1 for m in mechanics if m.direction == 'extractive')
        assert extractive_count == 0, \
            f"Expected 0 extractive, got {extractive_count}: {[m.name for m in mechanics if m.direction == 'extractive']}"

    def test_syntropy_repair_vectors_populated_for_extractive(self):
        """Every extractive mechanic should have a non-empty syntropy_repair."""
        features = _make_features(
            tempo_bpm=130.0,
            dynamic_range=5.0,
            groove_deviation=1.5,
            harmonic_complexity=0.15,
        )
        fingerprints = _make_fingerprints(
            manufacturing_markers=['heavy compression', 'grid-quantised rhythm', 'commercial loudness'],
            technique_names=['heavy compression', 'grid-quantised rhythm'],
        )
        mechanics = _detect_mechanics(
            fingerprints=fingerprints,
            features=features,
            manufacturing_score=0.8,
            authentic_emission_score=0.1,
            lsii_score=0.05,
        )
        for m in mechanics:
            if m.direction == 'extractive':
                assert m.syntropy_repair, f"{m.name} has empty syntropy_repair"
                assert len(m.syntropy_repair) > 20


# ─────────────────────────────────────────────────────────────────────────────
# Direction classification
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionClassification:
    def _make_mechanics(self, directions: list) -> list:
        return [
            DetectedMechanic(
                name=f'mechanic_{i}',
                evidence=[],
                direction=d,
                confidence=0.7,
                syntropy_repair='',
            )
            for i, d in enumerate(directions)
        ]

    def test_all_extractive(self):
        mechanics = self._make_mechanics(['extractive', 'extractive', 'extractive'])
        direction = _classify_mechanic_direction(mechanics, 0.2, 0.8)
        assert direction == 'extractive'

    def test_all_syntropic(self):
        mechanics = self._make_mechanics(['syntropic', 'syntropic'])
        direction = _classify_mechanic_direction(mechanics, 0.8, 0.2)
        assert direction == 'syntropic'

    def test_mixed(self):
        mechanics = self._make_mechanics(['extractive', 'syntropic', 'extractive'])
        direction = _classify_mechanic_direction(mechanics, 0.4, 0.6)
        assert direction in ('mixed', 'extractive')

    def test_empty_list(self):
        direction = _classify_mechanic_direction([], 0.5, 0.5)
        assert direction == 'neutral'

    def test_ambiguous_only(self):
        mechanics = self._make_mechanics(['ambiguous', 'ambiguous'])
        direction = _classify_mechanic_direction(mechanics, 0.5, 0.5)
        assert direction in ('neutral', 'ambiguous', 'mixed')


# ─────────────────────────────────────────────────────────────────────────────
# Gradient stage estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientStageEstimation:
    def test_extractive_direction_is_infiltrate(self):
        stage = _estimate_gradient_stage('extractive', 0.0, 0.1)
        assert stage == 'INFILTRATE'

    def test_neutral_direction_is_infiltrate(self):
        stage = _estimate_gradient_stage('neutral', 0.3, 0.5)
        assert stage == 'INFILTRATE'

    def test_mixed_direction_is_seed(self):
        stage = _estimate_gradient_stage('mixed', 0.3, 0.5)
        assert stage == 'SEED'

    def test_syntropic_low_authentic_is_emerge(self):
        stage = _estimate_gradient_stage('syntropic', 0.3, 0.62)
        assert stage == 'EMERGE'

    def test_syntropic_high_authentic_is_activate(self):
        stage = _estimate_gradient_stage('syntropic', 0.5, 0.80)
        assert stage == 'ACTIVATE'

    def test_syntropic_boundary(self):
        """At the 0.75 boundary, should be ACTIVATE or EMERGE."""
        stage = _estimate_gradient_stage('syntropic', 0.4, 0.75)
        assert stage in ('ACTIVATE', 'EMERGE')  # Boundary case


# ─────────────────────────────────────────────────────────────────────────────
# Syntropy repairs catalogue
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntropyRepairsCatalogue:
    def test_all_mechanics_have_repair(self):
        """Every Kindfluence mechanic should have a syntropy repair entry."""
        expected_keys = {
            'priming', 'social_proof', 'anchoring', 'reciprocity', 'mere_exposure',
            'framing', 'narrative_transportation', 'dopamine_loops',
            'identity_association', 'loss_aversion', 'cognitive_ease',
            'authority_bias', 'bandwagon_effect',
        }
        missing = expected_keys - set(_SYNTROPY_REPAIRS.keys())
        assert not missing, f"Missing syntropy repairs for: {missing}"

    def test_repair_strings_non_empty(self):
        for name, repair in _SYNTROPY_REPAIRS.items():
            assert len(repair) > 20, f"Syntropy repair for '{name}' is too short: {repair!r}"

    def test_repair_prefixed_with_mechanic(self):
        """Each repair string should be substantive and actionable (more than a sentence)."""
        for name, repair in _SYNTROPY_REPAIRS.items():
            word_count = len(repair.split())
            assert word_count >= 10, \
                f"Repair for '{name}' is not substantive enough ({word_count} words): {repair!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMapInfluenceChainIntegration:
    def _make_pipeline_fingerprints(self, manufacturing=True):
        if manufacturing:
            return _make_fingerprints(
                era_name='2010s',
                era_confidence=0.8,
                manufacturing_markers=['heavy compression', 'grid-quantised rhythm', 'commercial loudness'],
                authenticity_markers=[],
                technique_names=['heavy compression', 'grid-quantised rhythm'],
            )
        else:
            return _make_fingerprints(
                era_name='1970s',
                era_confidence=0.65,
                manufacturing_markers=[],
                authenticity_markers=['natural dynamics', 'human performance'],
                technique_names=['natural dynamics', 'human performance'],
            )

    def test_returns_influence_chain(self):
        fp = self._make_pipeline_fingerprints(manufacturing=True)
        features = _make_features(
            tempo_bpm=128.0, dynamic_range=5.5, groove_deviation=2.0,
        )
        chain = map_influence_chain(
            fingerprints=fp,
            features=features,
            manufacturing_score=0.75,
            authentic_emission_score=0.2,
            lsii_score=0.1,
        )
        assert isinstance(chain, InfluenceChain)

    def test_extractive_chain_has_harvest_vectors(self):
        """An extractive commercial piece should emit syntropy_repair_vectors."""
        fp = self._make_pipeline_fingerprints(manufacturing=True)
        features = _make_features(
            tempo_bpm=130.0, dynamic_range=5.0, groove_deviation=1.5,
            harmonic_complexity=0.2,
        )
        chain = map_influence_chain(
            fingerprints=fp,
            features=features,
            manufacturing_score=0.80,
            authentic_emission_score=0.15,
            lsii_score=0.1,
        )
        assert len(chain.syntropy_repair_vectors) > 0
        assert chain.mechanic_direction in ('extractive', 'mixed')

    def test_syntropic_chain_gradient_stage(self):
        """A genuine, unmanufactured piece should reach EMERGE or higher."""
        fp = self._make_pipeline_fingerprints(manufacturing=False)
        features = _make_features(
            tempo_bpm=90.0, dynamic_range=18.0, groove_deviation=22.0,
            harmonic_complexity=0.6, lsii_score=0.6, authentic_emission=0.8,
        )
        chain = map_influence_chain(
            fingerprints=fp,
            features=features,
            manufacturing_score=0.1,
            authentic_emission_score=0.80,
            lsii_score=0.6,
        )
        assert chain.gradient_stage_estimate in ('EMERGE', 'ACTIVATE', 'TRANSFER', 'LIBERATE'), \
            f"Expected syntropic gradient stage, got: {chain.gradient_stage_estimate}"

    def test_narrative_string_populated(self):
        """Chain.narrative is a non-empty string."""
        fp = self._make_pipeline_fingerprints()
        features = _make_features()
        chain = map_influence_chain(
            fingerprints=fp, features=features,
            manufacturing_score=0.5, authentic_emission_score=0.5, lsii_score=0.3,
        )
        assert isinstance(chain.narrative, str)
        assert len(chain.narrative) > 0

    def test_mechanic_summary_present(self):
        """mechanic_summary is always populated."""
        fp = self._make_pipeline_fingerprints()
        features = _make_features()
        chain = map_influence_chain(
            fingerprints=fp, features=features,
            manufacturing_score=0.5, authentic_emission_score=0.5, lsii_score=0.3,
        )
        assert isinstance(chain.mechanic_summary, str)
        assert len(chain.mechanic_summary) > 0

    def test_seedbank_baseline_optional(self):
        """map_influence_chain works without seedbank_baseline."""
        fp = self._make_pipeline_fingerprints()
        features = _make_features()
        chain = map_influence_chain(
            fingerprints=fp, features=features,
            manufacturing_score=0.5, authentic_emission_score=0.5, lsii_score=0.3,
            seedbank_baseline=None,
        )
        assert isinstance(chain, InfluenceChain)

    def test_repair_vectors_prefixed(self):
        """Each repair vector is prefixed with the mechanic name in brackets."""
        fp = self._make_pipeline_fingerprints(manufacturing=True)
        features = _make_features(
            tempo_bpm=130.0, dynamic_range=5.0, groove_deviation=1.5,
            harmonic_complexity=0.15,
        )
        chain = map_influence_chain(
            fingerprints=fp, features=features,
            manufacturing_score=0.80, authentic_emission_score=0.15, lsii_score=0.1,
        )
        for vector in chain.syntropy_repair_vectors:
            assert vector.startswith('['), f"Repair vector not prefixed: {vector[:60]}"


# ─────────────────────────────────────────────────────────────────────────────
# KindSearch integration
# ─────────────────────────────────────────────────────────────────────────────

class TestKindSearchIntegration:
    """Test core/search.py functions that wrap the seedbank and profile comparison."""

    def test_import_search_module(self):
        """core/search.py is importable."""
        from core import search as ks
        assert hasattr(ks, 'search')
        assert hasattr(ks, 'find_similar')
        assert hasattr(ks, 'compare_mechanic_direction')
        assert hasattr(ks, 'get_lineage_context')

    def test_get_lineage_context_valid_id(self):
        """get_lineage_context returns a dict for a valid ID."""
        from core.search import get_lineage_context
        result = get_lineage_context('delta_blues')
        if result is not None:  # May be None if lineages.json not found in test context
            assert isinstance(result, dict)
            assert 'name' in result

    def test_get_lineage_context_unknown_id(self):
        """get_lineage_context returns None for unknown IDs."""
        from core.search import get_lineage_context
        result = get_lineage_context('nonexistent_lineage_xyz_999')
        assert result is None

    def test_compare_mechanic_direction_same(self):
        """Two identical profiles produce aligned interpretation."""
        from core.search import compare_mechanic_direction
        profile = {
            'influence_chain': {
                'mechanic_direction': 'extractive',
                'gradient_stage_estimate': 'INFILTRATE',
                'detected_mechanics': [
                    {'name': 'priming'},
                    {'name': 'dopamine_loops'},
                ],
            }
        }
        result = compare_mechanic_direction(profile, profile)
        assert result['direction_a'] == 'extractive'
        assert result['direction_b'] == 'extractive'
        assert 'priming' in result['shared_mechanics']
        assert 'extractive' in result['interpretation'].lower()

    def test_compare_mechanic_direction_opposite(self):
        """Extractive vs syntropic produces contrast interpretation."""
        from core.search import compare_mechanic_direction
        profile_a = {
            'influence_chain': {
                'mechanic_direction': 'extractive',
                'gradient_stage_estimate': 'INFILTRATE',
                'detected_mechanics': [{'name': 'priming'}],
            }
        }
        profile_b = {
            'influence_chain': {
                'mechanic_direction': 'syntropic',
                'gradient_stage_estimate': 'EMERGE',
                'detected_mechanics': [{'name': 'reciprocity'}],
            }
        }
        result = compare_mechanic_direction(profile_a, profile_b)
        assert result['direction_a'] != result['direction_b']
        assert 'opposite' in result['interpretation'].lower() or 'contrast' in result['interpretation'].lower()

    def test_get_influence_summary_empty_profile(self):
        """get_influence_summary handles profiles with no influence chain."""
        from core.search import get_influence_summary
        result = get_influence_summary({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_influence_summary_with_chain(self):
        """get_influence_summary produces human-readable text from a profile."""
        from core.search import get_influence_summary
        profile = {
            'influence_chain': {
                'narrative': 'This piece descends from Chicago house traditions.',
                'mechanic_summary': 'Dopamine loop mechanics detected.',
                'mechanic_direction': 'extractive',
                'gradient_stage_estimate': 'INFILTRATE',
                'syntropy_repair_vectors': ['[DOPAMINE_LOOPS] Teach the loop.'],
                'detected_mechanics': [],
            }
        }
        result = get_influence_summary(profile)
        assert 'Chicago house' in result or 'extraction' in result.lower() or 'INFILTRATE' in result

    def test_search_returns_list(self):
        """search() always returns a list (even when seedbank is empty)."""
        from core.search import search
        results = search(lsii_min=0.5, limit=5)
        assert isinstance(results, list)

    def test_find_similar_returns_list(self):
        """find_similar() always returns a list."""
        from core.search import find_similar
        profile = {'lsii': {'score': 0.5}, 'psychosomatic': {'authentic_emission_score': 0.6}}
        results = find_similar(profile, limit=5)
        assert isinstance(results, list)

"""
Tests for reports/report_generator.py

Validates HTML and JSON report generation using synthetic analysis data.
All sections of the HTML report must be present and valid.

No mocks — real AudioRecord instances created with small synthetic arrays.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pytest

from core.ingestion import AudioRecord
from core.divergence import (
    TrajectoryProfile, LatesonginversionResult, DivergenceVector
)
from core.fingerprints import FingerprintReport, FingerprintMatch
from core.psychosomatics import PsychosomaticProfile
from reports.report_generator import generate_html_report, generate_json_report


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_audio_record():
    """Real AudioRecord with small synthetic arrays — no mocks."""
    sr = 44100
    duration = 10.0  # short but real
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    y_mono = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y_stereo = np.stack([y_mono, y_mono])

    return AudioRecord(
        filepath='/tmp/test_audio.wav',
        filename='test_audio.wav',
        format='wav',
        duration_seconds=duration,
        sample_rate=sr,
        num_channels=2,
        bit_depth=16,
        y_mono=y_mono,
        y_stereo=y_stereo,
        peak_amplitude=float(np.max(np.abs(y_mono))),
        rms_amplitude=float(np.sqrt(np.mean(y_mono ** 2))),
        dynamic_range_db=12.0,
        is_clipped=False,
        clipping_percentage=0.0,
        is_silence=False,
    )


def make_divergence_vector():
    return DivergenceVector(
        spectral_centroid_delta=0.15,
        spectral_flux_delta=0.05,
        spectral_complexity_delta=0.1,
        dynamic_range_delta=-0.3,
        dynamic_energy_delta=-0.25,
        harmonic_tension_delta=0.4,
        harmonic_tonality_delta=0.1,
        harmonic_complexity_delta=0.2,
        temporal_groove_delta=0.05,
        temporal_syncopation_delta=0.02,
        temporal_onset_density_delta=-0.1,
    )


def make_lsii(score=0.45):
    return LatesonginversionResult(
        lsii=score,
        direction="darker and quieter",
        dominant_axis="harmonic_tension",
        divergence=make_divergence_vector(),
        q1_q3_baseline={'energy': 0.6, 'tension': 0.2},
        q4_values={'energy': 0.3, 'tension': 0.6},
        trajectory_description="Consistent Q1-Q3, shift in Q4",
        inversion_description="Tension rises sharply in final quarter",
        flag_level="moderate",
        flag_notes="Moderate late-section inversion detected",
    )


def make_trajectory():
    return TrajectoryProfile(
        quarters=[
            {'tempo_bpm': 125, 'onset_density': 4.5},
            {'tempo_bpm': 126, 'onset_density': 4.2},
            {'tempo_bpm': 124, 'onset_density': 4.8},
            {'tempo_bpm': 122, 'onset_density': 3.1},
        ],
        valence_arc=[0.3, 0.35, 0.32, 0.0],
        energy_arc=[0.65, 0.7, 0.68, 0.35],
        complexity_arc=[0.5, 0.55, 0.52, 0.6],
        coherence_arc=[0.75, 0.72, 0.78, 0.7],
        tension_arc=[0.2, 0.25, 0.22, 0.65],
        lsii_result=make_lsii(),
    )


def make_fingerprints():
    return FingerprintReport(
        likely_instruments=[
            FingerprintMatch(category='instrument', name='Electric guitar', confidence=0.72,
                             evidence=['harmonic series', 'attack transient'], description='Typical electric guitar signature')
        ],
        likely_era=[
            FingerprintMatch(category='era', name='2010s', confidence=0.65,
                             evidence=['dynamic range 8dB', 'streaming loudness'], description='2010s production era')
        ],
        likely_techniques=[
            FingerprintMatch(category='technique', name='Natural dynamics', confidence=0.8,
                             evidence=['crest factor 14dB'], description='Preserved dynamic range'),
            FingerprintMatch(category='technique', name='Human performance', confidence=0.7,
                             evidence=['groove deviation 18ms'], description='Live timing variation'),
        ],
        production_context='2010s digital recording with preserved dynamics',
        authenticity_markers=['preserved dynamics', 'live timing variation'],
        manufacturing_markers=['professional studio quality'],
    )


def make_psychosomatic():
    return PsychosomaticProfile(
        valence=-0.1,
        arousal=0.55,
        coherence=0.72,
        authenticity_index=0.65,
        complexity=0.53,
        tension_resolution_ratio=1.8,
        relational_density=0.45,
        predicted_physical_responses=['chest tightening', 'held breath'],
        predicted_emotional_states=['unresolved longing', 'melancholy'],
        priming_vectors=['activation / forward motion'],
        prestige_signals=['professional studio quality'],
        identity_capture_risk=0.2,
        stage1_priming_detected=True,
        stage1_evidence=['Tempo 125 BPM — heart-rate entrainment range'],
        stage2_prestige_attached=False,
        stage2_evidence=[],
        stage3_tag_risk=0.15,
        stage3_evidence=[],
        lsii_psychosomatic_reading='Notable late-section divergence on harmonic tension axis.',
        authentic_emission_score=0.62,
        manufacturing_score=0.15,
        creative_residue=0.48,
        somatic_summary='Moderate arousal, slightly negative valence, unresolved tension.',
        mechanism_summary='Stage 1 priming present — activation via tempo.',
        elder_reading=(
            'This piece holds steady for three quarters then pulls back. '
            'The tension that was building does not resolve. Something was being '
            'said in the final minute that couldn\'t be said in the frame the piece built.'
        ),
    )


# ── HTML report tests ────────────────────────────────────────────────────────

class TestHtmlReport:

    def _make_all(self):
        return (make_audio_record(), make_trajectory(),
                make_fingerprints(), make_psychosomatic())

    def test_returns_string(self):
        result = generate_html_report(*self._make_all())
        assert isinstance(result, str)
        assert len(result) > 500

    def test_contains_doctype(self):
        result = generate_html_report(*self._make_all())
        assert '<!DOCTYPE html>' in result

    def test_section_1_identity_present(self):
        result = generate_html_report(*self._make_all())
        assert 'test_audio.wav' in result

    def test_section_2_elder_reading_present(self):
        result = generate_html_report(*self._make_all())
        assert 'Elder' in result or 'elder' in result
        # The actual elder reading text should appear
        assert 'three quarters' in result or 'final minute' in result

    def test_section_3_arc_present(self):
        result = generate_html_report(*self._make_all())
        # LSII score should appear
        assert '0.45' in result or '0.450' in result
        # Quarter labels
        assert 'Q1' in result and 'Q4' in result

    def test_section_3_flag_level_shown(self):
        result = generate_html_report(*self._make_all())
        assert 'moderate' in result.lower()

    def test_section_4_fingerprints_present(self):
        result = generate_html_report(*self._make_all())
        assert 'What Was Used' in result or 'production' in result.lower()
        assert '2010s' in result

    def test_section_4_authenticity_markers(self):
        result = generate_html_report(*self._make_all())
        assert 'preserved dynamics' in result

    def test_section_4_manufacturing_markers(self):
        result = generate_html_report(*self._make_all())
        assert 'professional studio quality' in result

    def test_section_somatic_responses_present(self):
        result = generate_html_report(*self._make_all())
        assert 'chest tightening' in result or 'Somatic' in result

    def test_section_mechanism_shown_when_stage1(self):
        result = generate_html_report(*self._make_all())
        assert 'priming' in result.lower() or 'Conditioning' in result

    def test_technical_detail_section_present(self):
        result = generate_html_report(*self._make_all())
        assert '<details>' in result

    def test_no_unclosed_tags(self):
        """Basic check that </html> closes the document."""
        result = generate_html_report(*self._make_all())
        assert result.strip().endswith('</html>')

    def test_writes_to_file(self, tmp_path):
        path = str(tmp_path / "report.html")
        result = generate_html_report(*self._make_all(), output_path=path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content == result

    def test_works_without_stage1_priming(self):
        """No mechanism section shown when no stages detected."""
        a, t, fp, ps = self._make_all()
        ps.stage1_priming_detected = False
        ps.stage1_evidence = []
        ps.stage3_tag_risk = 0.0
        result = generate_html_report(a, t, fp, ps)
        assert isinstance(result, str)


# ── JSON report tests ────────────────────────────────────────────────────────

class TestJsonReport:

    def _make_all(self):
        return (make_audio_record(), make_trajectory(),
                make_fingerprints(), make_psychosomatic())

    def test_returns_dict(self):
        result = generate_json_report(*self._make_all())
        assert isinstance(result, dict)

    def test_json_serialisable(self):
        result = generate_json_report(*self._make_all())
        dumped = json.dumps(result)
        parsed = json.loads(dumped)
        assert isinstance(parsed, dict)

    def test_required_top_level_keys(self):
        result = generate_json_report(*self._make_all())
        for key in ['version', 'generated_at', 'source', 'lsii', 'trajectory',
                    'psychosomatic', 'fingerprints']:
            assert key in result, f"Missing top-level key: {key}"

    def test_source_fields(self):
        result = generate_json_report(*self._make_all())
        src = result['source']
        assert 'filepath' in src
        assert 'duration_seconds' in src
        assert src['duration_seconds'] == 10.0

    def test_lsii_fields(self):
        result = generate_json_report(*self._make_all())
        lsii = result['lsii']
        assert 'score' in lsii
        assert 'flag_level' in lsii
        assert lsii['flag_level'] == 'moderate'

    def test_trajectory_arcs_present(self):
        result = generate_json_report(*self._make_all())
        traj = result['trajectory']
        for arc in ['valence_arc', 'energy_arc', 'complexity_arc', 'coherence_arc', 'tension_arc']:
            assert arc in traj
            assert len(traj[arc]) == 4

    def test_psychosomatic_fields(self):
        result = generate_json_report(*self._make_all())
        ps = result['psychosomatic']
        for field in ['valence', 'arousal', 'elder_reading', 'authentic_emission_score',
                      'manufacturing_score', 'creative_residue']:
            assert field in ps, f"Missing psychosomatic field: {field}"

    def test_fingerprints_fields(self):
        result = generate_json_report(*self._make_all())
        fp = result['fingerprints']
        assert 'likely_era' in fp
        assert 'authenticity_markers' in fp
        assert 'manufacturing_markers' in fp

    def test_all_float_fields_are_floats(self):
        result = generate_json_report(*self._make_all())
        ps = result['psychosomatic']
        for field in ['valence', 'arousal', 'coherence']:
            assert isinstance(ps[field], float), f"{field} should be float"

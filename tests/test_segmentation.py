"""
Tests for core/segmentation.py — Quarter-based segment splitting.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {name}. Run tests/generate_fixtures.py first.")
    return path


from core.ingestion import load
from core.segmentation import segment, Segment, SegmentationResult


class TestSegmentation:

    def test_segment_returns_four_quarters(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        result = segment(record)
        assert isinstance(result, SegmentationResult)
        assert len(result.quarters) == 4

    def test_segments_cover_full_duration(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        result = segment(record)
        first_start = result.quarters[0].start_time
        last_end = result.quarters[-1].end_time
        assert first_start == pytest.approx(0.0, abs=0.5)
        assert last_end == pytest.approx(record.duration_seconds, abs=1.0)

    def test_quarters_labelled_q1_to_q4(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        result = segment(record)
        labels = [s.label for s in result.quarters]
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            assert any(q in label for label in labels), f"{q} not found in labels"

    def test_each_segment_has_audio_data(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        result = segment(record)
        for seg in result.quarters:
            assert seg.y is not None
            assert len(seg.y) > 0

    def test_short_clip_still_segments(self):
        """Short clips should degrade gracefully — fewer segments or shorter quarters."""
        path = _fixture('short_3s.wav')
        record = load(path)
        result = segment(record)
        assert len(result.quarters) >= 1

    def test_lsii_fixture_q4_different_from_q1(self):
        """The LSII test fixture has a different pitch in Q4 — segments should differ."""
        path = _fixture('lsii_test.wav')
        record = load(path)
        result = segment(record)
        assert len(result.quarters) == 4
        # Q1 and Q4 are the same length but different audio
        q1_energy = np.sum(result.quarters[0].y ** 2)
        q4_energy = np.sum(result.quarters[3].y ** 2)
        # Q4 is quieter (0.2 amplitude vs 0.5) so energy should differ
        assert q1_energy != pytest.approx(q4_energy, rel=0.05)

    def test_segment_objects_have_required_attributes(self):
        path = _fixture('sine_60s.wav')
        record = load(path)
        result = segment(record)
        for seg in result.quarters:
            assert hasattr(seg, 'label')
            assert hasattr(seg, 'start_time')
            assert hasattr(seg, 'end_time')
            assert hasattr(seg, 'y')
            assert hasattr(seg, 'sample_rate')

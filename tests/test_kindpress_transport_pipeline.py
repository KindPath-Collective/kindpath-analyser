import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kindpress.transport import (
    DEFAULT_PROFILE,
    MINIMAL_PROFILE,
    benchmark_transport,
    decode_from_transport,
    encode_for_transport,
)
from seedbank.nanomap import build_kindpress_extraction_state


def _profile(lsii=0.5, valence=0.2, arousal=0.7):
    return {
        "generated_at": "2026-03-13T00:00:00+00:00",
        "source": {"filepath": "tests/fixtures/lsii_test.wav"},
        "lsii": {"score": lsii},
        "psychosomatic": {
            "valence": valence,
            "arousal": arousal,
            "coherence": 0.8,
            "complexity": 0.5,
            "creative_residue": 0.4,
            "authentic_emission_score": 0.7,
            "manufacturing_score": 0.3,
            "identity_capture_risk": 0.2,
            "stage3_tag_risk": 0.1,
            "authenticity_index": 0.65,
            "tension_resolution_ratio": 0.6,
        },
        "trajectory": {"valence_arc": [0.1, 0.2, 0.3, 0.2]},
        "fingerprints": {"production_context": "test"},
        "influence_chain": {"mechanic_direction": "neutral"},
    }


def _baseline():
    return {
        "_tag_version": "baseline:v1",
        "score": 0.3,
        "valence": 0.0,
        "arousal": 0.5,
        "coherence": 0.6,
        "complexity": 0.4,
        "creative_residue": 0.3,
        "authentic_emission_score": 0.6,
        "manufacturing_score": 0.4,
        "identity_capture_risk": 0.3,
        "stage3_tag_risk": 0.2,
        "authenticity_index": 0.5,
        "tension_resolution_ratio": 0.7,
    }


def test_transport_encode_decode_roundtrip_reasonable_error():
    profile = _profile()
    baseline = _baseline()

    payload = encode_for_transport(profile, baseline, DEFAULT_PROFILE)
    decoded = decode_from_transport(payload, baseline)

    assert "valence" in decoded
    assert abs(decoded["valence"] - profile["psychosomatic"]["valence"]) < 0.01
    assert abs(decoded["arousal"] - profile["psychosomatic"]["arousal"]) < 0.01


def test_minimal_profile_payload_smaller_than_default():
    profile = _profile()
    baseline = _baseline()

    p_default = encode_for_transport(profile, baseline, DEFAULT_PROFILE)
    p_minimal = encode_for_transport(profile, baseline, MINIMAL_PROFILE)

    n_default = len(json.dumps(p_default).encode("utf-8"))
    n_minimal = len(json.dumps(p_minimal).encode("utf-8"))

    assert n_minimal <= n_default


def test_benchmark_transport_returns_expected_metrics():
    records = [_profile(lsii=0.4), _profile(lsii=0.8, valence=0.4, arousal=0.9)]
    baseline = _baseline()

    report = benchmark_transport(records, baseline)
    assert report
    for _, metrics in report.items():
        assert "payload_bytes_mean" in metrics
        assert "reconstruction_error_mean" in metrics
        assert "vs_json_ratio" in metrics


def test_extraction_state_includes_whole_and_fragmented_sections():
    profile = _profile()
    clone_meta = {
        "packet_sha256": "abc",
        "packet_hash": "def",
        "packet_k_version": "kindpath:runtime:v1",
        "packet_path": "/tmp/packet.json",
    }
    warning_fingerprints = [
        {
            "index": 1,
            "raw_sha256": "wraw",
            "packet_sha256": "wpacket",
            "packet_hash": "whash",
            "packet_k_version": "kindpath:warning:v1",
        }
    ]

    state = build_kindpress_extraction_state(
        run_id="run_test",
        profile=profile,
        clone_meta=clone_meta,
        warning_fingerprints=warning_fingerprints,
    )

    assert "whole_dataset" in state
    assert "fragmented_datasets" in state
    assert state["whole_dataset"]["kindpress_k_version"] == "kindpath:runtime:v1"
    assert any(f.get("kind") == "warning_packet" for f in state["fragmented_datasets"])

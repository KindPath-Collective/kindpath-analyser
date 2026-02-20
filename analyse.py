#!/usr/bin/env python3
"""
KindPath Creative Analyser
==========================
A Frequency Field Scientist & Creative Seedbank

Usage:
    python analyse.py --file track.mp3
    python analyse.py --file track.mp3 --full-provenance
    python analyse.py --file track.mp3 --deposit --context "Independent, 2019"
    python analyse.py --corpus ./music_folder --output report.json

This is open source. The knowledge belongs to everyone.
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import load
from core.segmentation import segment
from core.feature_extractor import extract
from core.divergence import compute_lsii, compute_trajectory
from core.fingerprints import analyse_fingerprints


def analyse_file(filepath: str, full_provenance: bool = False,
                 deposit: bool = False, context: str = None) -> dict:
    """
    Run the full analysis pipeline on a single file.
    Returns a structured profile dict.
    """
    print(f"\n{'='*60}")
    print(f"  KindPath Creative Analyser")
    print(f"  {filepath}")
    print(f"{'='*60}")

    # ── STAGE 1: INGESTION ──────────────────────────────────────
    record = load(filepath)
    if record.is_silence:
        print("ERROR: File appears silent. Aborting.")
        return {}

    # ── STAGE 2: SEGMENTATION ───────────────────────────────────
    segments = segment(record)

    # ── STAGE 3: FEATURE EXTRACTION ─────────────────────────────
    print(f"\n[FEATURE EXTRACTION]")
    quarter_features = []
    for q in segments.quarters:
        print(f"  Extracting {q.label}...", end=' ', flush=True)
        features = extract(q, record.sample_rate)
        quarter_features.append(features)
        print(f"✓ key:{features.harmonic.key_estimate} "
              f"tempo:{features.temporal.tempo_bpm:.0f}bpm "
              f"tension:{features.harmonic.tension_ratio:.2f}")

    # ── STAGE 4: DIVERGENCE / LSII ──────────────────────────────
    print(f"\n[LATE-SONG INVERSION INDEX]")
    trajectory = compute_trajectory(quarter_features)
    lsii = trajectory.lsii_result

    print(f"  LSII Score   : {lsii.lsii:.4f}  [{lsii.flag_level.upper()}]")
    print(f"  Direction    : {lsii.direction}")
    print(f"  Dominant Axis: {lsii.dominant_axis}")
    print(f"  Trajectory   : {lsii.trajectory_description}")
    print(f"  Inversion    : {lsii.inversion_description}")
    print(f"  Flag Notes   : {lsii.flag_notes}")

    # ── STAGE 5: FINGERPRINTING ──────────────────────────────────
    print(f"\n[FINGERPRINT ANALYSIS]")
    q_all_features = quarter_features[0] if quarter_features else None
    groove_ms = q_all_features.temporal.groove_deviation_ms if q_all_features else None
    crest_db = q_all_features.dynamic.crest_factor_db if q_all_features else None
    dynamic_range = q_all_features.dynamic.dynamic_range_db if q_all_features else None

    fingerprints = analyse_fingerprints(
        record,
        groove_deviation_ms=groove_ms,
        crest_factor_db=crest_db,
        dynamic_range_db=dynamic_range,
    )

    print(f"  Production   : {fingerprints.production_context}")
    if fingerprints.likely_era:
        print(f"  Era Matches  : {', '.join([f'{m.name}({m.confidence:.2f})' for m in fingerprints.likely_era[:2]])}")
    if fingerprints.likely_techniques:
        for t in fingerprints.likely_techniques[:3]:
            print(f"  Technique    : {t.name} (confidence: {t.confidence:.2f})")
    if fingerprints.authenticity_markers:
        print(f"  Authenticity : {'; '.join(fingerprints.authenticity_markers[:2])}")
    if fingerprints.manufacturing_markers:
        print(f"  ⚠ Manufacture: {'; '.join(fingerprints.manufacturing_markers)}")

    # ── STAGE 6: PSYCHOSOMATIC PROFILE ──────────────────────────
    print(f"\n[PSYCHOSOMATIC PROFILE]")
    print(f"  Valence Arc  : {' → '.join([f'{v:.2f}' for v in trajectory.valence_arc])}")
    print(f"  Energy Arc   : {' → '.join([f'{v:.2f}' for v in trajectory.energy_arc])}")
    print(f"  Complexity   : {' → '.join([f'{v:.2f}' for v in trajectory.complexity_arc])}")
    print(f"  Tension Arc  : {' → '.join([f'{v:.2f}' for v in trajectory.tension_arc])}")
    print(f"  Coherence    : {' → '.join([f'{v:.2f}' for v in trajectory.coherence_arc])}")

    # ── ASSEMBLE PROFILE ─────────────────────────────────────────
    profile = {
        "metadata": {
            "analysed_at": datetime.now().isoformat(),
            "tool": "KindPath Creative Analyser v0.1",
            "filepath": filepath,
            "filename": record.filename,
            "duration_seconds": record.duration_seconds,
            "sample_rate": record.sample_rate,
        },
        "audio_health": {
            "peak_db": float(20 * __import__('numpy').log10(record.peak_amplitude + 1e-10)),
            "dynamic_range_db": record.dynamic_range_db,
            "is_clipped": record.is_clipped,
            "clipping_pct": record.clipping_percentage,
        },
        "lsii": {
            "score": lsii.lsii,
            "flag_level": lsii.flag_level,
            "direction": lsii.direction,
            "dominant_axis": lsii.dominant_axis,
            "trajectory": lsii.trajectory_description,
            "inversion": lsii.inversion_description,
            "flag_notes": lsii.flag_notes,
        },
        "trajectory": {
            "valence_arc": trajectory.valence_arc,
            "energy_arc": trajectory.energy_arc,
            "complexity_arc": trajectory.complexity_arc,
            "tension_arc": trajectory.tension_arc,
            "coherence_arc": trajectory.coherence_arc,
        },
        "fingerprints": {
            "production_context": fingerprints.production_context,
            "era_matches": [
                {"name": m.name, "confidence": m.confidence, "description": m.description}
                for m in fingerprints.likely_era
            ],
            "techniques": [
                {"name": m.name, "confidence": m.confidence, "description": m.description}
                for m in fingerprints.likely_techniques
            ],
            "instruments": [
                {"name": m.name, "confidence": m.confidence, "description": m.description}
                for m in fingerprints.likely_instruments
            ],
            "authenticity_markers": fingerprints.authenticity_markers,
            "manufacturing_markers": fingerprints.manufacturing_markers,
        },
    }

    if full_provenance:
        profile["quarter_features"] = [
            qf.to_dict() for qf in quarter_features if qf
        ]

    # ── DEPOSIT TO SEEDBANK ──────────────────────────────────────
    if deposit:
        _deposit_to_seedbank(profile, context)

    return profile


def _deposit_to_seedbank(profile: dict, context: str):
    """Add this profile to the seedbank records."""
    os.makedirs("seedbank/records", exist_ok=True)
    filename = profile['metadata']['filename'].replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    deposit_path = f"seedbank/records/{timestamp}_{filename}.json"

    profile['seedbank_context'] = context or "No context provided"
    with open(deposit_path, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\n[SEEDBANK] Profile deposited: {deposit_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KindPath Creative Analyser - A Frequency Field Scientist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyse.py --file track.mp3
  python analyse.py --file track.mp3 --full-provenance
  python analyse.py --file track.mp3 --deposit --context "Independent release, no label, 2019"
  python analyse.py --corpus ./music --output corpus_report.json
        """
    )
    parser.add_argument('--file', type=str, help='Audio file to analyse')
    parser.add_argument('--corpus', type=str, help='Directory of audio files to batch analyse')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--full-provenance', action='store_true',
                        help='Include full per-quarter feature data in output')
    parser.add_argument('--deposit', action='store_true',
                        help='Deposit profile to the seedbank')
    parser.add_argument('--context', type=str,
                        help='Context for seedbank deposit (release circumstances, etc.)')

    args = parser.parse_args()

    if not args.file and not args.corpus:
        parser.print_help()
        sys.exit(1)

    if args.file:
        profile = analyse_file(
            args.file,
            full_provenance=args.full_provenance,
            deposit=args.deposit,
            context=args.context,
        )
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(profile, f, indent=2)
            print(f"\n[OUTPUT] Profile saved to {args.output}")
        else:
            print(f"\n[PROFILE SUMMARY]")
            print(json.dumps({
                "filename": profile.get('metadata', {}).get('filename'),
                "lsii_score": profile.get('lsii', {}).get('score'),
                "lsii_flag": profile.get('lsii', {}).get('flag_level'),
                "era": profile.get('fingerprints', {}).get('era_matches', [{}])[0].get('name') if profile.get('fingerprints', {}).get('era_matches') else 'unknown',
                "authenticity_markers": profile.get('fingerprints', {}).get('authenticity_markers', []),
                "manufacturing_markers": profile.get('fingerprints', {}).get('manufacturing_markers', []),
            }, indent=2))

    elif args.corpus:
        extensions = {'.mp3', '.wav', '.flac', '.aiff', '.aif', '.ogg', '.m4a'}
        files = [
            os.path.join(args.corpus, f)
            for f in os.listdir(args.corpus)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        print(f"Found {len(files)} audio files in {args.corpus}")
        corpus_profiles = []
        for filepath in sorted(files):
            try:
                profile = analyse_file(filepath, full_provenance=args.full_provenance)
                corpus_profiles.append(profile)
            except Exception as e:
                print(f"ERROR processing {filepath}: {e}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(corpus_profiles, f, indent=2)
            print(f"\n[CORPUS] {len(corpus_profiles)} profiles saved to {args.output}")


if __name__ == "__main__":
    main()

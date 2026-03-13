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
from core.psychosomatics import build_psychosomatic_profile
from core.influence_mapper import map_influence_chain


def analyse_file(filepath: str, full_provenance: bool = False,
                 deposit: bool = False, context: str = None,
                 html_output: str = None) -> dict:
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
            "tool": "KindPath Creative Analyser v0.2",
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
        "psychosomatic": {
            "valence": psycho.valence,
            "arousal": psycho.arousal,
            "coherence": psycho.coherence,
            "authenticity_index": psycho.authenticity_index,
            "complexity": psycho.complexity,
            "tension_resolution_ratio": psycho.tension_resolution_ratio,
            "authentic_emission_score": psycho.authentic_emission_score,
            "manufacturing_score": psycho.manufacturing_score,
            "creative_residue": psycho.creative_residue,
            "identity_capture_risk": psycho.identity_capture_risk,
            "stage1_priming_detected": psycho.stage1_priming_detected,
            "stage2_prestige_attached": psycho.stage2_prestige_attached,
            "stage3_tag_risk": psycho.stage3_tag_risk,
            "somatic_summary": psycho.somatic_summary,
            "mechanism_summary": psycho.mechanism_summary,
            "elder_reading": psycho.elder_reading,
            "predicted_physical_responses": psycho.predicted_physical_responses,
            "predicted_emotional_states": psycho.predicted_emotional_states,
        },
        "influence_chain": {
            "primary_lineage": [
                {"id": n.id, "name": n.name, "confidence": n.confidence,
                 "era_range": list(n.era_range), "kindpath_notes": n.kindpath_notes}
                for n in influence_chain.primary_lineage
            ] if influence_chain else [],
            "narrative": influence_chain.narrative if influence_chain else "",
            "detected_mechanics": [
                {"name": m.name, "direction": m.direction,
                 "confidence": m.confidence, "evidence": m.evidence}
                for m in (influence_chain.detected_mechanics if influence_chain else [])
            ],
            "mechanic_direction": influence_chain.mechanic_direction if influence_chain else "neutral",
            "gradient_stage_estimate": influence_chain.gradient_stage_estimate if influence_chain else "INFILTRATE",
            "syntropy_repair_vectors": influence_chain.syntropy_repair_vectors if influence_chain else [],
            "mechanic_summary": influence_chain.mechanic_summary if influence_chain else "",
        },
    }

    if full_provenance:
        profile["quarter_features"] = [
            qf.to_dict() for qf in quarter_features if qf
        ]

    # ── HTML REPORT ──────────────────────────────────────────────
    if html_output:
        try:
            from reports.report_generator import generate_html_report
            html = generate_html_report(
                record=record,
                trajectory=trajectory,
                fingerprints=fingerprints,
                psychosomatic=psycho,
                output_path=html_output,
            )
            print(f"\n[HTML REPORT] Saved to {html_output}")
        except ImportError:
            print("[HTML REPORT] report_generator not available — skipping")

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


def analyse_file_full(
    filepath: str,
    use_stems: bool = False,
    vocal_analysis: bool = False,
    full_provenance: bool = False,
    deposit: bool = False,
    context: str = None,
    html_output: str = None,
) -> dict:
    """
    Full integrated analysis pipeline.

    Stages:
      1. Ingestion
      2. Segmentation
      3. [Optional] Stem separation (requires demucs)
      4. Feature extraction (per stem if available, else mixed)
      5. [Optional] Vocal prosodic analysis
      6. Divergence / LSII
      7. Fingerprinting
      8. Influence chain mapping (Module 7)
      9. Psychosomatic profiling
      10. Report generation (JSON always, HTML if requested)
      11. [Optional] Seedbank deposit

    Args:
        use_stems: Run stem separation before analysis (requires demucs).
        vocal_analysis: Include deep vocal prosodic analysis on the vocal stem.
        full_provenance: Include per-quarter feature data in output.
        deposit: Deposit profile to the seedbank.
        context: Release context for seedbank deposit.
        html_output: Path to write HTML report.
    """
    # Delegate to analyse_file — use_stems and vocal_analysis are additive
    # extensions that degrade gracefully when dependencies are unavailable
    profile = analyse_file(
        filepath,
        full_provenance=full_provenance,
        deposit=deposit,
        context=context,
        html_output=html_output,
    )

    # Stem separation (optional, requires demucs)
    if use_stems and profile:
        try:
            from core.stem_separator import separate
            from core.ingestion import load as _load
            print("\n[STEM SEPARATION]")
            record = _load(filepath)
            stems = separate(record)
            print(f"  Quality: {stems.separation_quality}")
            profile['stem_separation'] = {
                'model': stems.separation_model,
                'quality': stems.separation_quality,
                'stems_available': [
                    k for k in ['vocals', 'drums', 'bass', 'piano', 'guitar', 'other']
                    if isinstance(getattr(stems, k, None), __import__('numpy').ndarray)
                ],
            }
        except ImportError:
            print("[STEM SEPARATION] demucs not installed — skipping (pip install demucs)")
        except Exception as e:
            print(f"[STEM SEPARATION] Failed: {e} — skipping")

    # Vocal prosodic analysis (optional, requires vocal stem)
    if vocal_analysis and profile:
        try:
            from core.vocal_prosodics import analyse_vocal
            print("\n[VOCAL PROSODICS]")
            # Uses stem separation result if available, else mixed signal
            print("  Vocal prosodic analysis requires stem separation. Run with --stems.")
        except ImportError:
            print("[VOCAL PROSODICS] vocal_prosodics module not available — skipping")

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="KindPath Creative Analyser - A Frequency Field Scientist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyse.py --file track.mp3
  python analyse.py --file track.mp3 --full-provenance
  python analyse.py --file track.mp3 --html report.html
  python analyse.py --file track.mp3 --stems --vocal-analysis
  python analyse.py --file track.mp3 --deposit --context "Independent release, no label, 2019"
  python analyse.py --corpus ./music --output corpus_report.json
  python analyse.py --seedbank-stats
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
    # New flags (AGENTS.md Module 8)
    parser.add_argument('--stems', action='store_true',
                        help='Run stem separation before analysis (requires demucs)')
    parser.add_argument('--vocal-analysis', action='store_true',
                        help='Include deep vocal prosodic analysis')
    parser.add_argument('--html', type=str, metavar='PATH',
                        help='Output HTML report to this path')
    parser.add_argument('--compare', type=str, metavar='SEEDBANK_ID',
                        help='Compare against a seedbank record ID')
    parser.add_argument('--seedbank-stats', action='store_true',
                        help='Show seedbank statistics and exit')
    parser.add_argument('--corpus-trend', type=str, metavar='CORPUS_JSON',
                        help='Generate temporal trend analysis for a corpus JSON file')

    args = parser.parse_args()

    # Seedbank stats (standalone command)
    if args.seedbank_stats:
        try:
            from seedbank.index import get_stats
            stats = get_stats()
            print(json.dumps(stats, indent=2))
        except ImportError:
            print("[SEEDBANK] seedbank module not available")
        sys.exit(0)

    # Corpus trend
    if args.corpus_trend:
        try:
            from core.corpus_analyser import analyse_corpus
            with open(args.corpus_trend) as f:
                profiles = json.load(f)
            result = analyse_corpus(profiles)
            print(f"\n[CORPUS TREND]")
            print(f"  Periods: {', '.join(str(p) for p in result.periods)}")
            print(f"  Summary: {result.trend_summary}")
            if args.output:
                output_data = {
                    'periods': result.periods,
                    'lsii_trend': result.lsii_trend,
                    'authentic_emission_trend': result.authentic_emission_trend,
                    'manufacturing_trend': result.manufacturing_trend,
                    'trend_summary': result.trend_summary,
                    'sos_periods': result.sos_periods,
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n[OUTPUT] Corpus trend saved to {args.output}")
        except ImportError:
            print("[CORPUS TREND] corpus_analyser not available")
        except Exception as e:
            print(f"[CORPUS TREND] Error: {e}")
        sys.exit(0)

    if not args.file and not args.corpus:
        parser.print_help()
        sys.exit(1)

    if args.file:
        profile = analyse_file_full(
            args.file,
            use_stems=getattr(args, 'stems', False),
            vocal_analysis=getattr(args, 'vocal_analysis', False),
            full_provenance=args.full_provenance,
            deposit=args.deposit,
            context=args.context,
            html_output=getattr(args, 'html', None),
        )
        if args.compare:
            try:
                from core.search import compare_to_seedbank
                comparison = compare_to_seedbank(profile, args.compare)
                print(f"\n[COMPARISON vs {args.compare}]")
                print(json.dumps(comparison, indent=2))
            except (ImportError, Exception) as e:
                print(f"[COMPARE] {e}")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(profile, f, indent=2)
            print(f"\n[OUTPUT] Profile saved to {args.output}")
        else:
            print(f"\n[PROFILE SUMMARY]")
            influence = profile.get('influence_chain', {})
            print(json.dumps({
                "filename": profile.get('metadata', {}).get('filename'),
                "lsii_score": profile.get('lsii', {}).get('score'),
                "lsii_flag": profile.get('lsii', {}).get('flag_level'),
                "era": profile.get('fingerprints', {}).get('era_matches', [{}])[0].get('name') if profile.get('fingerprints', {}).get('era_matches') else 'unknown',
                "authenticity_markers": profile.get('fingerprints', {}).get('authenticity_markers', []),
                "manufacturing_markers": profile.get('fingerprints', {}).get('manufacturing_markers', []),
                "mechanic_direction": influence.get('mechanic_direction', 'neutral'),
                "gradient_stage": influence.get('gradient_stage_estimate', ''),
                "syntropy_repair_count": len(influence.get('syntropy_repair_vectors', [])),
                "elder_reading": profile.get('psychosomatic', {}).get('elder_reading', '')[:200],
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

"""
KindPath Analyser :: Report Generator

Takes a complete analysis and produces two outputs:
1. A rich HTML report — human readable, visually informative, teachable
2. A structured JSON export — machine readable, seedbank-compatible

The HTML report IS the educational instrument. It should be accessible to
a curious person with no audio engineering background. Every metric has a
plain-language description of what it means and why it matters.

No mystification. The knowledge belongs to everyone.
"""
from __future__ import annotations

import json
import datetime
from typing import Optional
from dataclasses import asdict

from core.ingestion import AudioRecord
from core.divergence import TrajectoryProfile
from core.fingerprints import FingerprintReport
from core.psychosomatics import PsychosomaticProfile


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(
    record: AudioRecord,
    trajectory: TrajectoryProfile,
    fingerprints: FingerprintReport,
    psychosomatic: PsychosomaticProfile,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a complete HTML analysis report.
    Returns the HTML string. If output_path provided, also writes to file.
    """
    lsii = trajectory.lsii_result.lsii
    flag = trajectory.lsii_result.flag_level

    # Colour the LSII by flag level
    lsii_colour = {
        "none": "#aaaaaa",
        "low": "#cccccc",
        "moderate": "#F5A623",
        "high": "#F5A623",
        "extreme": "#ff8c00",
    }.get(flag, "#aaaaaa")

    filename = record.filename if hasattr(record, 'filename') else record.filepath.split('/')[-1]
    duration = record.duration_seconds if hasattr(record, 'duration_seconds') else getattr(record, 'duration', 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KindPath Analysis — {_escape(filename)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d0d0d; color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 15px; line-height: 1.6; padding: 24px;
      max-width: 820px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.4rem; color: #ffffff; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1rem; color: #cccccc; margin: 28px 0 10px; border-bottom: 1px solid #2a2a2a; padding-bottom: 6px; }}
    h3 {{ font-size: 0.95rem; color: #aaaaaa; margin: 12px 0 4px; }}
    .meta {{ font-size: 0.8rem; color: #666; margin-bottom: 24px; }}
    .elder {{
      background: #161616; border-left: 3px solid {lsii_colour};
      padding: 18px 20px; margin: 10px 0 20px; border-radius: 0 4px 4px 0;
      font-size: 1.05rem; line-height: 1.75; color: #d8d8d8;
    }}
    .lsii-score {{
      font-size: 3.2rem; font-weight: 700; color: {lsii_colour};
      font-variant-numeric: tabular-nums; letter-spacing: -1px;
    }}
    .lsii-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
    .flag {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;
      background: {'#3a2a00' if flag in ('moderate','high','extreme') else '#1a1a1a'};
      color: {lsii_colour}; margin-left: 8px; }}
    .arc-grid {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0;
    }}
    .arc-cell {{
      background: #161616; border-radius: 4px; padding: 12px;
      text-align: center;
    }}
    .arc-cell.q4-flagged {{ border: 1px solid {lsii_colour}; }}
    .arc-label {{ font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
    .arc-val {{ font-size: 1.6rem; font-weight: 600; color: #cccccc; font-variant-numeric: tabular-nums; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
    .bar-label {{ width: 160px; font-size: 0.8rem; color: #888; flex-shrink: 0; }}
    .bar-bg {{ flex: 1; height: 8px; background: #1e1e1e; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: {lsii_colour}; border-radius: 4px; transition: width 0.3s; }}
    .marker-list {{ margin: 6px 0; }}
    .marker {{
      display: inline-block; padding: 2px 8px; border-radius: 12px;
      font-size: 0.75rem; margin: 3px 4px 3px 0; white-space: nowrap;
    }}
    .marker.authentic {{ background: #0d2b1a; color: #4caf50; border: 1px solid #2d5a3a; }}
    .marker.manufacturing {{ background: #2a1a00; color: #F5A623; border: 1px solid #5a3a00; }}
    .somatic-list {{ margin: 6px 0; padding-left: 18px; }}
    .somatic-list li {{ font-size: 0.85rem; color: #999; margin: 3px 0; }}
    .section-explain {{ font-size: 0.8rem; color: #555; margin: 6px 0 12px; font-style: italic; }}
    details {{ margin: 10px 0; }}
    summary {{ cursor: pointer; color: #666; font-size: 0.85rem; user-select: none; padding: 4px 0; }}
    summary:hover {{ color: #aaa; }}
    .detail-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.78rem; }}
    .detail-table th {{ text-align: left; color: #666; padding: 4px 8px; border-bottom: 1px solid #222; }}
    .detail-table td {{ padding: 4px 8px; color: #999; border-bottom: 1px solid #1a1a1a; font-variant-numeric: tabular-nums; }}
    .mechanism-box {{
      background: #1a1200; border: 1px solid #3a2800; border-radius: 4px;
      padding: 14px 16px; margin: 10px 0;
    }}
    .mechanism-box p {{ font-size: 0.85rem; color: #b8a060; line-height: 1.6; }}
    @media (max-width: 600px) {{
      .arc-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .bar-label {{ width: 120px; font-size: 0.72rem; }}
    }}
  </style>
</head>
<body>

<!-- SECTION 1: Identity -->
<h1>{_escape(filename)}</h1>
<div class="meta">
  Duration: {duration:.1f}s &nbsp;|&nbsp;
  Sample rate: {record.sample_rate} Hz &nbsp;|&nbsp;
  Channels: {_num_channels(record)} &nbsp;|&nbsp;
  Analysed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<!-- SECTION 2: The Elder's Reading -->
<h2>The Elder's Reading</h2>
<p class="section-explain">A synthesis of what this piece is doing — audio read as information, not entertainment.</p>
<div class="elder">{_escape(psychosomatic.elder_reading)}</div>

<!-- SECTION 3: Emotional Arc -->
<h2>The Emotional Arc <span class="lsii-score">{lsii:.3f}</span> <span class="flag">{flag}</span></h2>
<p class="section-explain">
  The Late-Song Inversion Index (LSII) measures how much the final quarter of this piece diverges
  from the emotional and sonic trajectory established by the first three quarters.
  A high score means the creator stepped outside the frame they built.
  <strong>Scale:</strong> 0.0–0.2 consistent · 0.2–0.4 notable shift · 0.4–0.6 significant departure · 0.6+ strong inversion.
</p>
{_arc_grid(trajectory, lsii, lsii_colour)}

<h3>Valence &amp; Arousal</h3>
{_bar_row("Emotional valence", (psychosomatic.valence + 1) / 2, lsii_colour)}
{_bar_row("Arousal / activation", psychosomatic.arousal, lsii_colour)}
{_bar_row("Creative coherence", psychosomatic.coherence, lsii_colour)}
{_bar_row("Complexity / density", psychosomatic.complexity, lsii_colour)}

<h3>LSII axis breakdown</h3>
<p class="section-explain">Which feature axis is responsible for the late-section shift?</p>
{_lsii_axes_bars(trajectory, lsii_colour)}

<!-- SECTION 4: What Was Used -->
<h2>What Was Used To Make This</h2>
<p class="section-explain">
  Technique belongs to everyone. What is detectable in a signal was put there by the creator
  and released with the work. These are better instruments to read it.
</p>
{_fingerprint_section(fingerprints)}

<!-- SECTION 5: Somatic Response -->
<h2>Predicted Somatic Response</h2>
<p class="section-explain">What this piece is likely doing to a human body, based on the signal characteristics.</p>
<h3>Physical responses</h3>
<ul class="somatic-list">{''.join(f'<li>{_escape(r)}</li>' for r in psychosomatic.predicted_physical_responses)}</ul>
<h3>Emotional states</h3>
<ul class="somatic-list">{''.join(f'<li>{_escape(s)}</li>' for s in psychosomatic.predicted_emotional_states)}</ul>

{_mechanism_section(psychosomatic)}

<!-- SECTION 5.5: Frequency Field Alignment -->
{_solfeggio_section(psychosomatic)}

<!-- SECTION 6: HMoE Framing -->
{_hmoe_section(psychosomatic, trajectory)}

<!-- SECTION 7: Technical Detail (collapsible) -->
<details>
<summary>Technical detail — feature values per quarter</summary>
{_technical_table(trajectory)}
</details>

<div class="meta" style="margin-top: 32px;">
  Generated by KindPath Analyser · Technique belongs to everyone
</div>

</body>
</html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    return html


# ── JSON export ──────────────────────────────────────────────────────────────

def generate_json_report(
    record: AudioRecord,
    trajectory: TrajectoryProfile,
    fingerprints: FingerprintReport,
    psychosomatic: PsychosomaticProfile,
) -> dict:
    """
    Generate a complete structured JSON profile — the seedbank deposit format.
    """
    lsii = trajectory.lsii_result

    return {
        "version": "1.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
        "source": {
            "filepath": record.filepath,
            "filename": record.filename if hasattr(record, 'filename') else record.filepath.split('/')[-1],
            "duration_seconds": record.duration_seconds if hasattr(record, 'duration_seconds') else getattr(record, 'duration', 0),
            "sample_rate": record.sample_rate,
            "channels": _num_channels(record),
        },
        "lsii": {
            "score": round(lsii.lsii, 4),
            "flag_level": lsii.flag_level,
            "direction": lsii.direction,
            "dominant_axis": lsii.dominant_axis,
            "trajectory_description": lsii.trajectory_description,
            "inversion_description": lsii.inversion_description,
        },
        "trajectory": {
            "valence_arc": [round(v, 4) for v in trajectory.valence_arc],
            "energy_arc": [round(v, 4) for v in trajectory.energy_arc],
            "complexity_arc": [round(v, 4) for v in trajectory.complexity_arc],
            "coherence_arc": [round(v, 4) for v in trajectory.coherence_arc],
            "tension_arc": [round(v, 4) for v in trajectory.tension_arc],
        },
        "psychosomatic": {
            "valence": round(psychosomatic.valence, 4),
            "arousal": round(psychosomatic.arousal, 4),
            "coherence": round(psychosomatic.coherence, 4),
            "authenticity_index": round(psychosomatic.authenticity_index, 4),
            "complexity": round(psychosomatic.complexity, 4),
            "tension_resolution_ratio": round(psychosomatic.tension_resolution_ratio, 4),
            "authentic_emission_score": round(psychosomatic.authentic_emission_score, 4),
            "manufacturing_score": round(psychosomatic.manufacturing_score, 4),
            "creative_residue": round(psychosomatic.creative_residue, 4),
            "identity_capture_risk": round(psychosomatic.identity_capture_risk, 4),
            "stage1_priming_detected": psychosomatic.stage1_priming_detected,
            "stage2_prestige_attached": psychosomatic.stage2_prestige_attached,
            "stage3_tag_risk": round(psychosomatic.stage3_tag_risk, 4),
            "predicted_physical_responses": psychosomatic.predicted_physical_responses,
            "predicted_emotional_states": psychosomatic.predicted_emotional_states,
            "priming_vectors": psychosomatic.priming_vectors,
            "prestige_signals": psychosomatic.prestige_signals,
            "elder_reading": psychosomatic.elder_reading,
            "somatic_summary": psychosomatic.somatic_summary,
            "mechanism_summary": psychosomatic.mechanism_summary,
            "solfeggio_alignment": (
                {
                    "vocal_fundamental_deviation_hz": round(psychosomatic.solfeggio_alignment.vocal_fundamental_deviation_hz, 3),
                    "solfeggio_grid_proximity": round(psychosomatic.solfeggio_alignment.solfeggio_grid_proximity, 4),
                    "nearest_solfeggio_hz": psychosomatic.solfeggio_alignment.nearest_solfeggio_hz,
                    "nearest_solfeggio_name": psychosomatic.solfeggio_alignment.nearest_solfeggio_name,
                    "institutional_distance": round(psychosomatic.solfeggio_alignment.institutional_distance, 4),
                    "tuning_deviation_cents": round(psychosomatic.solfeggio_alignment.tuning_deviation_cents, 2),
                    "alignment_reading": psychosomatic.solfeggio_alignment.alignment_reading,
                    "dominant_solfeggio": psychosomatic.solfeggio_alignment.dominant_solfeggio,
                }
                if psychosomatic.solfeggio_alignment is not None
                else None
            ),
        },
        "fingerprints": {
            "production_context": fingerprints.production_context,
            "authenticity_markers": fingerprints.authenticity_markers,
            "manufacturing_markers": fingerprints.manufacturing_markers,
            "likely_era": [
                {
                    "name": m.name,
                    "confidence": round(m.confidence, 3),
                    "description": m.description,
                    "evidence": m.evidence,
                }
                for m in fingerprints.likely_era
            ],
            "likely_techniques": [
                {
                    "name": m.name,
                    "confidence": round(m.confidence, 3),
                    "description": m.description,
                    "evidence": m.evidence,
                }
                for m in fingerprints.likely_techniques
            ],
            "likely_instruments": [
                {
                    "name": m.name,
                    "confidence": round(m.confidence, 3),
                    "description": m.description,
                }
                for m in fingerprints.likely_instruments
            ],
        },
        "hmoe_framing": {
            "note": "HMoE (Heterogeneous Multiplicity of Evidence) = var(creative_residue) x mean(effective_n). Corpus-level; compute via kindpress.reason.compute_tag_calibration() once deposited.",
            "per_record_inputs": {
                "creative_residue": {
                    "value": round(psychosomatic.creative_residue, 4),
                    "role": "primary_m_signal",
                    "description": "Raw residue after genre/era subtraction. The m in Phi = k*m^n.",
                },
                "authentic_emission_score": {
                    "value": round(psychosomatic.authentic_emission_score, 4),
                    "role": "delta_corroboration",
                    "description": "Validates that creative_residue is genuine signal, not noise.",
                },
                "manufacturing_score": {
                    "value": round(psychosomatic.manufacturing_score, 4),
                    "role": "inverse_residue",
                    "description": "Inverse of authentic signal. High manufacturing = low genuine m contribution.",
                },
                "lsii": {
                    "value": round(trajectory.lsii_result.lsii, 4),
                    "role": "temporal_residue",
                    "description": "Hidden authentic signal that surfaces in the final section.",
                },
            },
        },
    }


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _escape(s: str) -> str:
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))


def _num_channels(record: AudioRecord) -> int:
    if hasattr(record, 'num_channels') and record.num_channels:
        return record.num_channels
    if record.y_stereo is not None:
        try:
            return record.y_stereo.shape[0]
        except Exception:
            pass
    return 1


def _pct(v: float) -> str:
    return f"{max(0.0, min(1.0, float(v))) * 100:.0f}%"


def _bar_row(label: str, value: float, colour: str) -> str:
    pct = _pct(value)
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{_escape(label)}</div>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct};background:{colour}"></div></div>'
        f'<span style="font-size:0.8rem;color:#666;min-width:34px;text-align:right">{float(value):.2f}</span>'
        f'</div>'
    )


def _arc_grid(trajectory: TrajectoryProfile, lsii: float, colour: str) -> str:
    arc = trajectory.energy_arc or [0, 0, 0, 0]
    tension = trajectory.tension_arc or [0, 0, 0, 0]
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    cells = []
    for i in range(min(4, len(arc))):
        flagged = (i == 3 and lsii > 0.35)
        extra_class = ' q4-flagged' if flagged else ''
        cells.append(
            f'<div class="arc-cell{extra_class}">'
            f'<div class="arc-label">{labels[i]}</div>'
            f'<div class="arc-val">{arc[i]:.2f}</div>'
            f'<div class="arc-label" style="margin-top:4px">energy</div>'
            f'<div style="font-size:0.85rem;color:#777;margin-top:2px">tension {tension[i]:.2f}</div>'
            f'</div>'
        )
    return f'<div class="arc-grid">{"".join(cells)}</div>'


def _lsii_axes_bars(trajectory: TrajectoryProfile, colour: str) -> str:
    div = trajectory.lsii_result.divergence
    axes = [
        ("Spectral brightness", div.spectral_centroid_delta),
        ("Dynamic energy", div.dynamic_energy_delta),
        ("Dynamic range", div.dynamic_range_delta),
        ("Harmonic tension", div.harmonic_tension_delta),
        ("Temporal groove", div.temporal_groove_delta),
    ]
    rows = []
    for label, val in axes:
        # Map signed delta to bar width: 0 at left, 0.5 = no change, 1.0 = max
        normalised = (float(val) + 1.0) / 2.0
        rows.append(_bar_row(label, normalised, colour))
    return '\n'.join(rows)


def _fingerprint_section(fp: FingerprintReport) -> str:
    parts = []

    if fp.production_context:
        parts.append(
            f'<p style="color:#999;margin:6px 0">{_escape(fp.production_context)}</p>'
        )

    if fp.authenticity_markers or fp.manufacturing_markers:
        parts.append('<div class="marker-list">')
        for m in fp.authenticity_markers:
            parts.append(f'<span class="marker authentic">{_escape(m)}</span>')
        for m in fp.manufacturing_markers:
            parts.append(f'<span class="marker manufacturing">{_escape(m)}</span>')
        parts.append('</div>')

    if fp.likely_era:
        parts.append('<h3>Production era</h3>')
        for m in fp.likely_era[:3]:
            parts.append(
                f'<div style="margin:6px 0">'
                f'<span style="color:#ccc">{_escape(m.name)}</span>'
                f'<span style="color:#555;font-size:0.8rem;margin-left:8px">{m.confidence:.0%} confidence</span>'
                f'<div style="font-size:0.8rem;color:#666;margin-top:2px">{_escape(m.description)}</div>'
                f'</div>'
            )

    if fp.likely_techniques:
        parts.append('<h3>Detected techniques</h3>')
        for m in fp.likely_techniques[:6]:
            evidence_str = '; '.join(m.evidence[:2]) if m.evidence else ''
            parts.append(
                f'<div style="margin:6px 0">'
                f'<span style="color:#ccc">{_escape(m.name)}</span>'
                f'<span style="color:#555;font-size:0.8rem;margin-left:8px">{m.confidence:.0%}</span>'
                f'<div style="font-size:0.8rem;color:#666">{_escape(m.description)}</div>'
                + (f'<div style="font-size:0.75rem;color:#444;font-style:italic">{_escape(evidence_str)}</div>' if evidence_str else '')
                + '</div>'
            )

    return '\n'.join(parts) if parts else '<p class="section-explain">No fingerprint data available.</p>'


def _mechanism_section(ps: PsychosomaticProfile) -> str:
    if not ps.stage1_priming_detected and not ps.stage2_prestige_attached and ps.stage3_tag_risk < 0.3:
        return ''

    evidence_html = ''
    if ps.stage1_evidence:
        evidence_html += '<h3>Stage 1 — Psychosomatic priming</h3><ul class="somatic-list">'
        evidence_html += ''.join(f'<li>{_escape(e)}</li>' for e in ps.stage1_evidence)
        evidence_html += '</ul>'
    if ps.stage2_evidence:
        evidence_html += '<h3>Stage 2 — Prestige signalling</h3><ul class="somatic-list">'
        evidence_html += ''.join(f'<li>{_escape(e)}</li>' for e in ps.stage2_evidence)
        evidence_html += '</ul>'
    if ps.stage3_evidence:
        evidence_html += f'<h3>Stage 3 — Identity capture risk: {ps.stage3_tag_risk:.0%}</h3><ul class="somatic-list">'
        evidence_html += ''.join(f'<li>{_escape(e)}</li>' for e in ps.stage3_evidence)
        evidence_html += '</ul>'

    return f"""
<h2>Conditioning Mechanics Detected</h2>
<p class="section-explain">
  This is not accusation — it is literacy. These are mechanisms present everywhere in recorded music.
  Naming them is what makes them visible. The mechanism is systemic, not necessarily consciously malicious.
</p>
<div class="mechanism-box">
  <p>{_escape(ps.mechanism_summary)}</p>
</div>
{evidence_html}"""


def _solfeggio_section(psychosomatic: PsychosomaticProfile) -> str:
    """Render the frequency field alignment section (Section 5.5).

    Shows three simultaneous measurements:
    - Vocal fundamental deviation from biological reference (~130 Hz)
    - Solfeggio grid proximity (alignment with pre-institutional harmonic series)
    - Institutional distance (0 = biological, 1 = full A440 institutional frame)

    Returns empty string if no solfeggio_alignment data is available.
    """
    sa = psychosomatic.solfeggio_alignment
    if sa is None:
        return ''

    def _bar(value: float, invert: bool = False) -> str:
        """Render a simple HTML progress bar (0.0–1.0)."""
        pct = int(max(0.0, min(1.0, value)) * 100)
        colour = '#F5A623' if value > 0.6 else '#555'
        return (
            f'<div style="background:#1a1a1a;border-radius:3px;height:10px;width:100%;max-width:300px;">'
            f'<div style="background:{colour};width:{pct}%;height:100%;border-radius:3px;"></div></div>'
        )

    vocal_dev_normalised = min(abs(sa.vocal_fundamental_deviation_hz) / 30.0, 1.0)

    return f"""<section class="solfeggio-section" style="margin-top:32px;">
<h2>Frequency Field Alignment</h2>
<p class="section-explain">
  Three simultaneous readings: how far this piece sits from biological vocal resonance,
  how closely its tonal material aligns with the pre-institutional harmonic series
  (Guido d&#39;Arezzo, ~1025 AD), and its distance from the A440 institutional standard.
</p>

<table class="freq-table" style="width:100%;border-collapse:collapse;margin:16px 0;">
  <thead>
    <tr style="text-align:left;border-bottom:1px solid #333;">
      <th style="padding:8px 12px 8px 0;">Measurement</th>
      <th style="padding:8px 12px;">Value</th>
      <th style="padding:8px 0;">Signal</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #222;">
      <td style="padding:8px 12px 8px 0;">
        <strong>Vocal fundamental deviation</strong><br>
        <small style="color:#888;">Distance from biological mean (~130 Hz). High = tonal centre far from resting human voice.</small>
      </td>
      <td style="padding:8px 12px;">{sa.vocal_fundamental_deviation_hz:+.1f} Hz</td>
      <td style="padding:8px 0;">{_bar(vocal_dev_normalised)}</td>
    </tr>
    <tr style="border-bottom:1px solid #222;">
      <td style="padding:8px 12px 8px 0;">
        <strong>Solfeggio grid proximity</strong><br>
        <small style="color:#888;">How closely prominent frequencies sit within the 396–852 Hz grid. Nearest: {_escape(sa.nearest_solfeggio_name)} ({sa.nearest_solfeggio_hz:.0f} Hz).</small>
      </td>
      <td style="padding:8px 12px;">{sa.solfeggio_grid_proximity:.3f}</td>
      <td style="padding:8px 0;">{_bar(sa.solfeggio_grid_proximity)}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px 8px 0;">
        <strong>Institutional distance</strong><br>
        <small style="color:#888;">0 = biologically centred, 1 = fully anchored to A440 standard. The 1939 ISO standardisation sits at 1.0.</small>
      </td>
      <td style="padding:8px 12px;">{sa.institutional_distance:.3f}</td>
      <td style="padding:8px 0;">{_bar(sa.institutional_distance)}</td>
    </tr>
  </tbody>
</table>

<p style="color:#aaa;font-style:italic;margin-top:8px;">{_escape(sa.alignment_reading)}</p>
</section>"""


def _hmoe_section(psychosomatic: PsychosomaticProfile, trajectory: TrajectoryProfile) -> str:
    """
    Section 6: HMoE (Heterogeneous Multiplicity of Evidence) framing.

    Explains how creative_residue, authentic_emission_score, and LSII each feed
    into the HMoE corpus metric. Per-record inputs only — corpus-level HMoE
    requires seedbank depth and is computed via kindpress.reason.
    """
    cr = psychosomatic.creative_residue
    ae = psychosomatic.authentic_emission_score
    ms = psychosomatic.manufacturing_score
    lsii = trajectory.lsii_result.lsii

    if cr > 0.65:
        cr_reading = "High creative residue — this piece carries significant signal beyond genre convention. It contributes strong discriminating power to any corpus it joins."
    elif cr > 0.35:
        cr_reading = "Moderate creative residue — partially genre-conventional, partially distinctive. Contributes measurable but not exceptional HMoE."
    else:
        cr_reading = "Low creative residue — signal closely matches genre/era conventions. k accounts for most of the variation; low HMoE contribution."

    if ae > 0.65:
        ae_reading = "Authentic emission is strong — the residue is corroborated. Information density is real, not noise."
    elif ae > 0.35:
        ae_reading = "Moderate authentic emission — residue is partially corroborated. Some distinctive signal may be performance variation rather than deliberate authorship."
    else:
        ae_reading = "Low authentic emission — residue may reflect noise or production artifacts rather than genuine creative signal."

    if lsii > 0.4:
        lsii_reading = f"LSII of {lsii:.3f} is significant — hidden residue amplifies toward the end. The creator's authentic signal may be most concentrated in the final quarter."
    else:
        lsii_reading = f"LSII of {lsii:.3f} is low — creative residue is distributed consistently across the piece."

    return (
        '<h2>HMoE Reading</h2>'
        '<p class="section-explain">'
        'HMoE — <em>Heterogeneous Multiplicity of Evidence</em> — measures the information richness '
        'of a corpus. Formula: <strong>var(creative_residue) &times; mean(effective_n)</strong>. '
        'Each piece contributes through the three signals below. A piece with high creative residue, '
        'corroborated by authentic emission, expands the HMoE of any corpus it enters.'
        '</p>'
        '<h3>Creative Residue — primary m input</h3>'
        '<p class="section-explain">What remains after known genre and era signatures are subtracted. The raw m in Φ = km<sup>n</sup>.</p>'
        + _bar_row("Creative residue", cr, "#7ab8f5")
        + f'<p class="section-explain" style="margin-top:4px">{_escape(cr_reading)}</p>'
        '<h3>Authentic Emission — Δ corroboration</h3>'
        '<p class="section-explain">Corroborates that the residue is genuine creative signal, not extraction noise.</p>'
        + _bar_row("Authentic emission", ae, "#7ab8f5")
        + _bar_row("Manufacturing score (inverse)", ms, "#F5A623")
        + f'<p class="section-explain" style="margin-top:4px">{_escape(ae_reading)}</p>'
        '<h3>LSII — temporal residue signal</h3>'
        '<p class="section-explain">The late-song inversion is a form of m with temporal specificity.</p>'
        + f'<p class="section-explain">{_escape(lsii_reading)}</p>'
        '<p class="section-explain" style="margin-top:14px;font-style:normal;color:#444;border-left:2px solid #2a2a2a;padding-left:10px">'
        '<strong>Corpus-level HMoE</strong> requires multiple deposited records. '
        'Run <code>kindpress.reason.compute_tag_calibration(tag_name)</code> once this piece is in the seedbank.'
        '</p>'
    )


def _technical_table(trajectory: TrajectoryProfile) -> str:
    if not trajectory.quarters:
        return '<p style="color:#444;font-size:0.8rem">No quarter data available.</p>'

    rows = ''
    arc_fields = [
        ('valence', trajectory.valence_arc),
        ('energy', trajectory.energy_arc),
        ('complexity', trajectory.complexity_arc),
        ('coherence', trajectory.coherence_arc),
        ('tension', trajectory.tension_arc),
    ]
    for label, arc in arc_fields:
        if arc and len(arc) >= 4:
            cells = ''.join(f'<td>{arc[i]:.3f}</td>' for i in range(4))
            rows += f'<tr><th>{label}</th>{cells}</tr>'

    # Quarter-level raw features if available
    for i, q in enumerate(trajectory.quarters[:4]):
        if isinstance(q, dict):
            for k, v in list(q.items())[:6]:
                try:
                    rows += f'<tr><td>Q{i+1} {_escape(k)}</td><td colspan="3">{float(v):.3f}</td></tr>'
                except (TypeError, ValueError):
                    pass

    return f"""
<table class="detail-table">
  <thead><tr><th>Feature</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

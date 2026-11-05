"""Aerospace Control Room Dashboard, Markdown Dossier & JSON Audit Generator.

Generates standalone TRL 4 qualification artifacts:
- `reports/rad_trl4_scorecard.html`: Interactive dark-theme HTML5 dashboard with
  embedded SVG Weibull cross-section curves, South Atlantic Anomaly (SAA) orbital
  upset profiles, Protected vs Unprotected GNC attitude error trajectories, and
  Fast High-Side SEL Current-Limiting Breaker waveforms.
- `reports/TRL4_DOSSIER.md`: Comprehensive publication-grade Markdown dossier.
- `reports/trl4_audit.json`: Machine-readable cryptographic JSON audit manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from rad_fault_hil.see_physics import COTS_COMPONENT_LIBRARY
from rad_fault_hil.verification import CampaignQualificationBundle


def _svg_polyline(points: List[Tuple[float, float]], color: str, width: float = 2.2, dash: str = "") -> str:
    pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} points="{pts_str}" />'


def _build_weibull_svg() -> str:
    """Render an embedded SVG chart of 4-parameter Weibull SEU cross-sections across COTS components."""
    w, h = 580, 300
    pad_l, pad_r, pad_t, pad_b = 65, 25, 30, 45
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    let_vals = np.logspace(-0.3, 2.0, 120)  # 0.5 to 100 MeV*cm2/mg
    log_let_min, log_let_max = math.log10(0.5), 2.0
    log_sig_min, log_sig_max = -6.0, -0.5

    def to_xy(let_v: float, sig_v: float) -> Tuple[float, float]:
        lx = math.log10(max(0.5, let_v))
        ly = math.log10(max(1e-6, sig_v))
        x = pad_l + (lx - log_let_min) / (log_let_max - log_let_min) * pw
        y = pad_t + ph - (ly - log_sig_min) / (log_sig_max - log_sig_min) * ph
        return x, max(pad_t, min(pad_t + ph, y))

    colors = {
        "STM32H7": "#38bdf8",
        "POLARFIRE_SOC": "#34d399",
        "ZYNQ_ULTRASCALE": "#f472b6",
        "MICRON_DDR4": "#facc15",
        "SIC_MOSFET": "#fb923c",
    }

    lines_svg = []
    legend_svg = []
    idx = 0
    for comp_id, spec in COTS_COMPONENT_LIBRARY.items():
        pts: List[Tuple[float, float]] = []
        for lv in let_vals:
            sv = float(spec.weibull_seu.evaluate(float(lv)))
            if sv > 1e-6:
                pts.append(to_xy(float(lv), sv))
        if pts:
            lines_svg.append(_svg_polyline(pts, colors.get(comp_id, "#ffffff"), width=2.4))
        lx = pad_l + 12 + (idx % 3) * 168
        ly = pad_t + 16 + (idx // 3) * 18
        legend_svg.append(
            f'<rect x="{lx}" y="{ly - 9}" width="12" height="4" fill="{colors.get(comp_id, "#fff")}" />'
            f'<text x="{lx + 16}" y="{ly - 4}" fill="#cbd5e1" font-size="10" font-family="monospace">{comp_id}</text>'
        )
        idx += 1

    grid_svg = []
    for exp_y in range(-6, 0):
        _, gy = to_xy(1.0, 10.0 ** exp_y)
        grid_svg.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + pw}" y2="{gy:.1f}" stroke="#1e293b" stroke-width="1" />'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end" font-family="monospace">10^{exp_y}</text>'
        )
    for let_tick in (1, 2, 5, 10, 20, 50, 100):
        gx, _ = to_xy(float(let_tick), 1e-3)
        grid_svg.append(
            f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + ph}" stroke="#1e293b" stroke-width="1" />'
            f'<text x="{gx:.1f}" y="{pad_t + ph + 16}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="monospace">{let_tick}</text>'
        )

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />
      {''.join(grid_svg)}
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#334155" stroke-width="1.2" />
      {''.join(lines_svg)}
      {''.join(legend_svg)}
      <text x="{pad_l + pw / 2}" y="{h - 8}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">Linear Energy Transfer LET [MeV·cm²/mg]</text>
      <text x="16" y="{pad_t + ph / 2}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 16 {pad_t + ph / 2})" font-family="sans-serif">Weibull σ_SEU(LET) [cm²/device]</text>
    </svg>"""


def _build_orbital_saa_svg(bundle: CampaignQualificationBundle) -> str:
    """Render SVG of orbital SAA flux spike, SPE flare window, and cumulative TID."""
    w, h = 580, 300
    pad_l, pad_r, pad_t, pad_b = 60, 55, 30, 45
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    recs = bundle.step_records
    max_t = max(1.0, recs[-1].time_s)
    max_flux = max(1.0, max(r.total_proton_flux_cm2_s for r in recs))
    max_tid = max(1e-9, max(r.cumulative_tid_krad_si for r in recs))

    flux_pts: List[Tuple[float, float]] = []
    tid_pts: List[Tuple[float, float]] = []
    for r in recs:
        x = pad_l + (r.time_s / max_t) * pw
        yf = pad_t + ph - (r.total_proton_flux_cm2_s / max_flux) * (ph * 0.88)
        yt = pad_t + ph - (r.cumulative_tid_krad_si / max_tid) * (ph * 0.88)
        flux_pts.append((x, yf))
        tid_pts.append((x, yt))

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#334155" stroke-width="1.2" />
      {_svg_polyline(flux_pts, "#f43f5e", width=2.3)}
      {_svg_polyline(tid_pts, "#38bdf8", width=2.2, dash="5,3")}
      <text x="{pad_l + 12}" y="{pad_t + 18}" fill="#f43f5e" font-size="11" font-family="monospace">■ Attenuated Proton Flux (SAA {bundle.radiation_summary.peak_saa_multiplier_observed:.1f}x + SPE Burst)</text>
      <text x="{pad_l + 12}" y="{pad_t + 34}" fill="#38bdf8" font-size="11" font-family="monospace">-- Cumulative Shielded TID ({bundle.radiation_summary.total_tid_krad_si:.4f} krad(Si))</text>
      <text x="{pad_l + pw / 2}" y="{h - 8}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">Orbital Mission Elapsed Time [s]</text>
      <text x="16" y="{pad_t + ph / 2}" fill="#f43f5e" font-size="11" text-anchor="middle" transform="rotate(-90 16 {pad_t + ph / 2})" font-family="sans-serif">Proton Flux [p / (cm²·s)]</text>
    </svg>"""


def _build_gnc_comparison_svg(bundle: CampaignQualificationBundle) -> str:
    """Render SVG comparing Unprotected COTS GNC attitude divergence vs Protected TMR GNC."""
    w, h = 580, 300
    pad_l, pad_r, pad_t, pad_b = 60, 25, 30, 45
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    recs = bundle.step_records
    max_t = max(1.0, recs[-1].time_s)
    max_err = max(10.0, max(r.unprotected_pointing_err_deg for r in recs))

    prot_pts: List[Tuple[float, float]] = []
    unprot_pts: List[Tuple[float, float]] = []
    for r in recs:
        x = pad_l + (r.time_s / max_t) * pw
        yp = pad_t + ph - (r.protected_pointing_err_deg / max_err) * (ph * 0.88)
        yu = pad_t + ph - (r.unprotected_pointing_err_deg / max_err) * (ph * 0.88)
        prot_pts.append((x, yp))
        unprot_pts.append((x, yu))

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#334155" stroke-width="1.2" />
      {_svg_polyline(unprot_pts, "#ef4444", width=2.4)}
      {_svg_polyline(prot_pts, "#10b981", width=2.6)}
      <text x="{pad_l + 12}" y="{pad_t + 18}" fill="#ef4444" font-size="11" font-family="monospace">■ Unprotected Single-String COTS (Tumbles to {max_err:.1f}°)</text>
      <text x="{pad_l + 12}" y="{pad_t + 34}" fill="#10b981" font-size="11" font-family="monospace">■ RAD-FAULT-HIL Protected TMR + SEC-DED (Final {recs[-1].protected_pointing_err_deg:.4f}°)</text>
      <text x="{pad_l + pw / 2}" y="{h - 8}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">Orbital Mission Elapsed Time [s]</text>
      <text x="18" y="{pad_t + ph / 2}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 18 {pad_t + ph / 2})" font-family="sans-serif">Attitude Pointing Error [deg]</text>
    </svg>"""


def _build_sel_breaker_svg(bundle: CampaignQualificationBundle) -> str:
    """Render SVG of microsecond SEL parasitic thyristor surge & fast high-side breaker cutoff."""
    w, h = 580, 300
    pad_l, pad_r, pad_t, pad_b = 60, 25, 30, 45
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    wf = bundle.sel_waveform
    max_t = float(wf.time_us[-1])
    max_i = max(5.5, float(np.max(wf.unprotected_current_a)))

    prot_pts: List[Tuple[float, float]] = []
    unprot_pts: List[Tuple[float, float]] = []
    for idx in range(0, len(wf.time_us), 2):
        t = float(wf.time_us[idx])
        x = pad_l + (t / max_t) * pw
        yp = pad_t + ph - (float(wf.protected_current_a[idx]) / max_i) * (ph * 0.88)
        yu = pad_t + ph - (float(wf.unprotected_current_a[idx]) / max_i) * (ph * 0.88)
        prot_pts.append((x, yp))
        unprot_pts.append((x, yu))

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#334155" stroke-width="1.2" />
      {_svg_polyline(unprot_pts, "#f97316", width=2.2, dash="6,3")}
      {_svg_polyline(prot_pts, "#22d3ee", width=2.6)}
      <text x="{pad_l + 12}" y="{pad_t + 18}" fill="#f97316" font-size="11" font-family="monospace">-- Unprotected Latchup Surge ({wf.total_unprotected_energy_uj:.1f} μJ &gt; {wf.damage_threshold_energy_uj:.0f} μJ Burnout!)</text>
      <text x="{pad_l + 12}" y="{pad_t + 34}" fill="#22d3ee" font-size="11" font-family="monospace">■ Fast High-Side Breaker Cutoff ({wf.cutoff_complete_time_us:.2f} μs, {wf.total_protected_energy_uj:.1f} μJ, {wf.thermal_survival_margin_ratio:.1f}x Margin)</text>
      <text x="{pad_l + pw / 2}" y="{h - 8}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">Time Since Parasitic PNPN Latchup Trigger [μs]</text>
      <text x="18" y="{pad_t + ph / 2}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 18 {pad_t + ph / 2})" font-family="sans-serif">VDD Rail Current I_SEL [A]</text>
    </svg>"""


def generate_all_qualification_reports(
    bundle: CampaignQualificationBundle,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Generate HTML5 scorecard dashboard, Markdown dossier, and JSON audit manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    html_file = out_path / "rad_trl4_scorecard.html"
    md_file = out_path / "TRL4_DOSSIER.md"
    json_file = out_path / "trl4_audit.json"

    # 1. Build JSON Audit Manifest
    audit_payload: Dict[str, Any] = {
        "project": "RAD-FAULT-HIL",
        "trl_level": "TRL 4 Validated Breadboard & HIL Fault-Injection Testbed",
        "all_pillars_passed": bundle.all_pillars_passed,
        "sha256_digest": bundle.sha256_digest,
        "seed": bundle.seed,
        "orbit_regime": bundle.regime.value,
        "shielding": {
            "aluminum_mm": bundle.shielding.aluminum_mm,
            "tantalum_mm": bundle.shielding.tantalum_mm,
            "areal_density_g_cm2": round(bundle.shielding.areal_density_g_cm2, 4),
            "al_equivalent_mm": round(bundle.shielding.al_equivalent_mm, 4),
        },
        "radiation_transport_metrics": {
            "peak_saa_multiplier": round(bundle.radiation_summary.peak_saa_multiplier_observed, 3),
            "shielded_tid_krad_si": round(bundle.radiation_summary.total_tid_krad_si, 6),
            "unshielded_tid_krad_si": round(bundle.unshielded_radiation_summary.total_tid_krad_si, 6),
            "saa_transit_steps": bundle.radiation_summary.saa_transit_steps,
            "spe_transit_steps": bundle.radiation_summary.spe_transit_steps,
        },
        "campaign_fault_injection_summary": {
            "target_component_id": bundle.target_component_id,
            "total_injected_events": len(bundle.fault_schedule),
            "seu_corrected_by_secded": bundle.step_records[-1].cumulative_seu_corrected,
            "mbu_detected_by_secded": bundle.step_records[-1].cumulative_mbu_detected,
            "tmr_voter_resync_events": bundle.step_records[-1].cumulative_tmr_resyncs,
            "sel_breaker_trips": bundle.step_records[-1].cumulative_sel_trips,
            "final_protected_pointing_error_deg": round(
                bundle.step_records[-1].protected_pointing_err_deg, 5
            ),
            "final_unprotected_pointing_error_deg": round(
                bundle.step_records[-1].unprotected_pointing_err_deg, 3
            ),
            "final_protected_corrupted_words": bundle.step_records[-1].protected_corrupted_words,
            "final_unprotected_corrupted_words": bundle.step_records[-1].unprotected_corrupted_words,
            "sel_protected_energy_uj": round(bundle.sel_waveform.total_protected_energy_uj, 3),
            "sel_unprotected_energy_uj": round(bundle.sel_waveform.total_unprotected_energy_uj, 3),
            "sel_damage_threshold_uj": bundle.sel_waveform.damage_threshold_energy_uj,
            "sel_thermal_survival_margin": round(bundle.sel_waveform.thermal_survival_margin_ratio, 3),
        },
        "mtbue_analysis": {k: round(v, 3) for k, v in bundle.mtbue_metrics.items()},
        "cots_component_library_rates": {
            cid: {
                "part_number": s.part_number,
                "expected_seu_per_day": round(s.expected_seu_per_day, 5),
                "expected_mbu_per_day": round(s.expected_mbu_per_day, 6),
                "expected_sefi_per_day": round(s.expected_sefi_per_day, 6),
                "expected_sel_per_day": round(s.expected_sel_per_day, 6),
                "tid_margin_ratio": round(s.tid_margin_ratio, 2),
            }
            for cid, s in bundle.component_see_summaries.items()
        },
        "pillars": [
            {
                "pillar_id": p.pillar_id,
                "title": p.title,
                "passed": p.passed,
                "measured_value": p.measured_value,
                "threshold_description": p.threshold_description,
                "unit": p.unit,
                "details": p.details,
            }
            for p in bundle.pillars
        ],
    }
    json_file.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    # 2. Build Markdown Dossier
    pillar_rows_md = "\n".join(
        f"| `{p.pillar_id}` | {p.title} | **{p.measured_value} {p.unit}** | {p.threshold_description} | {'PASS' if p.passed else 'FAIL'} |"
        for p in bundle.pillars
    )
    comp_rows_md = "\n".join(
        f"| `{cid}` | {s.part_number} | {s.expected_seu_per_day:.4f} | {s.expected_mbu_per_day:.5f} | {s.expected_sefi_per_day:.5f} | {s.expected_sel_per_day:.6f} | {s.tid_margin_ratio:.1f}x |"
        for cid, s in bundle.component_see_summaries.items()
    )

    md_content = f"""# RAD-FAULT-HIL: TRL 4 Space Radiation & Avionics Fault-Injection Qualification Dossier

- **Qualification Status**: `{'7/7 PILLARS PASSED (TRL 4 QUALIFIED)' if bundle.all_pillars_passed else 'FAILED'}`
- **Cryptographic Campaign SHA-256**: `{bundle.sha256_digest}`
- **Orbital Regime**: `{bundle.regime.value}` (Graded Enclosure: `{bundle.shielding.aluminum_mm} mm Al + {bundle.shielding.tantalum_mm} mm Ta`, `{bundle.shielding.al_equivalent_mm:.2f} mm Al-eq`)
- **Primary Avionics Target**: `{bundle.target_component_id}` (`{COTS_COMPONENT_LIBRARY[bundle.target_component_id].part_number}`)

---

## 1. Executive Summary: Unprotected COTS vs. RAD-FAULT-HIL Protected Architecture

| Qualification Metric | Unprotected Single-String COTS | RAD-FAULT-HIL Protected COTS (TMR + SEC-DED + Fast SEL Breaker) |
| :--- | :--- | :--- |
| **GNC Attitude Pointing Error (Post-SAA + SPE)** | `{bundle.step_records[-1].unprotected_pointing_err_deg:.2f} deg` (Catastrophic Loss of Pointing) | **`{bundle.step_records[-1].protected_pointing_err_deg:.4f} deg`** (Nominal Fine Pointing) |
| **SRAM Corrupted Words (End of Campaign)** | `{bundle.step_records[-1].unprotected_corrupted_words} / 256 words corrupted` | **`{bundle.step_records[-1].protected_corrupted_words} / 256 words corrupted`** (`{bundle.step_records[-1].cumulative_seu_corrected}` SEUs scrubbed, `{bundle.step_records[-1].cumulative_mbu_detected}` MBUs reloaded) |
| **SEL Parasitic Latchup Energy Deposited** | `{bundle.sel_waveform.total_unprotected_energy_uj:.1f} uJ` (> `{bundle.sel_waveform.damage_threshold_energy_uj:.0f} uJ` Bond-Wire Melt / Burnout) | **`{bundle.sel_waveform.total_protected_energy_uj:.1f} uJ`** (`{bundle.sel_waveform.cutoff_complete_time_us:.2f} us` Cutoff, **`{bundle.sel_waveform.thermal_survival_margin_ratio:.1f}x`** Thermal Margin) |
| **Memory MTBUE (Mean Time Between Unrecoverable Errors)** | `{bundle.mtbue_metrics['mttf_unprotected_hours']:.2f} h` (MTTF to first silent corruption) | **`{bundle.mtbue_metrics['mtbue_secded_scrubbed_hours']:.1f} h`** (**`{bundle.mtbue_metrics['scrubbing_improvement_factor']:.1f}x`** over unscrubbed SEC-DED) |

---

## 2. Seven-Pillar TRL 4 Verification Scorecard

| Pillar ID | Verification Pillar | Measured Value | Qualification Threshold | Verdict |
| :--- | :--- | :--- | :--- | :--- |
{pillar_rows_md}

---

## 3. Harmonized COTS Semiconductor SEE Rate Table

| Component ID | Device Part Number | SEU / day | MBU / day | SEFI / day | SEL / day | Annual TID Margin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{comp_rows_md}
"""
    md_file.write_text(md_content, encoding="utf-8")

    # 3. Build Standalone Dark-Theme HTML5 Control Room Scorecard
    svg_weibull = _build_weibull_svg()
    svg_saa = _build_orbital_saa_svg(bundle)
    svg_gnc = _build_gnc_comparison_svg(bundle)
    svg_sel = _build_sel_breaker_svg(bundle)

    pillar_rows_html = "".join(
        f"""<tr>
          <td class="mono">{p.pillar_id}</td>
          <td>{p.title}</td>
          <td class="mono val">{p.measured_value} {p.unit}</td>
          <td>{p.threshold_description}</td>
          <td><span class="badge {'pass' if p.passed else 'fail'}">{'PASS' if p.passed else 'FAIL'}</span></td>
        </tr>"""
        for p in bundle.pillars
    )

    comp_rows_html = "".join(
        f"""<tr>
          <td class="mono">{cid}</td>
          <td>{s.part_number}</td>
          <td class="mono">{s.expected_seu_per_day:.4f}</td>
          <td class="mono">{s.expected_mbu_per_day:.5f}</td>
          <td class="mono">{s.expected_sefi_per_day:.5f}</td>
          <td class="mono">{s.expected_sel_per_day:.6f}</td>
          <td class="mono">{s.tid_margin_ratio:.1f}x</td>
        </tr>"""
        for cid, s in bundle.component_see_summaries.items()
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RAD-FAULT-HIL | TRL 4 Space Radiation & Avionics Fault-Injection Control Room</title>
  <style>
    :root {{
      --bg: #070b14;
      --panel: #0f172a;
      --border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --pass: #10b981;
      --fail: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px 32px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0 0 6px 0; font-size: 22px; color: #f8fafc; }}
    .subtitle {{ color: var(--muted); font-size: 13px; }}
    .status-banner {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--pass);
      color: #34d399;
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 700;
      font-family: monospace;
      font-size: 14px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin: 6px 0 4px 0; color: var(--accent); font-family: monospace; }}
    .kpi-sub {{ color: var(--muted); font-size: 12px; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
      margin-bottom: 26px;
    }}
    .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }}
    .chart-title {{ font-size: 14px; font-weight: 600; margin: 0 0 10px 4px; color: #cbd5e1; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 24px;
    }}
    th, td {{
      padding: 11px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
    }}
    th {{ background: #131c31; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    .mono {{ font-family: monospace; }}
    .val {{ color: #38bdf8; font-weight: 600; }}
    .badge {{
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      font-family: monospace;
    }}
    .badge.pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .badge.fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
    h2 {{ font-size: 16px; color: #f1f5f9; margin: 24px 0 12px 0; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>RAD-FAULT-HIL: COTS-to-Orbit Radiation Transport &amp; Avionics Fault-Injection Testbed</h1>
      <div class="subtitle">
        Orbit: <strong>{bundle.regime.value}</strong> | Graded Shielding: <strong>{bundle.shielding.aluminum_mm}mm Al + {bundle.shielding.tantalum_mm}mm Ta</strong> ({bundle.shielding.al_equivalent_mm:.2f}mm Al-eq) | SHA-256: <span class="mono">{bundle.sha256_digest[:24]}...</span>
      </div>
    </div>
    <div class="status-banner">TRL 4 VERIFIED: {sum(1 for p in bundle.pillars if p.passed)}/{len(bundle.pillars)} PILLARS PASSED</div>
  </header>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">SAA Peak Proton Spike</div>
      <div class="kpi-value">{bundle.radiation_summary.peak_saa_multiplier_observed:.1f}x</div>
      <div class="kpi-sub">Al+Ta TID Attenuation: {bundle.unshielded_radiation_summary.total_tid_krad_si / max(1e-9, bundle.radiation_summary.total_tid_krad_si):.2f}x</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">SEC-DED (72,64) Scrubber</div>
      <div class="kpi-value">{bundle.step_records[-1].cumulative_seu_corrected} SEUs Fixed</div>
      <div class="kpi-sub">{bundle.step_records[-1].cumulative_mbu_detected} MBUs Detected | MTBUE Gain: {bundle.mtbue_metrics['scrubbing_improvement_factor']:.0f}x</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">TMR GNC Pointing Error</div>
      <div class="kpi-value">{bundle.step_records[-1].protected_pointing_err_deg:.4f}°</div>
      <div class="kpi-sub">Unprotected COTS Diverges to {bundle.step_records[-1].unprotected_pointing_err_deg:.1f}° ({bundle.step_records[-1].cumulative_tmr_resyncs} TMR Resyncs)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fast SEL Breaker Margin</div>
      <div class="kpi-value">{bundle.sel_waveform.thermal_survival_margin_ratio:.1f}x Safe</div>
      <div class="kpi-sub">Cutoff in {bundle.sel_waveform.cutoff_complete_time_us:.2f} μs ({bundle.sel_waveform.total_protected_energy_uj:.1f} μJ vs {bundle.sel_waveform.damage_threshold_energy_uj:.0f} μJ limit)</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-card">
      <div class="chart-title">1. Heavy-Ion 4-Parameter Weibull Cross-Section σ(LET) Across COTS Library</div>
      {svg_weibull}
    </div>
    <div class="chart-card">
      <div class="chart-title">2. South Atlantic Anomaly (SAA 50x+) &amp; Solar Particle Event (SPE) Orbital Profile</div>
      {svg_saa}
    </div>
    <div class="chart-card">
      <div class="chart-title">3. Unprotected COTS GNC Divergence vs. RAD-FAULT-HIL Protected TMR Voter</div>
      {svg_gnc}
    </div>
    <div class="chart-card">
      <div class="chart-title">4. Fast High-Side SEL Current-Limiting Breaker Microsecond Cutoff Waveform</div>
      {svg_sel}
    </div>
  </div>

  <h2>7-Pillar TRL 4 Verification &amp; Qualification Scorecard</h2>
  <table>
    <thead>
      <tr>
        <th>Pillar ID</th>
        <th>Qualification Requirement</th>
        <th>Measured Metric</th>
        <th>Acceptance Threshold</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {pillar_rows_html}
    </tbody>
  </table>

  <h2>COTS Semiconductor Component Radiation Rate &amp; TID Margin Benchmark</h2>
  <table>
    <thead>
      <tr>
        <th>Component ID</th>
        <th>Part Number &amp; Architecture</th>
        <th>Expected SEU / Day</th>
        <th>Expected MBU / Day</th>
        <th>Expected SEFI / Day</th>
        <th>Expected SEL / Day</th>
        <th>Annual TID Margin</th>
      </tr>
    </thead>
    <tbody>
      {comp_rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    html_file.write_text(html_content, encoding="utf-8")

    return {
        "html": html_file,
        "markdown": md_file,
        "json": json_file,
    }

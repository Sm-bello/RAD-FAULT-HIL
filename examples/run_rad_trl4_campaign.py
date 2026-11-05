"""End-to-End Executable Benchmark for RAD-FAULT-HIL TRL 4 Qualification.

Compares an Unprotected COTS Flight Computer (which suffers catastrophic
unrecoverable GNC attitude divergence and destructive SEL thermal burnout during
a South Atlantic Anomaly + Solar Particle Event transit) against the
RAD-FAULT-HIL Protected COTS Architecture (TMR Voter + Extended Hamming SEC-DED
(72,64) Active Scrubber + Fast High-Side SEL Current-Limiting Breaker).
"""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rad_fault_hil.hil_bridge import HilTransportMode
from rad_fault_hil.radiation_env import OrbitRegime
from rad_fault_hil.reporting import generate_all_qualification_reports
from rad_fault_hil.verification import run_qualification_campaign


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "rad_campaign.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    camp_cfg = cfg["campaign"]
    shield_cfg = cfg["shielding"]
    av_cfg = cfg["avionics_target"]

    print("=" * 84)
    print("RAD-FAULT-HIL: COTS-to-Orbit Radiation Transport & Avionics HIL Qualification")
    print("=" * 84)

    bundle = run_qualification_campaign(
        regime=OrbitRegime(camp_cfg["orbit_regime"]),
        aluminum_mm=float(shield_cfg["aluminum_mm"]),
        tantalum_mm=float(shield_cfg["tantalum_mm"]),
        target_component_id=str(av_cfg["component_id"]),
        duration_s=float(camp_cfg["duration_s"]),
        dt_s=float(camp_cfg["dt_s"]),
        memory_words=int(av_cfg["memory_words"]),
        scrub_period_s=float(av_cfg["scrub_period_s"]),
        accelerated_beam_scale=float(camp_cfg["accelerated_beam_scale"]),
        seed=int(camp_cfg["seed"]),
        hil_mode=HilTransportMode(av_cfg["hil_transport_mode"]),
    )

    reports_dir = PROJECT_ROOT / "reports"
    artifact_paths = generate_all_qualification_reports(bundle, reports_dir)

    final_rec = bundle.step_records[-1]
    print(f"Orbit Regime            : {bundle.regime.value} ({bundle.radiation_summary.duration_s:.0f} s)")
    print(
        f"Graded Shielding        : {bundle.shielding.aluminum_mm} mm Al + "
        f"{bundle.shielding.tantalum_mm} mm Ta ({bundle.shielding.al_equivalent_mm:.2f} mm Al-eq)"
    )
    print(f"SAA Peak Flux Spike     : {bundle.radiation_summary.peak_saa_multiplier_observed:.2f}x")
    print(f"Injected HIL SEE Faults : {len(bundle.fault_schedule)} events over CCSDS Binary Bridge")
    print(
        f"SEC-DED (72,64) Scrubber: {final_rec.cumulative_seu_corrected} SEUs corrected, "
        f"{final_rec.cumulative_mbu_detected} MBUs detected"
    )
    print(
        f"GNC Pointing Error      : Protected TMR = {final_rec.protected_pointing_err_deg:.4f} deg | "
        f"Unprotected COTS = {final_rec.unprotected_pointing_err_deg:.2f} deg"
    )
    print(
        f"Fast SEL Breaker        : Cutoff = {bundle.sel_waveform.cutoff_complete_time_us:.2f} us | "
        f"E_thermal = {bundle.sel_waveform.total_protected_energy_uj:.1f} uJ "
        f"({bundle.sel_waveform.thermal_survival_margin_ratio:.1f}x margin vs {bundle.sel_waveform.damage_threshold_energy_uj:.0f} uJ limit)"
    )
    print("-" * 84)
    for p in bundle.pillars:
        status = "PASS" if p.passed else "FAIL"
        print(f"[{status}] {p.pillar_id:<32} : {p.measured_value} {p.unit} ({p.threshold_description})")
    print("-" * 84)
    print(f"Campaign SHA-256 Digest : {bundle.sha256_digest}")
    print(f"HTML5 Control Room      : {artifact_paths['html']}")
    print(f"Markdown Dossier        : {artifact_paths['markdown']}")
    print(f"JSON Audit Manifest     : {artifact_paths['json']}")
    print("=" * 84)

    return 0 if bundle.all_pillars_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

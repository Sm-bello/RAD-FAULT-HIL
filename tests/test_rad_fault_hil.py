"""Comprehensive Unit & Integration Test Suite for RAD-FAULT-HIL.

Covers orbital radiation transport, South Atlantic Anomaly (SAA) 50x spikes,
graded Al+Ta shielding attenuation, 4-parameter Weibull & 2-parameter Bendel
cross-section integration, Extended Hamming SEC-DED (72,64) coding & scrubbing,
TMR GNC state voting & resynchronization, Fast SEL Current-Limiting Breaker
thermal survival, CCSDS CRC-16-CCITT binary HIL bridge (loopback & UDP socket),
and full 7-pillar TRL 4 campaign qualification.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from rad_fault_hil.fault_tolerant_avionics import (
    FastSelCurrentBreaker,
    HammingSecDed7264,
    MemoryScrubber,
    SecDedStatus,
    TMRGncVoter,
)
from rad_fault_hil.hil_bridge import (
    CcsdsAvionicsTelemetryFrame,
    CcsdsFaultCode,
    CcsdsFaultInjectionCommand,
    FaultInjectionHilBridge,
    HilTransportMode,
    compute_crc16_ccitt,
)
from rad_fault_hil.radiation_env import (
    GradedShieldingConfig,
    ORBIT_PRESETS,
    OrbitRegime,
    SpaceRadiationTransportEngine,
)
from rad_fault_hil.reporting import generate_all_qualification_reports
from rad_fault_hil.see_physics import (
    BendelProtonParameters,
    COTS_COMPONENT_LIBRARY,
    FaultMechanism,
    InjectedFaultEvent,
    SeeCrossSectionEngine,
    WeibullParameters,
)
from rad_fault_hil.verification import run_qualification_campaign


def test_orbit_presets_and_stormer_cutoff_rigidity() -> None:
    """Verify all 5 orbital regimes exist and geomagnetic cutoff decreases from equator to poles/lunar."""
    for regime in (
        OrbitRegime.VLEO,
        OrbitRegime.LEO,
        OrbitRegime.POLAR_SSO,
        OrbitRegime.GTO,
        OrbitRegime.LUNAR,
    ):
        assert regime in ORBIT_PRESETS

    rc_equator = SpaceRadiationTransportEngine.compute_stormer_cutoff_rigidity_gv(0.0, 500.0)
    rc_midlat = SpaceRadiationTransportEngine.compute_stormer_cutoff_rigidity_gv(45.0, 500.0)
    rc_polar = SpaceRadiationTransportEngine.compute_stormer_cutoff_rigidity_gv(85.0, 500.0)
    rc_lunar = SpaceRadiationTransportEngine.compute_stormer_cutoff_rigidity_gv(0.0, 384400.0)

    assert rc_equator > rc_midlat > rc_polar >= 0.05
    assert rc_lunar == pytest.approx(0.05)


def test_south_atlantic_anomaly_50x_flux_spike() -> None:
    """Verify SAA core produces >= 50x trapped proton flux enhancement in LEO."""
    engine = SpaceRadiationTransportEngine(regime=OrbitRegime.LEO)
    in_saa_core, mult_core = engine.compute_saa_enhancement(
        lat_deg=engine.SAA_CENTER_LAT_DEG,
        lon_deg=engine.SAA_CENTER_LON_DEG,
        altitude_km=550.0,
    )
    in_saa_outside, mult_outside = engine.compute_saa_enhancement(
        lat_deg=35.0,
        lon_deg=120.0,
        altitude_km=550.0,
    )
    assert in_saa_core is True
    assert mult_core >= 50.0
    assert in_saa_outside is False
    assert mult_outside == pytest.approx(1.0)


def test_graded_al_ta_shielding_attenuation_monotonicity() -> None:
    """Verify graded Aluminum + Tantalum shielding monotonically reduces transmission and orbital TID."""
    bare = GradedShieldingConfig(aluminum_mm=0.0, tantalum_mm=0.0)
    light = GradedShieldingConfig(aluminum_mm=1.5, tantalum_mm=0.2)
    heavy = GradedShieldingConfig(aluminum_mm=4.0, tantalum_mm=1.0)

    assert bare.areal_density_g_cm2 < light.areal_density_g_cm2 < heavy.areal_density_g_cm2
    assert bare.al_equivalent_mm < light.al_equivalent_mm < heavy.al_equivalent_mm

    att_bare = SpaceRadiationTransportEngine.compute_shielding_attenuation(bare)
    att_light = SpaceRadiationTransportEngine.compute_shielding_attenuation(light)
    att_heavy = SpaceRadiationTransportEngine.compute_shielding_attenuation(heavy)

    assert att_bare["tid_transmission"] > att_light["tid_transmission"] > att_heavy["tid_transmission"]
    assert att_bare["proton_transmission"] > att_light["proton_transmission"] > att_heavy["proton_transmission"]


def test_orbital_radiation_profile_propagation_and_spe() -> None:
    """Verify orbital propagation captures SAA transits, SPE flare window, and positive TID accumulation."""
    engine = SpaceRadiationTransportEngine(
        regime=OrbitRegime.LEO,
        shielding=GradedShieldingConfig(aluminum_mm=2.0, tantalum_mm=0.5),
        include_spe_flare=True,
        spe_start_s=1000.0,
        spe_duration_s=800.0,
    )
    summary = engine.propagate_orbital_radiation_profile(duration_s=3600.0, dt_s=20.0)
    assert len(summary.points) == 180
    assert summary.peak_saa_multiplier_observed >= 50.0
    assert summary.saa_transit_steps > 0
    assert summary.spe_transit_steps > 0
    assert summary.total_tid_krad_si > 0.0
    assert summary.shielding_tid_attenuation_factor > 2.0


def test_weibull_4_parameter_heavy_ion_cross_section() -> None:
    """Verify 4-parameter Weibull cross-section is zero below LET_th and approaches sigma_sat."""
    wb = WeibullParameters(
        let_th_mev_cm2_mg=2.5,
        w_mev_cm2_mg=15.0,
        s_shape=2.0,
        sigma_sat_cm2=5.0e-2,
    )
    assert wb.evaluate(0.5) == 0.0
    assert wb.evaluate(2.5) == 0.0
    mid_val = wb.evaluate(2.5 + 15.0)
    expected_mid = 5.0e-2 * (1.0 - np.exp(-1.0))
    assert mid_val == pytest.approx(expected_mid, rel=1e-6)
    assert wb.evaluate(500.0) == pytest.approx(5.0e-2, rel=1e-6)

    arr = np.array([1.0, 2.5, 17.5, 300.0])
    res = wb.evaluate(arr)
    assert res[0] == 0.0
    assert res[1] == 0.0
    assert res[2] == pytest.approx(expected_mid, rel=1e-6)
    assert res[3] == pytest.approx(5.0e-2, rel=1e-5)


def test_bendel_2_parameter_proton_upset_model() -> None:
    """Verify Bendel 2-parameter proton model is zero below A_p and monotonically increases with proton energy."""
    bp = BendelProtonParameters(a_p_mev=18.0, b_p_mev=24.0, device_bit_multiplier=1.0e6)
    assert bp.evaluate(10.0) == 0.0
    assert bp.evaluate(18.0) == 0.0
    s_30 = bp.evaluate(30.0)
    s_100 = bp.evaluate(100.0)
    s_300 = bp.evaluate(300.0)
    assert 0.0 < s_30 < s_100 < s_300


def test_cots_component_library_rate_integration() -> None:
    """Verify SEE rate integration across all 5 harmonized COTS components."""
    expected_ids = {"STM32H7", "POLARFIRE_SOC", "ZYNQ_ULTRASCALE", "MICRON_DDR4", "SIC_MOSFET"}
    assert set(COTS_COMPONENT_LIBRARY.keys()) == expected_ids

    rad_engine = SpaceRadiationTransportEngine(regime=OrbitRegime.LEO)
    rad_summary = rad_engine.propagate_orbital_radiation_profile(duration_s=1800.0, dt_s=30.0)
    see_engine = SeeCrossSectionEngine()
    summaries = see_engine.summarize_all_components(rad_summary)

    for cid in expected_ids:
        s = summaries[cid]
        assert s.expected_seu_per_day > 0.0
        assert s.peak_saa_seu_rate_s > s.mean_seu_rate_s
        assert s.tid_margin_ratio > 1.0


def test_deterministic_fault_schedule_generation() -> None:
    """Verify Poisson SEE fault schedule is deterministic for identical seeds and contains all 4 fault modes."""
    rad_engine = SpaceRadiationTransportEngine(regime=OrbitRegime.LEO)
    rad_summary = rad_engine.propagate_orbital_radiation_profile(duration_s=3600.0, dt_s=20.0)
    see_engine = SeeCrossSectionEngine(accelerated_beam_scale=180.0)

    sched_a = see_engine.generate_orbital_fault_schedule(rad_engine, rad_summary, seed=42)
    sched_b = see_engine.generate_orbital_fault_schedule(rad_engine, rad_summary, seed=42)
    assert sched_a == sched_b
    mechs = {e.mechanism for e in sched_a}
    assert FaultMechanism.SEU in mechs
    assert FaultMechanism.MBU in mechs
    assert FaultMechanism.SEFI in mechs
    assert FaultMechanism.SEL in mechs


def test_hamming_secded_72_64_all_single_bit_corrections() -> None:
    """Verify Extended Hamming SEC-DED (72,64) corrects 100% of single-bit flips across all 72 bit positions."""
    payload = 0x0123_4567_89AB_CDEF
    cw = HammingSecDed7264.encode(payload)

    clean = HammingSecDed7264.decode(cw)
    assert clean.status == SecDedStatus.NO_ERROR
    assert clean.data_64bit == payload

    for bit_pos in range(72):
        corrupted_cw = cw ^ (1 << bit_pos)
        res = HammingSecDed7264.decode(corrupted_cw)
        assert res.status == SecDedStatus.SINGLE_BIT_CORRECTED
        assert res.corrected_bit_position == bit_pos
        assert res.data_64bit == payload
        assert res.corrected_codeword_72bit == cw


def test_hamming_secded_72_64_double_bit_mbu_detection() -> None:
    """Verify Extended Hamming SEC-DED (72,64) detects 100% of double-bit upsets without false correction."""
    payload = 0xFEDC_BA98_7654_3210
    cw = HammingSecDed7264.encode(payload)

    for b1 in range(72):
        for b2 in range(b1 + 1, min(72, b1 + 12)):
            mbu_cw = cw ^ (1 << b1) ^ (1 << b2)
            res = HammingSecDed7264.decode(mbu_cw)
            assert res.status == SecDedStatus.DOUBLE_BIT_DETECTED
            assert res.overall_parity == 0
            assert res.syndrome != 0


def test_active_memory_scrubber_and_mtbue_amplification() -> None:
    """Verify MemoryScrubber repairs latent 1-bit SEUs, reloads 2-bit MBUs, and amplifies MTBUE > 100x."""
    scrubber = MemoryScrubber(num_words=64, scrub_period_s=5.0)
    scrubber.inject_word_fault(word_index=5, bit_mask_72bit=(1 << 14))
    scrubber.inject_word_fault(word_index=19, bit_mask_72bit=(1 << 3) | (1 << 4))

    report = scrubber.scrub_pass(reload_on_double_error=True)
    assert report.single_bit_errors_corrected == 1
    assert report.double_bit_errors_detected == 1
    assert report.unrecoverable_words_reloaded == 1

    prot_bad, unprot_bad = scrubber.count_corrupted_words()
    assert prot_bad == 0
    assert unprot_bad == 2

    mtbue = MemoryScrubber.compute_mtbue_hours(total_seu_rate_s=0.02, num_words=256, scrub_period_s=1.0)
    assert mtbue["scrubbing_improvement_factor"] > 100.0
    assert mtbue["mtbue_secded_scrubbed_hours"] > mtbue["mtbue_secded_unscrubbed_hours"]


def test_tmr_gnc_voter_single_lane_masking_and_resync() -> None:
    """Verify TMRGncVoter masks single-lane IEEE-754 bit flips and resynchronizes the faulty lane."""
    voter = TMRGncVoter()
    voter.inject_lane_fault(lane_idx=1, state_reg_idx=0, bit_mask=(1 << 62), is_sefi=False)

    vote_res = voter.vote_and_resync()
    assert vote_res.voter_mismatch_detected is True
    assert vote_res.faulty_lane_index == 1
    assert vote_res.resynchronized is True
    assert np.allclose(voter.lanes[0], voter.lanes[1])
    assert np.allclose(voter.lanes[1], voter.lanes[2])


def test_tmr_gnc_sefi_recovery_vs_unprotected_divergence() -> None:
    """Verify protected TMR GNC maintains fine pointing (< 0.25 deg) while unprotected COTS tumbles (> 35 deg)."""
    voter = TMRGncVoter()
    voter.inject_lane_fault(lane_idx=0, state_reg_idx=1, bit_mask=(1 << 60), is_sefi=True)

    for _ in range(30):
        voter.step_dynamics(dt_s=5.0)

    prot_err_deg, unprot_err_deg = voter.get_pointing_errors_deg()
    assert prot_err_deg < 0.25
    assert unprot_err_deg > 35.0


def test_fast_sel_current_breaker_thermal_survival() -> None:
    """Verify FastSelCurrentBreaker cuts off latchup in < 5 us with > 5x thermal energy margin."""
    breaker = FastSelCurrentBreaker(damage_threshold_uj=250.0)
    wf = breaker.simulate_sel_event(saturation_current_a=4.80)

    assert wf.breaker_triggered is True
    assert wf.cutoff_complete_time_us < 5.0
    assert wf.total_protected_energy_uj < 50.0
    assert wf.thermal_survival_margin_ratio > 5.0
    assert wf.unprotected_burned_out is True
    assert wf.total_unprotected_energy_uj > wf.damage_threshold_energy_uj


def test_ccsds_crc16_hil_bridge_loopback_and_udp_socket() -> None:
    """Verify CCSDS binary TC/TM framing, CRC-16-CCITT corruption rejection, and UDP Renode socket transport."""
    ev = InjectedFaultEvent(
        event_index=0,
        time_s=125.0,
        component_id="STM32H7",
        mechanism=FaultMechanism.SEU,
        target_lane=2,
        memory_word_index=10,
        bit_mask=1 << 11,
        let_mev_cm2_mg=9.5,
        in_saa=True,
        spe_active=False,
    )

    for mode in (HilTransportMode.LOOPBACK, HilTransportMode.UDP_RENODE):
        bridge = FaultInjectionHilBridge(
            scrubber=MemoryScrubber(num_words=32),
            tmr_voter=TMRGncVoter(),
            sel_breaker=FastSelCurrentBreaker(),
            mode=mode,
        )
        tm_ok = bridge.inject_event_over_bridge(ev, corrupt_wire_byte=False)
        assert tm_ok is not None
        assert tm_ok.seu_corrected_total == 1
        assert tm_ok.tmr_resync_total == 1

        tm_bad = bridge.inject_event_over_bridge(ev, corrupt_wire_byte=True)
        assert tm_bad is None
        assert bridge.crc_errors_rejected == 1


def test_end_to_end_7_pillar_qualification_and_reporting(tmp_path: Path) -> None:
    """Execute full TRL 4 campaign, verify all 7 pillars pass, and verify HTML/MD/JSON report generation."""
    bundle = run_qualification_campaign(
        regime=OrbitRegime.LEO,
        duration_s=3600.0,
        dt_s=20.0,
        seed=20261105,
    )
    assert bundle.all_pillars_passed is True
    assert len(bundle.pillars) == 7
    assert all(p.passed for p in bundle.pillars)

    paths = generate_all_qualification_reports(bundle, tmp_path)
    assert paths["html"].exists() and paths["html"].stat().st_size > 4000
    assert paths["markdown"].exists() and paths["markdown"].stat().st_size > 1000
    assert paths["json"].exists() and paths["json"].stat().st_size > 1000

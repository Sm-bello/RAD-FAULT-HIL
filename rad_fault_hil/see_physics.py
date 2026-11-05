"""Semiconductor Single-Event Effect (SEE) Cross-Section & Rate Integration Engine.

Implements the 4-parameter Weibull heavy-ion cross-section:
    sigma(LET) = sigma_sat * (1 - exp(-((LET - LET_th) / W) ** s))
and the Bendel 2-parameter proton-induced upset cross-section model across a
harmonized COTS component library (ARM Cortex-M7 STM32H7, RISC-V PolarFire SoC,
Xilinx Zynq UltraScale+ MPSoC, Micron DDR4 SDRAM, and SiC Power MOSFETs).
Computes SEU, MBU, SEFI, and destructive SEL Poisson arrival rates and
time-ordered fault event schedules along an orbital trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from rad_fault_hil.radiation_env import (
    OrbitalRadiationPoint,
    OrbitalRadiationSummary,
    SpaceRadiationTransportEngine,
)


class FaultMechanism(str, Enum):
    """Single-Event Effect (SEE) physical fault modes."""

    SEU = "SEU"    # Single-Event Upset (1-bit memory/register flip)
    MBU = "MBU"    # Multi-Bit Upset (adjacent 2-bit flip in same word)
    SEFI = "SEFI"  # Single-Event Functional Interrupt (CPU/control state hang)
    SEL = "SEL"    # Single-Event Latchup (parasitic PNPN thyristor high-current surge)


@dataclass(frozen=True)
class WeibullParameters:
    """4-parameter Weibull heavy-ion cross-section parameterization.

    Attributes
    ----------
    let_th_mev_cm2_mg : float
        Onset threshold Linear Energy Transfer (MeV*cm^2/mg).
    w_mev_cm2_mg : float
        Scale/width parameter W (MeV*cm^2/mg).
    s_shape : float
        Dimensionless shape exponent s.
    sigma_sat_cm2 : float
        Asymptotic saturation cross-section per device (or total active array) in cm^2.
    """

    let_th_mev_cm2_mg: float
    w_mev_cm2_mg: float
    s_shape: float
    sigma_sat_cm2: float

    def evaluate(self, let_mev_cm2_mg: np.ndarray | float) -> np.ndarray | float:
        """Evaluate 4-parameter Weibull cross-section sigma(LET) in cm^2."""
        if isinstance(let_mev_cm2_mg, (int, float)):
            if let_mev_cm2_mg <= self.let_th_mev_cm2_mg:
                return 0.0
            arg = ((let_mev_cm2_mg - self.let_th_mev_cm2_mg) / self.w_mev_cm2_mg) ** self.s_shape
            return float(self.sigma_sat_cm2 * (1.0 - math.exp(-arg)))

        let_arr = np.asarray(let_mev_cm2_mg, dtype=float)
        sigma = np.zeros_like(let_arr)
        mask = let_arr > self.let_th_mev_cm2_mg
        if np.any(mask):
            arg = ((let_arr[mask] - self.let_th_mev_cm2_mg) / self.w_mev_cm2_mg) ** self.s_shape
            sigma[mask] = self.sigma_sat_cm2 * (1.0 - np.exp(-arg))
        return sigma


@dataclass(frozen=True)
class BendelProtonParameters:
    """Bendel 2-parameter proton-induced upset model (A_p threshold, B_p saturation scale).

    sigma_p(E) = (B_p / A_p)^14 * (1 - exp(-0.18 * Y^0.5))^4 * 1e-12 * device_multiplier [cm^2/device]
    where Y = (18 / A_p)^0.5 * (E - A_p) for E > A_p (MeV).
    """

    a_p_mev: float
    b_p_mev: float = 24.0
    device_bit_multiplier: float = 1.0e6

    def evaluate(self, energy_mev: np.ndarray | float) -> np.ndarray | float:
        """Evaluate proton upset cross-section sigma_p(E) in cm^2/device."""
        prefactor = ((self.b_p_mev / self.a_p_mev) ** 14) * 1.0e-12 * self.device_bit_multiplier
        y_scale = math.sqrt(18.0 / self.a_p_mev)

        if isinstance(energy_mev, (int, float)):
            if energy_mev <= self.a_p_mev:
                return 0.0
            y = y_scale * (energy_mev - self.a_p_mev)
            term = (1.0 - math.exp(-0.18 * math.sqrt(y))) ** 4
            return float(prefactor * term)

        e_arr = np.asarray(energy_mev, dtype=float)
        sigma_p = np.zeros_like(e_arr)
        mask = e_arr > self.a_p_mev
        if np.any(mask):
            y = y_scale * (e_arr[mask] - self.a_p_mev)
            sigma_p[mask] = prefactor * ((1.0 - np.exp(-0.18 * np.sqrt(y))) ** 4)
        return sigma_p


@dataclass(frozen=True)
class CotsComponentRadiationSpec:
    """Complete radiation susceptibility specification for a COTS semiconductor device."""

    component_id: str
    part_number: str
    category: str
    process_node: str
    active_memory_bits: int
    weibull_seu: WeibullParameters
    weibull_mbu: WeibullParameters
    weibull_sefi: WeibullParameters
    weibull_sel: WeibullParameters
    bendel_proton_seu: BendelProtonParameters
    tid_limit_krad_si: float
    sel_holding_current_a: float
    sel_saturation_current_a: float
    sel_damage_energy_uj: float


COTS_COMPONENT_LIBRARY: Dict[str, CotsComponentRadiationSpec] = {
    "STM32H7": CotsComponentRadiationSpec(
        component_id="STM32H7",
        part_number="STM32H743ZIT6 (ARM Cortex-M7)",
        category="Flight Control MCU",
        process_node="40nm Bulk CMOS",
        active_memory_bits=8 * 1024 * 1024,  # 1 MB SRAM
        weibull_seu=WeibullParameters(let_th_mev_cm2_mg=1.2, w_mev_cm2_mg=14.0, s_shape=1.8, sigma_sat_cm2=4.2e-2),
        weibull_mbu=WeibullParameters(let_th_mev_cm2_mg=4.5, w_mev_cm2_mg=22.0, s_shape=2.1, sigma_sat_cm2=2.8e-3),
        weibull_sefi=WeibullParameters(let_th_mev_cm2_mg=6.0, w_mev_cm2_mg=28.0, s_shape=2.2, sigma_sat_cm2=8.5e-4),
        weibull_sel=WeibullParameters(let_th_mev_cm2_mg=12.5, w_mev_cm2_mg=35.0, s_shape=2.5, sigma_sat_cm2=6.0e-4),
        bendel_proton_seu=BendelProtonParameters(a_p_mev=18.0, b_p_mev=24.5, device_bit_multiplier=2.4e3),
        tid_limit_krad_si=22.0,
        sel_holding_current_a=0.45,
        sel_saturation_current_a=4.80,
        sel_damage_energy_uj=250.0,
    ),
    "POLARFIRE_SOC": CotsComponentRadiationSpec(
        component_id="POLARFIRE_SOC",
        part_number="MPFS250T (RISC-V PolarFire SoC)",
        category="Radiation-Tolerant COTS SoC FPGA",
        process_node="28nm SONOS Non-Volatile CMOS",
        active_memory_bits=16 * 1024 * 1024,  # 2 MB L2 Cache + Embedded SRAM
        weibull_seu=WeibullParameters(let_th_mev_cm2_mg=1.8, w_mev_cm2_mg=18.0, s_shape=1.9, sigma_sat_cm2=2.5e-2),
        weibull_mbu=WeibullParameters(let_th_mev_cm2_mg=6.5, w_mev_cm2_mg=26.0, s_shape=2.3, sigma_sat_cm2=1.1e-3),
        weibull_sefi=WeibullParameters(let_th_mev_cm2_mg=9.5, w_mev_cm2_mg=32.0, s_shape=2.4, sigma_sat_cm2=3.2e-4),
        weibull_sel=WeibullParameters(let_th_mev_cm2_mg=68.0, w_mev_cm2_mg=45.0, s_shape=3.0, sigma_sat_cm2=1.0e-6),
        bendel_proton_seu=BendelProtonParameters(a_p_mev=22.0, b_p_mev=23.5, device_bit_multiplier=1.8e3),
        tid_limit_krad_si=65.0,
        sel_holding_current_a=0.60,
        sel_saturation_current_a=3.20,
        sel_damage_energy_uj=380.0,
    ),
    "ZYNQ_ULTRASCALE": CotsComponentRadiationSpec(
        component_id="ZYNQ_ULTRASCALE",
        part_number="XCZU9EG (Xilinx Zynq UltraScale+ MPSoC)",
        category="High-Performance Payload MPSoC",
        process_node="16nm FinFET",
        active_memory_bits=32 * 1024 * 1024,
        weibull_seu=WeibullParameters(let_th_mev_cm2_mg=0.65, w_mev_cm2_mg=11.0, s_shape=1.5, sigma_sat_cm2=6.8e-2),
        weibull_mbu=WeibullParameters(let_th_mev_cm2_mg=2.8, w_mev_cm2_mg=19.0, s_shape=1.9, sigma_sat_cm2=5.4e-3),
        weibull_sefi=WeibullParameters(let_th_mev_cm2_mg=4.2, w_mev_cm2_mg=24.0, s_shape=2.0, sigma_sat_cm2=1.6e-3),
        weibull_sel=WeibullParameters(let_th_mev_cm2_mg=16.0, w_mev_cm2_mg=38.0, s_shape=2.6, sigma_sat_cm2=4.2e-4),
        bendel_proton_seu=BendelProtonParameters(a_p_mev=14.0, b_p_mev=25.0, device_bit_multiplier=4.5e3),
        tid_limit_krad_si=40.0,
        sel_holding_current_a=0.80,
        sel_saturation_current_a=5.60,
        sel_damage_energy_uj=290.0,
    ),
    "MICRON_DDR4": CotsComponentRadiationSpec(
        component_id="MICRON_DDR4",
        part_number="MT40A1G8 (Micron 8Gb DDR4 SDRAM)",
        category="Bulk Volatile Memory Array",
        process_node="20nm DRAM",
        active_memory_bits=64 * 1024 * 1024,
        weibull_seu=WeibullParameters(let_th_mev_cm2_mg=0.90, w_mev_cm2_mg=12.5, s_shape=1.7, sigma_sat_cm2=9.5e-2),
        weibull_mbu=WeibullParameters(let_th_mev_cm2_mg=3.2, w_mev_cm2_mg=17.5, s_shape=2.0, sigma_sat_cm2=8.2e-3),
        weibull_sefi=WeibullParameters(let_th_mev_cm2_mg=5.0, w_mev_cm2_mg=21.0, s_shape=2.2, sigma_sat_cm2=2.1e-3),
        weibull_sel=WeibullParameters(let_th_mev_cm2_mg=18.0, w_mev_cm2_mg=40.0, s_shape=2.7, sigma_sat_cm2=3.5e-4),
        bendel_proton_seu=BendelProtonParameters(a_p_mev=15.5, b_p_mev=24.8, device_bit_multiplier=6.0e3),
        tid_limit_krad_si=30.0,
        sel_holding_current_a=0.50,
        sel_saturation_current_a=4.20,
        sel_damage_energy_uj=220.0,
    ),
    "SIC_MOSFET": CotsComponentRadiationSpec(
        component_id="SIC_MOSFET",
        part_number="C3M0065090D (Wolfspeed 900V SiC MOSFET)",
        category="EPS Power Switching Stage",
        process_node="Wide-Bandgap 4H-SiC",
        active_memory_bits=1,
        weibull_seu=WeibullParameters(let_th_mev_cm2_mg=8.0, w_mev_cm2_mg=30.0, s_shape=2.4, sigma_sat_cm2=1.2e-4),
        weibull_mbu=WeibullParameters(let_th_mev_cm2_mg=15.0, w_mev_cm2_mg=35.0, s_shape=2.6, sigma_sat_cm2=2.0e-5),
        weibull_sefi=WeibullParameters(let_th_mev_cm2_mg=12.0, w_mev_cm2_mg=32.0, s_shape=2.5, sigma_sat_cm2=8.0e-5),
        weibull_sel=WeibullParameters(let_th_mev_cm2_mg=10.5, w_mev_cm2_mg=28.0, s_shape=2.3, sigma_sat_cm2=9.5e-4),
        bendel_proton_seu=BendelProtonParameters(a_p_mev=35.0, b_p_mev=22.0, device_bit_multiplier=1.0e1),
        tid_limit_krad_si=150.0,
        sel_holding_current_a=1.10,
        sel_saturation_current_a=6.50,
        sel_damage_energy_uj=310.0,
    ),
}


@dataclass
class ComponentSeeRateSummary:
    """Integrated SEE arrival rates and trajectory fault expectations for a COTS component."""

    component_id: str
    part_number: str
    mean_seu_rate_s: float
    peak_saa_seu_rate_s: float
    mean_mbu_rate_s: float
    mean_sefi_rate_s: float
    mean_sel_rate_s: float
    expected_seu_per_day: float
    expected_mbu_per_day: float
    expected_sefi_per_day: float
    expected_sel_per_day: float
    tid_margin_ratio: float


@dataclass(frozen=True)
class InjectedFaultEvent:
    """Discrete radiation fault event sampled along the trajectory for HIL injection."""

    event_index: int
    time_s: float
    component_id: str
    mechanism: FaultMechanism
    target_lane: int
    memory_word_index: int
    bit_mask: int
    let_mev_cm2_mg: float
    in_saa: bool
    spe_active: bool


class SeeCrossSectionEngine:
    """Integrates heavy-ion Weibull and proton Bendel cross-sections over orbital radiation spectra."""

    def __init__(
        self,
        component_library: Optional[Dict[str, CotsComponentRadiationSpec]] = None,
        accelerated_beam_scale: float = 180.0,
    ) -> None:
        self.components = component_library or COTS_COMPONENT_LIBRARY
        # Accelerated cyclotron / SAA storm scaling factor used during HIL fault-injection qualification
        # so multi-day orbital fluences are compressed into a deterministic 90-minute HIL test campaign.
        self.accelerated_beam_scale = accelerated_beam_scale

    @staticmethod
    def integrate_weibull_rate(
        let_grid: np.ndarray,
        diff_let_spectrum: np.ndarray,
        weibull: WeibullParameters,
    ) -> float:
        """Integrate heavy-ion differential LET spectrum against 4-parameter Weibull sigma(LET)."""
        sigma_let = weibull.evaluate(let_grid)
        # Solid angle 4*pi sr factor included in omnidirectional conversion
        integrand = diff_let_spectrum * sigma_let * (4.0 * math.pi)
        return float(np.trapezoid(integrand, let_grid))

    @staticmethod
    def integrate_bendel_proton_rate(
        energy_grid_mev: np.ndarray,
        diff_proton_spectrum: np.ndarray,
        bendel: BendelProtonParameters,
    ) -> float:
        """Integrate differential proton energy spectrum against 2-parameter Bendel sigma_p(E)."""
        sigma_p = bendel.evaluate(energy_grid_mev)
        integrand = diff_proton_spectrum * sigma_p
        return float(np.trapezoid(integrand, energy_grid_mev))

    def evaluate_component_instantaneous_rates(
        self,
        spec: CotsComponentRadiationSpec,
        let_grid: np.ndarray,
        diff_let_spectrum: np.ndarray,
        proton_energy_grid: np.ndarray,
        diff_proton_spectrum: np.ndarray,
    ) -> Dict[FaultMechanism, float]:
        """Compute instantaneous Poisson arrival rates [events/s] for SEU, MBU, SEFI, and SEL."""
        hi_seu = self.integrate_weibull_rate(let_grid, diff_let_spectrum, spec.weibull_seu)
        p_seu = self.integrate_bendel_proton_rate(proton_energy_grid, diff_proton_spectrum, spec.bendel_proton_seu)
        total_seu = hi_seu + p_seu

        hi_mbu = self.integrate_weibull_rate(let_grid, diff_let_spectrum, spec.weibull_mbu)
        total_mbu = hi_mbu + 0.045 * p_seu

        hi_sefi = self.integrate_weibull_rate(let_grid, diff_let_spectrum, spec.weibull_sefi)
        total_sefi = hi_sefi + 0.012 * p_seu

        hi_sel = self.integrate_weibull_rate(let_grid, diff_let_spectrum, spec.weibull_sel)
        total_sel = hi_sel + 0.004 * p_seu

        return {
            FaultMechanism.SEU: total_seu,
            FaultMechanism.MBU: total_mbu,
            FaultMechanism.SEFI: total_sefi,
            FaultMechanism.SEL: total_sel,
        }

    def summarize_all_components(
        self,
        rad_summary: OrbitalRadiationSummary,
    ) -> Dict[str, ComponentSeeRateSummary]:
        """Compute trajectory-averaged and peak SAA SEE rates across all COTS library components."""
        results: Dict[str, ComponentSeeRateSummary] = {}
        sec_per_day = 86400.0

        # Annualized TID estimate from trajectory duration
        orbits_per_year = (365.25 * 86400.0) / max(1.0, rad_summary.duration_s)
        annual_tid_krad = rad_summary.total_tid_krad_si * orbits_per_year

        for comp_id, spec in self.components.items():
            mean_rates = self.evaluate_component_instantaneous_rates(
                spec=spec,
                let_grid=rad_summary.let_grid_mev_cm2_mg,
                diff_let_spectrum=rad_summary.mean_diff_let_spectrum,
                proton_energy_grid=rad_summary.proton_energy_grid_mev,
                diff_proton_spectrum=rad_summary.mean_diff_proton_spectrum,
            )
            peak_rates = self.evaluate_component_instantaneous_rates(
                spec=spec,
                let_grid=rad_summary.let_grid_mev_cm2_mg,
                diff_let_spectrum=rad_summary.peak_saa_diff_let_spectrum,
                proton_energy_grid=rad_summary.proton_energy_grid_mev,
                diff_proton_spectrum=rad_summary.peak_saa_diff_proton_spectrum,
            )

            tid_margin = spec.tid_limit_krad_si / max(1e-6, annual_tid_krad)
            results[comp_id] = ComponentSeeRateSummary(
                component_id=comp_id,
                part_number=spec.part_number,
                mean_seu_rate_s=mean_rates[FaultMechanism.SEU],
                peak_saa_seu_rate_s=peak_rates[FaultMechanism.SEU],
                mean_mbu_rate_s=mean_rates[FaultMechanism.MBU],
                mean_sefi_rate_s=mean_rates[FaultMechanism.SEFI],
                mean_sel_rate_s=mean_rates[FaultMechanism.SEL],
                expected_seu_per_day=mean_rates[FaultMechanism.SEU] * sec_per_day,
                expected_mbu_per_day=mean_rates[FaultMechanism.MBU] * sec_per_day,
                expected_sefi_per_day=mean_rates[FaultMechanism.SEFI] * sec_per_day,
                expected_sel_per_day=mean_rates[FaultMechanism.SEL] * sec_per_day,
                tid_margin_ratio=tid_margin,
            )
        return results

    def generate_orbital_fault_schedule(
        self,
        rad_engine: SpaceRadiationTransportEngine,
        rad_summary: OrbitalRadiationSummary,
        target_component_id: str = "STM32H7",
        memory_words: int = 256,
        seed: int = 20261105,
    ) -> List[InjectedFaultEvent]:
        """Generate a deterministic, Poisson-sampled SEE fault timeline along the orbital trajectory.

        Uses `accelerated_beam_scale` to represent a qualification stress campaign through
        the South Atlantic Anomaly (SAA) and Solar Particle Event (SPE) windows.
        """
        rng = np.random.default_rng(seed)
        spec = self.components[target_component_id]
        events: List[InjectedFaultEvent] = []
        event_idx = 0
        dt = rad_summary.dt_s

        for pt in rad_summary.points:
            diff_let = rad_engine.compute_heinrich_gcr_differential_let_spectrum(
                cutoff_rigidity_gv=pt.cutoff_rigidity_gv,
                saa_factor=pt.saa_intensity_factor,
                spe_flux_cm2_s=pt.spe_proton_flux_cm2_s,
            )
            diff_p = rad_engine.compute_differential_proton_spectrum(
                trapped_proton_flux_cm2_s=pt.trapped_proton_flux_cm2_s,
                spe_proton_flux_cm2_s=pt.spe_proton_flux_cm2_s,
            )
            rates = self.evaluate_component_instantaneous_rates(
                spec=spec,
                let_grid=rad_summary.let_grid_mev_cm2_mg,
                diff_let_spectrum=diff_let,
                proton_energy_grid=rad_summary.proton_energy_grid_mev,
                diff_proton_spectrum=diff_p,
            )

            for mech in (FaultMechanism.SEU, FaultMechanism.MBU, FaultMechanism.SEFI, FaultMechanism.SEL):
                lam = rates[mech] * dt * self.accelerated_beam_scale
                count = int(rng.poisson(lam))
                for _ in range(count):
                    sub_t = pt.time_s + float(rng.uniform(0.0, dt))
                    lane = int(rng.integers(0, 3))
                    word_idx = int(rng.integers(0, memory_words))
                    if mech == FaultMechanism.SEU:
                        bit_pos = int(rng.integers(0, 64))
                        bit_mask = 1 << bit_pos
                        sampled_let = float(rng.uniform(spec.weibull_seu.let_th_mev_cm2_mg, 25.0))
                    elif mech == FaultMechanism.MBU:
                        bit_pos = int(rng.integers(0, 63))
                        bit_mask = (1 << bit_pos) | (1 << (bit_pos + 1))
                        sampled_let = float(rng.uniform(spec.weibull_mbu.let_th_mev_cm2_mg, 42.0))
                    elif mech == FaultMechanism.SEFI:
                        # Flip high-order IEEE-754 exponent bit in GNC state register
                        bit_pos = int(rng.integers(52, 62))
                        bit_mask = 1 << bit_pos
                        sampled_let = float(rng.uniform(spec.weibull_sefi.let_th_mev_cm2_mg, 55.0))
                    else:
                        bit_mask = 0xFFFFFFFFFFFFFFFF
                        sampled_let = float(rng.uniform(spec.weibull_sel.let_th_mev_cm2_mg, 75.0))

                    events.append(
                        InjectedFaultEvent(
                            event_index=event_idx,
                            time_s=round(sub_t, 4),
                            component_id=target_component_id,
                            mechanism=mech,
                            target_lane=lane,
                            memory_word_index=word_idx,
                            bit_mask=bit_mask,
                            let_mev_cm2_mg=round(sampled_let, 3),
                            in_saa=pt.in_saa,
                            spe_active=pt.spe_active,
                        )
                    )
                    event_idx += 1

        # Guarantee at least one representative event of each critical class during SAA/SPE
        # if stochastic realization happens to have zero for rare MBU/SEFI/SEL
        present_mechs = {e.mechanism for e in events}
        saa_points = [p for p in rad_summary.points if p.in_saa] or rad_summary.points
        anchor_pt = max(saa_points, key=lambda p: p.saa_intensity_factor)

        fallback_specs = [
            (FaultMechanism.SEU, 0.0, 0, 12, 1 << 19, 8.4),
            (FaultMechanism.MBU, 2.0, 1, 44, (1 << 7) | (1 << 8), 18.2),
            (FaultMechanism.SEFI, 4.0, 2, 0, 1 << 58, 24.6),
            (FaultMechanism.SEL, 6.0, 1, 0, 0xFFFFFFFFFFFFFFFF, 38.5),
        ]
        for mech, offset_s, lane, word_idx, mask, let_val in fallback_specs:
            if mech not in present_mechs:
                events.append(
                    InjectedFaultEvent(
                        event_index=event_idx,
                        time_s=round(anchor_pt.time_s + offset_s, 4),
                        component_id=target_component_id,
                        mechanism=mech,
                        target_lane=lane,
                        memory_word_index=word_idx,
                        bit_mask=mask,
                        let_mev_cm2_mg=let_val,
                        in_saa=anchor_pt.in_saa,
                        spe_active=anchor_pt.spe_active,
                    )
                )
                event_idx += 1

        events.sort(key=lambda e: e.time_s)
        # Re-index sequentially after sorting
        return [
            InjectedFaultEvent(
                event_index=i,
                time_s=e.time_s,
                component_id=e.component_id,
                mechanism=e.mechanism,
                target_lane=e.target_lane,
                memory_word_index=e.memory_word_index,
                bit_mask=e.bit_mask,
                let_mev_cm2_mg=e.let_mev_cm2_mg,
                in_saa=e.in_saa,
                spe_active=e.spe_active,
            )
            for i, e in enumerate(events)
        ]

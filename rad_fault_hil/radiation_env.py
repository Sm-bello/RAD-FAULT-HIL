"""Orbital Space Radiation Environment & Graded Shielding Transport Engine.

Models trapped proton/electron flux (including South Atlantic Anomaly 50x flux
spikes), Galactic Cosmic Ray (GCR) Heinrich Linear Energy Transfer (LET)
spectra, and Solar Particle Event (SPE / Carrington-class flare) proton
fluences across LEO, VLEO, Polar/SSO, GTO, and Lunar orbits, with graded
Aluminum + Tantalum (Al+Ta) Z-shielding stopping-power attenuation and Total
Ionizing Dose (TID in krad(Si)) accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class OrbitRegime(str, Enum):
    """Standard orbital regimes for COTS-to-Orbit radiation qualification."""

    VLEO = "VLEO"
    LEO = "LEO"
    POLAR_SSO = "POLAR_SSO"
    GTO = "GTO"
    LUNAR = "LUNAR"


@dataclass(frozen=True)
class OrbitParameters:
    """Orbital elements and geomagnetic exposure characteristics."""

    regime: OrbitRegime
    altitude_km: float
    apogee_km: float
    inclination_deg: float
    period_min: float
    base_trapped_proton_flux_cm2_s: float
    base_trapped_electron_flux_cm2_s: float
    saa_peak_multiplier: float = 55.0
    description: str = ""


ORBIT_PRESETS: Dict[OrbitRegime, OrbitParameters] = {
    OrbitRegime.VLEO: OrbitParameters(
        regime=OrbitRegime.VLEO,
        altitude_km=320.0,
        apogee_km=320.0,
        inclination_deg=45.0,
        period_min=90.9,
        base_trapped_proton_flux_cm2_s=85.0,
        base_trapped_electron_flux_cm2_s=1.2e4,
        saa_peak_multiplier=52.0,
        description="Very Low Earth Orbit (320 km, 45 deg) - strong geomagnetic shielding except SAA core",
    ),
    OrbitRegime.LEO: OrbitParameters(
        regime=OrbitRegime.LEO,
        altitude_km=550.0,
        apogee_km=550.0,
        inclination_deg=51.6,
        period_min=95.6,
        base_trapped_proton_flux_cm2_s=420.0,
        base_trapped_electron_flux_cm2_s=8.5e4,
        saa_peak_multiplier=58.0,
        description="Standard LEO / ISS-like (550 km, 51.6 deg) - regular South Atlantic Anomaly transits",
    ),
    OrbitRegime.POLAR_SSO: OrbitParameters(
        regime=OrbitRegime.POLAR_SSO,
        altitude_km=700.0,
        apogee_km=700.0,
        inclination_deg=98.2,
        period_min=98.8,
        base_trapped_proton_flux_cm2_s=680.0,
        base_trapped_electron_flux_cm2_s=2.4e5,
        saa_peak_multiplier=60.0,
        description="Sun-Synchronous Polar Orbit (700 km, 98.2 deg) - SAA + polar horn SPE/GCR exposure",
    ),
    OrbitRegime.GTO: OrbitParameters(
        regime=OrbitRegime.GTO,
        altitude_km=250.0,
        apogee_km=35786.0,
        inclination_deg=7.0,
        period_min=635.0,
        base_trapped_proton_flux_cm2_s=4800.0,
        base_trapped_electron_flux_cm2_s=3.5e6,
        saa_peak_multiplier=50.0,
        description="Geostationary Transfer Orbit (250x35786 km) - dual Van Allen belt traversal",
    ),
    OrbitRegime.LUNAR: OrbitParameters(
        regime=OrbitRegime.LUNAR,
        altitude_km=384400.0,
        apogee_km=384400.0,
        inclination_deg=28.5,
        period_min=1440.0,
        base_trapped_proton_flux_cm2_s=15.0,
        base_trapped_electron_flux_cm2_s=5.0e2,
        saa_peak_multiplier=1.0,
        description="Cis-Lunar / Deep Space - unshielded GCR heavy ions and direct Solar Particle Events",
    ),
}


@dataclass(frozen=True)
class GradedShieldingConfig:
    """Graded Aluminum (low-Z) + Tantalum (high-Z) avionics enclosure shielding."""

    aluminum_mm: float = 2.0
    tantalum_mm: float = 0.5

    @property
    def areal_density_g_cm2(self) -> float:
        """Total physical mass thickness in g/cm^2 (Al=2.70, Ta=16.65 g/cm^3)."""
        al_g_cm2 = (self.aluminum_mm / 10.0) * 2.70
        ta_g_cm2 = (self.tantalum_mm / 10.0) * 16.65
        return al_g_cm2 + ta_g_cm2

    @property
    def al_equivalent_mm(self) -> float:
        """Aluminum-equivalent thickness in mm including graded-Z stopping bonus."""
        # Tantalum provides high-Z electron/Bremsstrahlung & proton stopping power
        # when backed by Aluminum to absorb secondary low-energy X-rays.
        graded_synergy = 1.15 if (self.aluminum_mm > 0.5 and self.tantalum_mm > 0.1) else 1.0
        return self.aluminum_mm + (self.tantalum_mm * (16.65 / 2.70) * 0.82 * graded_synergy)


@dataclass
class OrbitalRadiationPoint:
    """Instantaneous radiation environment at a single orbital timestep."""

    time_s: float
    lat_deg: float
    lon_deg: float
    altitude_km: float
    in_saa: bool
    saa_intensity_factor: float
    spe_active: bool
    spe_proton_flux_cm2_s: float
    cutoff_rigidity_gv: float
    trapped_proton_flux_cm2_s: float
    trapped_electron_flux_cm2_s: float
    total_proton_flux_cm2_s: float
    gcr_integral_flux_cm2_s: float
    tid_rate_krad_si_s: float
    cumulative_tid_krad_si: float


@dataclass
class OrbitalRadiationSummary:
    """Trajectory-level summary of the orbital radiation transport simulation."""

    regime: OrbitRegime
    shielding: GradedShieldingConfig
    duration_s: float
    dt_s: float
    points: List[OrbitalRadiationPoint]
    let_grid_mev_cm2_mg: np.ndarray
    mean_diff_let_spectrum: np.ndarray
    peak_saa_diff_let_spectrum: np.ndarray
    proton_energy_grid_mev: np.ndarray
    mean_diff_proton_spectrum: np.ndarray
    peak_saa_diff_proton_spectrum: np.ndarray
    peak_saa_multiplier_observed: float
    saa_transit_steps: int
    spe_transit_steps: int
    total_tid_krad_si: float
    shielding_tid_attenuation_factor: float


class SpaceRadiationTransportEngine:
    """Physics-based orbital radiation environment and graded shielding transport engine."""

    # Standard SAA centroid in South America / South Atlantic
    SAA_CENTER_LAT_DEG: float = -26.0
    SAA_CENTER_LON_DEG: float = -45.0
    SAA_SIGMA_LAT_DEG: float = 11.5
    SAA_SIGMA_LON_DEG: float = 22.0

    def __init__(
        self,
        regime: OrbitRegime = OrbitRegime.LEO,
        shielding: Optional[GradedShieldingConfig] = None,
        solar_max: bool = False,
        include_spe_flare: bool = True,
        spe_start_s: float = 1800.0,
        spe_duration_s: float = 1200.0,
        spe_peak_proton_flux_cm2_s: float = 3.5e4,
    ) -> None:
        self.regime = regime
        self.orbit = ORBIT_PRESETS[regime]
        self.shielding = shielding or GradedShieldingConfig(aluminum_mm=2.0, tantalum_mm=0.5)
        self.solar_max = solar_max
        self.include_spe_flare = include_spe_flare
        self.spe_start_s = spe_start_s
        self.spe_duration_s = spe_duration_s
        self.spe_peak_proton_flux_cm2_s = spe_peak_proton_flux_cm2_s

        # Logarithmic LET grid from 0.1 to 100 MeV*cm^2/mg (80 bins)
        self.let_grid = np.logspace(-1.0, 2.0, 80)
        # Proton kinetic energy grid from 5 to 500 MeV (60 bins)
        self.proton_energy_grid = np.logspace(np.log10(5.0), np.log10(500.0), 60)

    @staticmethod
    def compute_stormer_cutoff_rigidity_gv(lat_deg: float, altitude_km: float) -> float:
        """Compute geomagnetic Størmer vertical cutoff rigidity R_c in GV."""
        if altitude_km > 60000.0:
            return 0.05  # Deep space / cis-lunar has negligible magnetospheric cutoff
        r_earth_km = 6371.0
        l_shell_r = (r_earth_km + altitude_km) / r_earth_km
        # Offset dipole magnetic latitude approximation
        mag_lat_rad = math.radians(lat_deg)
        cos_lat = math.cos(mag_lat_rad)
        rc_gv = 14.5 * (cos_lat ** 4) / (l_shell_r ** 2)
        return max(0.05, rc_gv)

    def compute_saa_enhancement(self, lat_deg: float, lon_deg: float, altitude_km: float) -> Tuple[bool, float]:
        """Compute South Atlantic Anomaly (SAA) trapped proton flux multiplier (>= 50x at core)."""
        if self.regime == OrbitRegime.LUNAR or altitude_km > 3000.0:
            return False, 1.0

        dlat = (lat_deg - self.SAA_CENTER_LAT_DEG) / self.SAA_SIGMA_LAT_DEG
        dlon = (lon_deg - self.SAA_CENTER_LON_DEG) / self.SAA_SIGMA_LON_DEG
        r2 = dlat * dlat + dlon * dlon

        # SAA boundary defined within 2.1-sigma ellipse
        in_saa = r2 <= (2.1 ** 2)
        if not in_saa:
            return False, 1.0

        # Peak multiplier reaches orbit.saa_peak_multiplier (>= 50x) at centroid
        multiplier = 1.0 + (self.orbit.saa_peak_multiplier - 1.0) * math.exp(-0.5 * r2)
        return True, multiplier

    @staticmethod
    def compute_shielding_attenuation(shielding: GradedShieldingConfig) -> Dict[str, float]:
        """Compute stopping-power transmission factors for protons, electrons, GCR, and TID."""
        x_eq_mm = shielding.al_equivalent_mm
        areal_g_cm2 = shielding.areal_density_g_cm2

        # Low-energy trapped protons & SPE protons are strongly attenuated by graded Al+Ta
        proton_transmission = math.exp(-0.32 * x_eq_mm) + 0.08 * math.exp(-0.045 * x_eq_mm)
        proton_transmission = min(1.0, max(0.02, proton_transmission))

        # Trapped electrons and Bremsstrahlung suppressed by high-Z Tantalum layer
        electron_transmission = math.exp(-0.85 * x_eq_mm) + 0.015 * math.exp(-0.12 * x_eq_mm)
        electron_transmission = min(1.0, max(0.002, electron_transmission))

        # High-energy GCR heavy ions penetrate deeper, moderate LET transformation
        gcr_transmission = math.exp(-0.065 * areal_g_cm2)
        gcr_transmission = min(1.0, max(0.15, gcr_transmission))

        # Overall TID attenuation factor relative to bare 0.2mm package
        tid_transmission = 0.65 * electron_transmission + 0.35 * proton_transmission
        return {
            "proton_transmission": proton_transmission,
            "electron_transmission": electron_transmission,
            "gcr_transmission": gcr_transmission,
            "tid_transmission": tid_transmission,
        }

    def compute_heinrich_gcr_differential_let_spectrum(
        self,
        cutoff_rigidity_gv: float,
        saa_factor: float = 1.0,
        spe_flux_cm2_s: float = 0.0,
    ) -> np.ndarray:
        """Compute attenuated differential LET spectrum dPhi/dLET [particles/(cm^2*s*(MeV*cm^2/mg))].

        Follows the Heinrich / CREME96 power-law + iron-peak knee at ~28 MeV*cm^2/mg
        plus secondary nuclear recoil LET tail from trapped SAA & SPE protons.
        """
        att = self.compute_shielding_attenuation(self.shielding)
        solar_mod = 0.55 if self.solar_max else 1.0
        # Geomagnetic transmission increases as cutoff rigidity decreases
        geomag_trans = math.exp(-0.18 * cutoff_rigidity_gv)

        let = self.let_grid
        # Baseline GCR heavy-ion differential flux (H through Fe peak + ultra-heavy tail)
        iron_knee = 1.0 / (1.0 + (let / 28.0) ** 4.2)
        ultra_heavy_tail = 1.5e-4 * (let / 28.0) ** (-2.8)
        gcr_diff = (
            2.4e-2
            * solar_mod
            * geomag_trans
            * att["gcr_transmission"]
            * (let ** -2.15)
            * (iron_knee + ultra_heavy_tail)
        )

        # Secondary silicon nuclear spallation recoils from SAA + SPE high-energy protons
        # Protons produce recoil nuclei with LET up to ~15 MeV*cm^2/mg in Silicon
        proton_recoil_scale = ((saa_factor - 1.0) * 1.8e-3 + (spe_flux_cm2_s / 1.0e4) * 4.5e-3) * att["proton_transmission"]
        proton_recoil_diff = proton_recoil_scale * (let ** -1.85) * np.exp(-let / 14.5)

        return gcr_diff + proton_recoil_diff

    def compute_differential_proton_spectrum(
        self,
        trapped_proton_flux_cm2_s: float,
        spe_proton_flux_cm2_s: float,
    ) -> np.ndarray:
        """Compute differential proton energy spectrum dPhi_p/dE [protons/(cm^2*s*MeV)] behind shielding."""
        e_mev = self.proton_energy_grid
        x_eq_mm = self.shielding.al_equivalent_mm
        # Minimum cutoff energy to penetrate x_eq_mm of Al (range-energy relation E_min ~ 13.5 * x_mm^0.58)
        e_cutoff_mev = 13.5 * (max(0.05, x_eq_mm) ** 0.58)

        # Trapped proton spectrum (harder in SAA / inner belt, E0 ~ 85 MeV)
        trapped_shape = np.exp(-e_mev / 85.0) / 85.0
        # Solar Particle Event (Carrington/King flare spectrum, E0 ~ 45 MeV)
        spe_shape = np.exp(-e_mev / 45.0) / 45.0

        # Stopping-power attenuation below cutoff energy
        stopping_window = 1.0 / (1.0 + np.exp(-(e_mev - e_cutoff_mev) / 4.0))

        diff_flux = (trapped_proton_flux_cm2_s * trapped_shape + spe_proton_flux_cm2_s * spe_shape) * stopping_window
        return diff_flux

    def propagate_orbital_radiation_profile(
        self,
        duration_s: float = 5400.0,
        dt_s: float = 10.0,
    ) -> OrbitalRadiationSummary:
        """Propagate spacecraft trajectory and compute time-resolved radiation environment.

        Parameters
        ----------
        duration_s : float
            Total simulation duration in seconds (default 5400s = 90 min orbit).
        dt_s : float
            Simulation timestep in seconds.

        Returns
        -------
        OrbitalRadiationSummary
            Full time-series radiation profile, spectra, and shielding metrics.
        """
        n_steps = max(2, int(round(duration_s / dt_s)))
        att = self.compute_shielding_attenuation(self.shielding)
        unshielded_att = self.compute_shielding_attenuation(GradedShieldingConfig(0.0, 0.0))

        points: List[OrbitalRadiationPoint] = []
        cumulative_tid = 0.0
        peak_saa_mult = 1.0
        saa_steps = 0
        spe_steps = 0

        omega_orb = 2.0 * math.pi / (self.orbit.period_min * 60.0)
        earth_rot_deg_s = 360.0 / 86164.0

        # Phase-align the trajectory so the spacecraft passes directly through the SAA core
        # at t_saa_target = 0.25 * duration_s for non-Lunar orbits
        t_saa_target = 0.25 * duration_s
        inc_rad = math.radians(max(self.orbit.inclination_deg, 28.0))
        target_sin_u = max(-0.98, min(0.98, math.sin(math.radians(self.SAA_CENTER_LAT_DEG)) / math.sin(inc_rad)))
        u_target = math.asin(target_sin_u)
        u0 = u_target - omega_orb * t_saa_target

        u_at_target = u0 + omega_orb * t_saa_target
        lon_orb_at_target = math.degrees(math.atan2(math.cos(inc_rad) * math.sin(u_at_target), math.cos(u_at_target)))
        raan0_deg = self.SAA_CENTER_LON_DEG - lon_orb_at_target + earth_rot_deg_s * t_saa_target

        accum_let_spec = np.zeros_like(self.let_grid)
        peak_saa_let_spec = np.zeros_like(self.let_grid)
        accum_p_spec = np.zeros_like(self.proton_energy_grid)
        peak_saa_p_spec = np.zeros_like(self.proton_energy_grid)

        for step in range(n_steps):
            t = step * dt_s
            u = u0 + omega_orb * t
            lat_deg = math.degrees(math.asin(math.sin(inc_rad) * math.sin(u)))
            lon_orb = math.degrees(math.atan2(math.cos(inc_rad) * math.sin(u), math.cos(u)))
            lon_deg = ((raan0_deg + lon_orb - earth_rot_deg_s * t + 180.0) % 360.0) - 180.0

            # Altitude variation for eccentric orbits (e.g., GTO)
            if self.orbit.apogee_km > self.orbit.altitude_km * 1.1:
                alt_km = 0.5 * (self.orbit.altitude_km + self.orbit.apogee_km) - 0.5 * (
                    self.orbit.apogee_km - self.orbit.altitude_km
                ) * math.cos(u)
            else:
                alt_km = self.orbit.altitude_km

            in_saa, saa_mult = self.compute_saa_enhancement(lat_deg, lon_deg, alt_km)
            if in_saa:
                saa_steps += 1
            if saa_mult > peak_saa_mult:
                peak_saa_mult = saa_mult

            cutoff_gv = self.compute_stormer_cutoff_rigidity_gv(lat_deg, alt_km)

            # Check Solar Particle Event (SPE) flare window
            spe_active = bool(
                self.include_spe_flare and (self.spe_start_s <= t <= (self.spe_start_s + self.spe_duration_s))
            )
            if spe_active:
                spe_steps += 1
                tau = (t - self.spe_start_s) / max(1.0, self.spe_duration_s)
                # Fast-rise exponential decay flare pulse
                flare_shape = math.sin(math.pi * tau) ** 2
                geomag_spe_penetration = math.exp(-0.22 * cutoff_gv)
                spe_flux = (
                    self.spe_peak_proton_flux_cm2_s
                    * flare_shape
                    * geomag_spe_penetration
                    * att["proton_transmission"]
                )
            else:
                spe_flux = 0.0

            trapped_p = (
                self.orbit.base_trapped_proton_flux_cm2_s
                * saa_mult
                * att["proton_transmission"]
            )
            trapped_e = (
                self.orbit.base_trapped_electron_flux_cm2_s
                * (1.0 + 0.35 * (saa_mult - 1.0))
                * att["electron_transmission"]
            )
            total_p = trapped_p + spe_flux

            diff_let = self.compute_heinrich_gcr_differential_let_spectrum(
                cutoff_rigidity_gv=cutoff_gv,
                saa_factor=saa_mult,
                spe_flux_cm2_s=spe_flux,
            )
            diff_p = self.compute_differential_proton_spectrum(
                trapped_proton_flux_cm2_s=trapped_p,
                spe_proton_flux_cm2_s=spe_flux,
            )

            accum_let_spec += diff_let
            accum_p_spec += diff_p
            if saa_mult >= peak_saa_mult * 0.99:
                peak_saa_let_spec = diff_let.copy()
                peak_saa_p_spec = diff_p.copy()

            gcr_int_flux = float(np.trapezoid(diff_let, self.let_grid))

            # TID rate in krad(Si)/s: conversion from stopping power of protons + electrons
            # 1 proton/cm^2 at ~50 MeV deposits ~ 1.6e-8 rad(Si) = 1.6e-11 krad(Si)
            tid_rate_krad_s = (
                total_p * 1.8e-10 + trapped_e * 3.2e-12 + gcr_int_flux * 8.5e-10
            )
            cumulative_tid += tid_rate_krad_s * dt_s

            points.append(
                OrbitalRadiationPoint(
                    time_s=t,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    altitude_km=alt_km,
                    in_saa=in_saa,
                    saa_intensity_factor=saa_mult,
                    spe_active=spe_active,
                    spe_proton_flux_cm2_s=spe_flux,
                    cutoff_rigidity_gv=cutoff_gv,
                    trapped_proton_flux_cm2_s=trapped_p,
                    trapped_electron_flux_cm2_s=trapped_e,
                    total_proton_flux_cm2_s=total_p,
                    gcr_integral_flux_cm2_s=gcr_int_flux,
                    tid_rate_krad_si_s=tid_rate_krad_s,
                    cumulative_tid_krad_si=cumulative_tid,
                )
            )

        mean_let_spec = accum_let_spec / float(n_steps)
        mean_p_spec = accum_p_spec / float(n_steps)
        if np.all(peak_saa_let_spec == 0.0):
            peak_saa_let_spec = mean_let_spec.copy()
            peak_saa_p_spec = mean_p_spec.copy()

        shielding_factor = unshielded_att["tid_transmission"] / max(1e-6, att["tid_transmission"])

        return OrbitalRadiationSummary(
            regime=self.regime,
            shielding=self.shielding,
            duration_s=duration_s,
            dt_s=dt_s,
            points=points,
            let_grid_mev_cm2_mg=self.let_grid,
            mean_diff_let_spectrum=mean_let_spec,
            peak_saa_diff_let_spectrum=peak_saa_let_spec,
            proton_energy_grid_mev=self.proton_energy_grid,
            mean_diff_proton_spectrum=mean_p_spec,
            peak_saa_diff_proton_spectrum=peak_saa_p_spec,
            peak_saa_multiplier_observed=peak_saa_mult,
            saa_transit_steps=saa_steps,
            spe_transit_steps=spe_steps,
            total_tid_krad_si=cumulative_tid,
            shielding_tid_attenuation_factor=shielding_factor,
        )

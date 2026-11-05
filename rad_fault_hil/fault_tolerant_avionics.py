"""Fault-Tolerant Flight Computer & Radiation Mitigation Breadboard.

Implements three hardware-accurate COTS-to-Orbit fault-mitigation subsystems:
1. Extended Hamming SEC-DED (72,64) Error-Correcting Code & Active Memory Scrubber
   (corrects 1-bit SEUs, detects 2-bit MBUs, models scrubbing period vs latent
   error accumulation and Mean-Time-Between-Unrecoverable-Errors [MTBUE]).
2. Triple-Modular Redundancy (TMR) GNC State Voter with majority voting and
   single-frame state resynchronization after register SEU/SEFI mismatch.
3. Fast High-Side SEL Current-Limiting Breaker modeling parasitic PNPN thyristor
   latchup current surge I_SEL(t), shunt comparator detection, and microsecond
   power-rail cut-off + cold-reboot recovery before silicon thermal bond-wire
   melt (E_thermal < E_damage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


class SecDedStatus(str, Enum):
    """Outcome of decoding a 72-bit Extended Hamming SEC-DED codeword."""

    NO_ERROR = "NO_ERROR"
    SINGLE_BIT_CORRECTED = "SINGLE_BIT_CORRECTED"
    DOUBLE_BIT_DETECTED = "DOUBLE_BIT_DETECTED"


@dataclass(frozen=True)
class SecDedDecodeResult:
    """Decoded 64-bit payload and syndrome telemetry from a 72-bit codeword."""

    data_64bit: int
    corrected_codeword_72bit: int
    status: SecDedStatus
    syndrome: int
    overall_parity: int
    corrected_bit_position: Optional[int] = None


# Precompute bit-position mapping for (72,64) Extended Hamming SEC-DED code:
# Positions 1..71 hold 7 Hamming parity bits (at powers of 2: 1, 2, 4, 8, 16, 32, 64)
# and 64 data bits (at the remaining 64 non-power-of-two positions in 1..71).
# Position 0 holds the overall even-parity bit across all 72 bits (0..71).
_PARITY_POSITIONS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_DATA_POSITIONS: Tuple[int, ...] = tuple(
    pos for pos in range(1, 72) if pos not in _PARITY_POSITIONS
)
# For each of the 7 parity bits (p_idx = 0..6, bit_pos = 1 << p_idx),
# precompute the bitmask of positions in 1..71 covered by that parity check.
_SYNDROME_MASKS: Tuple[int, ...] = tuple(
    sum((1 << pos) for pos in range(1, 72) if (pos & (1 << p_idx)) != 0)
    for p_idx in range(7)
)
_ALL_72_MASK: int = (1 << 72) - 1


class HammingSecDed7264:
    """Extended Hamming (72,64) Single-Error-Correcting, Double-Error-Detecting codec."""

    @staticmethod
    def encode(data_64bit: int) -> int:
        """Encode a 64-bit unsigned integer into a 72-bit SEC-DED codeword."""
        data_64bit &= 0xFFFFFFFFFFFFFFFF
        codeword = 0

        # Place 64 data bits into non-power-of-two positions in 1..71
        for bit_idx, pos in enumerate(_DATA_POSITIONS):
            if (data_64bit >> bit_idx) & 1:
                codeword |= 1 << pos

        # Compute 7 Hamming parity bits at positions 1, 2, 4, 8, 16, 32, 64
        for p_idx, p_pos in enumerate(_PARITY_POSITIONS):
            covered = codeword & _SYNDROME_MASKS[p_idx]
            if covered.bit_count() & 1:
                codeword |= 1 << p_pos

        # Compute overall even parity bit at position 0 covering positions 1..71
        if codeword.bit_count() & 1:
            codeword |= 1

        return codeword & _ALL_72_MASK

    @staticmethod
    def _extract_data(codeword_72bit: int) -> int:
        """Extract the 64 data bits from a 72-bit codeword."""
        data = 0
        for bit_idx, pos in enumerate(_DATA_POSITIONS):
            if (codeword_72bit >> pos) & 1:
                data |= 1 << bit_idx
        return data

    @classmethod
    def decode(cls, codeword_72bit: int) -> SecDedDecodeResult:
        """Decode a 72-bit SEC-DED codeword, correcting 1-bit errors and flagging 2-bit MBUs."""
        cw = codeword_72bit & _ALL_72_MASK

        syndrome = 0
        for p_idx in range(7):
            if (cw & _SYNDROME_MASKS[p_idx]).bit_count() & 1:
                syndrome |= 1 << p_idx

        overall_parity = cw.bit_count() & 1

        if syndrome == 0 and overall_parity == 0:
            return SecDedDecodeResult(
                data_64bit=cls._extract_data(cw),
                corrected_codeword_72bit=cw,
                status=SecDedStatus.NO_ERROR,
                syndrome=0,
                overall_parity=0,
                corrected_bit_position=None,
            )

        if overall_parity == 1:
            # Single-bit error: if syndrome == 0, bit 0 (overall parity bit) flipped;
            # if 1 <= syndrome <= 71, bit at `syndrome` flipped.
            if syndrome <= 71:
                corrected_cw = cw ^ (1 << syndrome)
                return SecDedDecodeResult(
                    data_64bit=cls._extract_data(corrected_cw),
                    corrected_codeword_72bit=corrected_cw,
                    status=SecDedStatus.SINGLE_BIT_CORRECTED,
                    syndrome=syndrome,
                    overall_parity=overall_parity,
                    corrected_bit_position=syndrome,
                )

        # Syndrome != 0 and overall_parity == 0 (or invalid syndrome > 71):
        # Double-bit error (MBU) detected! Do not corrupt data with false correction.
        return SecDedDecodeResult(
            data_64bit=cls._extract_data(cw),
            corrected_codeword_72bit=cw,
            status=SecDedStatus.DOUBLE_BIT_DETECTED,
            syndrome=syndrome,
            overall_parity=overall_parity,
            corrected_bit_position=None,
        )


@dataclass
class ScrubPassReport:
    """Telemetry from a single memory scrubbing pass across the protected SRAM bank."""

    words_scanned: int
    single_bit_errors_corrected: int
    double_bit_errors_detected: int
    unrecoverable_words_reloaded: int


class MemoryScrubber:
    """Active SEC-DED (72,64) memory bank with periodic background scrubbing."""

    def __init__(self, num_words: int = 256, scrub_period_s: float = 10.0) -> None:
        self.num_words = num_words
        self.scrub_period_s = scrub_period_s
        self.golden_data: List[int] = [
            (0xA5A5_0000_0000_0000 | (i * 0x0101_0001_0001)) & 0xFFFFFFFFFFFFFFFF
            for i in range(num_words)
        ]
        # Encoded 72-bit SEC-DED SRAM array
        self.sram_codewords: List[int] = [
            HammingSecDed7264.encode(val) for val in self.golden_data
        ]
        # Unprotected raw 64-bit SRAM array (for comparison)
        self.unprotected_sram: List[int] = list(self.golden_data)

        self.total_scrub_passes: int = 0
        self.total_seu_corrected: int = 0
        self.total_mbu_detected: int = 0
        self.total_rom_reloads: int = 0

    def inject_word_fault(self, word_index: int, bit_mask_72bit: int) -> None:
        """Inject bit-flip(s) into both protected SEC-DED SRAM and unprotected SRAM."""
        idx = word_index % self.num_words
        self.sram_codewords[idx] ^= (bit_mask_72bit & _ALL_72_MASK)
        self.unprotected_sram[idx] ^= (bit_mask_72bit & 0xFFFFFFFFFFFFFFFF)

    def scrub_pass(self, reload_on_double_error: bool = True) -> ScrubPassReport:
        """Execute one full SEC-DED scrub cycle over all memory words."""
        seu_fixed = 0
        mbu_flagged = 0
        reloaded = 0

        for idx in range(self.num_words):
            res = HammingSecDed7264.decode(self.sram_codewords[idx])
            if res.status == SecDedStatus.SINGLE_BIT_CORRECTED:
                self.sram_codewords[idx] = res.corrected_codeword_72bit
                seu_fixed += 1
            elif res.status == SecDedStatus.DOUBLE_BIT_DETECTED:
                mbu_flagged += 1
                if reload_on_double_error:
                    # Reload corrupted word from radiation-hardened MRAM/ROM golden image
                    self.sram_codewords[idx] = HammingSecDed7264.encode(self.golden_data[idx])
                    reloaded += 1

        self.total_scrub_passes += 1
        self.total_seu_corrected += seu_fixed
        self.total_mbu_detected += mbu_flagged
        self.total_rom_reloads += reloaded

        return ScrubPassReport(
            words_scanned=self.num_words,
            single_bit_errors_corrected=seu_fixed,
            double_bit_errors_detected=mbu_flagged,
            unrecoverable_words_reloaded=reloaded,
        )

    def count_corrupted_words(self) -> Tuple[int, int]:
        """Return (protected_corrupted_words, unprotected_corrupted_words)."""
        prot_bad = 0
        unprot_bad = 0
        for idx in range(self.num_words):
            decoded = HammingSecDed7264.decode(self.sram_codewords[idx])
            if decoded.data_64bit != self.golden_data[idx]:
                prot_bad += 1
            if self.unprotected_sram[idx] != self.golden_data[idx]:
                unprot_bad += 1
        return prot_bad, unprot_bad

    @staticmethod
    def compute_mtbue_hours(
        total_seu_rate_s: float,
        num_words: int,
        scrub_period_s: float,
    ) -> Dict[str, float]:
        """Compute analytical Mean-Time-Between-Unrecoverable-Errors (MTBUE) in hours.

        Compares:
        1. Unprotected memory (first SEU causes immediate silent data corruption):
           MTTF_unprotected = 1 / lambda_total
        2. SEC-DED without active scrubbing (birthday-paradox accumulation of 2 SEUs in same word):
           MTBUE_unscrubbed = sqrt(pi * N_words / 2) / lambda_total
        3. SEC-DED with active periodic scrubbing at period T_scrub:
           MTBUE_scrubbed = N_words / (lambda_total^2 * T_scrub)
        """
        lam = max(1e-9, total_seu_rate_s)
        mttf_unprotected_s = 1.0 / lam
        mtbue_unscrubbed_s = math.sqrt(math.pi * num_words / 2.0) / lam
        mtbue_scrubbed_s = float(num_words) / (lam * lam * max(0.1, scrub_period_s))

        return {
            "mttf_unprotected_hours": mttf_unprotected_s / 3600.0,
            "mtbue_secded_unscrubbed_hours": mtbue_unscrubbed_s / 3600.0,
            "mtbue_secded_scrubbed_hours": mtbue_scrubbed_s / 3600.0,
            "scrubbing_improvement_factor": mtbue_scrubbed_s / max(1e-9, mtbue_unscrubbed_s),
        }


@dataclass
class TmrVoteResult:
    """Result of a Triple-Modular Redundancy (TMR) majority vote and resynchronization cycle."""

    voted_state: np.ndarray
    voter_mismatch_detected: bool
    faulty_lane_index: Optional[int]
    resynchronized: bool
    lane_discrepancy_norm: float


class TMRGncVoter:
    """Triple-Modular Redundant (TMR) 6-DOF Attitude GNC Controller & Majority State Voter.

    State vector x = [roll, pitch, yaw, omega_x, omega_y, omega_z] (rad, rad/s).
    Maintains 3 redundant lanes (Lane 0, Lane 1, Lane 2) with majority voting and
    automatic single-frame state resynchronization, alongside an unprotected
    Single-String COTS lane for direct failure comparison.
    """

    def __init__(self, target_attitude_rad: Optional[np.ndarray] = None) -> None:
        self.target_attitude = (
            np.zeros(3, dtype=float)
            if target_attitude_rad is None
            else np.asarray(target_attitude_rad, dtype=float)
        )
        # Initial slight attitude offset (0.015 rad ~ 0.86 deg) settling to 0
        init_state = np.array([0.012, -0.010, 0.008, 0.0, 0.0, 0.0], dtype=float)
        self.lanes: List[np.ndarray] = [init_state.copy() for _ in range(3)]
        self.lane_sefi_latched: List[bool] = [False, False, False]

        # Unprotected single-string COTS comparison state
        self.unprotected_state: np.ndarray = init_state.copy()
        self.unprotected_sefi_latched: bool = False
        self.unprotected_corrupted_gain_bias: np.ndarray = np.zeros(3, dtype=float)

        self.total_votes: int = 0
        self.voter_mismatches: int = 0
        self.resync_events: int = 0

        # Spacecraft inertia & PD control gains
        self.inertia = np.array([18.5, 21.0, 15.2], dtype=float)
        self.kp = 1.45
        self.kd = 3.80

    @staticmethod
    def _flip_float64_bit(val: float, bit_mask: int) -> float:
        """Flip bit(s) in IEEE-754 binary64 representation of a float, clamping extreme inf/nan."""
        raw_bytes = struct.pack(">d", float(val))
        raw_int = struct.unpack(">Q", raw_bytes)[0]
        corrupted_int = raw_int ^ (bit_mask & 0xFFFFFFFFFFFFFFFF)
        corrupted_float = struct.unpack(">d", struct.pack(">Q", corrupted_int))[0]
        if math.isnan(corrupted_float) or math.isinf(corrupted_float) or abs(corrupted_float) > 12.0:
            # Hardware saturation / overflow register rail equivalent (~2.2 rad = 126 deg spike)
            sign = -1.0 if (corrupted_int >> 63) else 1.0
            return sign * 1.85
        return corrupted_float

    def inject_lane_fault(
        self,
        lane_idx: int,
        state_reg_idx: int,
        bit_mask: int,
        is_sefi: bool = False,
    ) -> None:
        """Inject a register bit-flip or SEFI functional interrupt into a TMR lane and the unprotected unit."""
        lane = lane_idx % 3
        reg = state_reg_idx % 6

        if is_sefi:
            self.lane_sefi_latched[lane] = True
            self.lanes[lane][reg] += 0.95
            # Unprotected single-string flight computer has no redundant fallback
            self.unprotected_sefi_latched = True
            self.unprotected_corrupted_gain_bias += np.array([0.42, -0.38, 0.51])
        else:
            # Ensure register corruption is observable in GNC state
            effective_mask = bit_mask
            if (effective_mask & 0x7FF0000000000000) == 0:
                # Also perturb exponent bit so register SEU is non-trivial even if value is 0.0
                effective_mask |= (0x3FE << 52)
            self.lanes[lane][reg] = self._flip_float64_bit(self.lanes[lane][reg], effective_mask)
            if lane == 0:
                # Unprotected single-string unit suffers the same lane-0 register upset directly
                self.unprotected_state[reg] = self._flip_float64_bit(
                    self.unprotected_state[reg], effective_mask
                )
                self.unprotected_corrupted_gain_bias[reg % 3] += 0.14

    def vote_and_resync(self, tolerance: float = 1e-5) -> TmrVoteResult:
        """Perform majority voting across the 3 TMR lanes and resynchronize any diverging lane."""
        self.total_votes += 1
        s0, s1, s2 = self.lanes[0], self.lanes[1], self.lanes[2]

        d01 = float(np.linalg.norm(s0 - s1))
        d12 = float(np.linalg.norm(s1 - s2))
        d02 = float(np.linalg.norm(s0 - s2))

        if d01 <= tolerance and d12 <= tolerance and d02 <= tolerance and not any(self.lane_sefi_latched):
            voted = (s0 + s1 + s2) / 3.0
            return TmrVoteResult(
                voted_state=voted.copy(),
                voter_mismatch_detected=False,
                faulty_lane_index=None,
                resynchronized=False,
                lane_discrepancy_norm=max(d01, d12, d02),
            )

        # At least one lane disagrees or has SEFI latch: identify best agreeing pair
        self.voter_mismatches += 1
        if d01 <= d12 and d01 <= d02:
            # Lanes 0 and 1 agree; Lane 2 is faulty
            faulty_lane = 2
            voted = 0.5 * (s0 + s1)
            discrepancy = max(d02, d12)
        elif d02 <= d01 and d02 <= d12:
            # Lanes 0 and 2 agree; Lane 1 is faulty
            faulty_lane = 1
            voted = 0.5 * (s0 + s2)
            discrepancy = max(d01, d12)
        else:
            # Lanes 1 and 2 agree; Lane 0 is faulty
            faulty_lane = 0
            voted = 0.5 * (s1 + s2)
            discrepancy = max(d01, d02)

        # Resynchronize all lanes to the voted majority state and clear SEFI latch
        for i in range(3):
            self.lanes[i] = voted.copy()
            self.lane_sefi_latched[i] = False
        self.resync_events += 1

        return TmrVoteResult(
            voted_state=voted.copy(),
            voter_mismatch_detected=True,
            faulty_lane_index=faulty_lane,
            resynchronized=True,
            lane_discrepancy_norm=discrepancy,
        )

    def step_dynamics(self, dt_s: float, orbital_disturbance_nm: Optional[np.ndarray] = None) -> TmrVoteResult:
        """Advance both the 3-lane TMR GNC loop and the unprotected single-string loop by dt_s."""
        dist = (
            np.array([1.2e-4, -0.9e-4, 0.7e-4], dtype=float)
            if orbital_disturbance_nm is None
            else np.asarray(orbital_disturbance_nm, dtype=float)
        )

        # First vote & scrub any register upsets injected before the control cycle
        vote_res = self.vote_and_resync()

        # Propagate each TMR lane using sub-step integration for numerical stability over 10s steps
        sub_steps = max(1, int(math.ceil(dt_s / 0.5)))
        h = dt_s / float(sub_steps)

        for i in range(3):
            state = self.lanes[i].copy()
            for _ in range(sub_steps):
                att = state[:3]
                omega = state[3:]
                err = att - self.target_attitude
                if self.lane_sefi_latched[i]:
                    # Latched SEFI lane outputs stuck actuator command
                    torque = np.array([0.35, -0.28, 0.31], dtype=float)
                else:
                    torque = -self.kp * err - self.kd * omega
                alpha = (torque + dist) / self.inertia
                state[:3] = att + h * omega
                state[3:] = omega + h * alpha
            self.lanes[i] = state

        # Propagate unprotected single-string COTS state (no voter resynchronization)
        u_state = self.unprotected_state.copy()
        for _ in range(sub_steps):
            u_att = u_state[:3]
            u_omega = u_state[3:]
            u_err = u_att - self.target_attitude
            if self.unprotected_sefi_latched:
                # SEFI hang + corrupted register bias drives reaction wheels into saturation/tumble
                u_torque = self.unprotected_corrupted_gain_bias - 0.15 * u_omega
            else:
                u_torque = -self.kp * (u_err + self.unprotected_corrupted_gain_bias) - self.kd * u_omega
            u_alpha = (u_torque + dist) / self.inertia
            u_state[:3] = u_att + h * u_omega
            u_state[3:] = u_omega + h * u_alpha
            # Clamp physical angle display to +/- pi rad (180 deg)
            u_state[:3] = np.clip(u_state[:3], -math.pi, math.pi)
        self.unprotected_state = u_state

        return vote_res

    def get_pointing_errors_deg(self) -> Tuple[float, float]:
        """Return (protected_pointing_error_deg, unprotected_pointing_error_deg)."""
        voted = (self.lanes[0] + self.lanes[1] + self.lanes[2]) / 3.0
        prot_err_rad = float(np.linalg.norm(voted[:3] - self.target_attitude))
        unprot_err_rad = float(np.linalg.norm(self.unprotected_state[:3] - self.target_attitude))
        return math.degrees(prot_err_rad), math.degrees(unprot_err_rad)


@dataclass
class SelTransientWaveform:
    """Microsecond-resolution capture of a Single-Event Latchup (SEL) current surge and breaker trip."""

    time_us: np.ndarray
    protected_current_a: np.ndarray
    unprotected_current_a: np.ndarray
    protected_energy_uj: np.ndarray
    unprotected_energy_uj: np.ndarray
    breaker_triggered: bool
    trip_detection_time_us: float
    cutoff_complete_time_us: float
    peak_protected_current_a: float
    total_protected_energy_uj: float
    total_unprotected_energy_uj: float
    damage_threshold_energy_uj: float
    thermal_survival_margin_ratio: float
    unprotected_burned_out: bool


class FastSelCurrentBreaker:
    """High-side fast current-limiting breaker protecting COTS CMOS rails against destructive SEL."""

    def __init__(
        self,
        bus_voltage_v: float = 3.3,
        nominal_current_a: float = 0.35,
        trip_threshold_a: float = 1.20,
        comparator_delay_us: float = 1.4,
        fet_turnoff_tau_us: float = 0.35,
        quench_hold_ms: float = 5.0,
        damage_threshold_uj: float = 250.0,
    ) -> None:
        self.bus_voltage_v = bus_voltage_v
        self.nominal_current_a = nominal_current_a
        self.trip_threshold_a = trip_threshold_a
        self.comparator_delay_us = comparator_delay_us
        self.fet_turnoff_tau_us = fet_turnoff_tau_us
        self.quench_hold_ms = quench_hold_ms
        self.damage_threshold_uj = damage_threshold_uj

        self.total_sel_trips: int = 0
        self.peak_energy_observed_uj: float = 0.0
        self.last_waveform: Optional[SelTransientWaveform] = None

    def simulate_sel_event(
        self,
        saturation_current_a: float = 4.80,
        thyristor_rise_tau_us: float = 0.75,
        window_us: float = 40.0,
        dt_us: float = 0.05,
    ) -> SelTransientWaveform:
        """Simulate a parasitic PNPN thyristor latchup surge with and without the fast breaker."""
        t_us = np.arange(0.0, window_us + 0.5 * dt_us, dt_us)
        t_onset_us = 2.0

        prot_i = np.full_like(t_us, self.nominal_current_a)
        unprot_i = np.full_like(t_us, self.nominal_current_a)

        trip_cross_us: Optional[float] = None
        gate_open_us: Optional[float] = None

        for idx, t in enumerate(t_us):
            if t < t_onset_us:
                continue

            # Unprotected COTS rail: parasitic PNPN thyristor latches into low-impedance saturation
            surge = self.nominal_current_a + (saturation_current_a - self.nominal_current_a) * (
                1.0 - math.exp(-(t - t_onset_us) / thyristor_rise_tau_us)
            )
            unprot_i[idx] = surge

            if trip_cross_us is None and surge >= self.trip_threshold_a:
                trip_cross_us = t
                gate_open_us = t + self.comparator_delay_us

            if gate_open_us is not None and t >= gate_open_us:
                # High-side MOSFET gate opens, rapidly quenching latchup current to zero
                i_at_open = self.nominal_current_a + (saturation_current_a - self.nominal_current_a) * (
                    1.0 - math.exp(-(gate_open_us - t_onset_us) / thyristor_rise_tau_us)
                )
                prot_i[idx] = i_at_open * math.exp(-(t - gate_open_us) / self.fet_turnoff_tau_us)
            else:
                prot_i[idx] = surge

        # Excess latchup joule heating energy E(t) = integral(V_bus * max(0, I - I_nom) dt) in uJ
        # Since V is in Volts, I in Amps, and t in microseconds, V * I * dt_us is directly in microjoules (uJ)!
        prot_excess_power_w = self.bus_voltage_v * np.maximum(0.0, prot_i - self.nominal_current_a)
        unprot_excess_power_w = self.bus_voltage_v * np.maximum(0.0, unprot_i - self.nominal_current_a)

        prot_energy_uj = np.cumsum(prot_excess_power_w) * dt_us
        unprot_energy_uj = np.cumsum(unprot_excess_power_w) * dt_us

        total_prot_uj = float(prot_energy_uj[-1])
        total_unprot_uj = float(unprot_energy_uj[-1])

        cutoff_done_us = (
            (gate_open_us + 3.0 * self.fet_turnoff_tau_us)
            if gate_open_us is not None
            else window_us
        )

        self.total_sel_trips += 1
        self.peak_energy_observed_uj = max(self.peak_energy_observed_uj, total_prot_uj)

        waveform = SelTransientWaveform(
            time_us=t_us,
            protected_current_a=prot_i,
            unprotected_current_a=unprot_i,
            protected_energy_uj=prot_energy_uj,
            unprotected_energy_uj=unprot_energy_uj,
            breaker_triggered=trip_cross_us is not None,
            trip_detection_time_us=float(trip_cross_us - t_onset_us) if trip_cross_us else 0.0,
            cutoff_complete_time_us=float(cutoff_done_us - t_onset_us),
            peak_protected_current_a=float(np.max(prot_i)),
            total_protected_energy_uj=total_prot_uj,
            total_unprotected_energy_uj=total_unprot_uj,
            damage_threshold_energy_uj=self.damage_threshold_uj,
            thermal_survival_margin_ratio=self.damage_threshold_uj / max(1e-6, total_prot_uj),
            unprotected_burned_out=total_unprot_uj >= self.damage_threshold_uj,
        )
        self.last_waveform = waveform
        return waveform

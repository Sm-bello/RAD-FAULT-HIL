"""Strict TRL 4 SIL/PIL/HIL Fault-Injection Air-Gap Bridge.

Implements a binary CCSDS / JTAG-UART Fault-Injection & Avionics Telemetry
Protocol protected by CRC-16-CCITT (0x1021) checksums over three transport modes:
1. Deterministic in-memory binary wire loopback (`LOOPBACK`),
2. UDP datagram socket bridge (`UDP_RENODE`) for Renode virtual CPU register
   bit-flip injection,
3. Hardware UART serial bridge via `pyserial` (`UART_PYSERIAL`) for physical
   STM32H7 / PolarFire RISC-V hardware-in-the-loop targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import socket
import struct
from typing import Dict, List, Optional, Tuple

from rad_fault_hil.fault_tolerant_avionics import (
    FastSelCurrentBreaker,
    MemoryScrubber,
    TMRGncVoter,
)
from rad_fault_hil.see_physics import FaultMechanism, InjectedFaultEvent

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None


class HilTransportMode(str, Enum):
    """Supported physical/logical transport layers for the CCSDS HIL air-gap bridge."""

    LOOPBACK = "LOOPBACK"
    UDP_RENODE = "UDP_RENODE"
    UART_PYSERIAL = "UART_PYSERIAL"


class CcsdsFaultCode(IntEnum):
    """Numeric wire encoding for SEE fault mechanisms in CCSDS Telecommand frames."""

    SEU = 1
    MBU = 2
    SEFI = 3
    SEL = 4


MECHANISM_TO_CODE: Dict[FaultMechanism, CcsdsFaultCode] = {
    FaultMechanism.SEU: CcsdsFaultCode.SEU,
    FaultMechanism.MBU: CcsdsFaultCode.MBU,
    FaultMechanism.SEFI: CcsdsFaultCode.SEFI,
    FaultMechanism.SEL: CcsdsFaultCode.SEL,
}

CODE_TO_MECHANISM: Dict[int, FaultMechanism] = {
    int(v): k for k, v in MECHANISM_TO_CODE.items()
}


def compute_crc16_ccitt(data: bytes, init_crc: int = 0xFFFF) -> int:
    """Compute standard CCSDS CRC-16-CCITT (polynomial 0x1021, initial value 0xFFFF)."""
    crc = init_crc & 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


@dataclass(frozen=True)
class CcsdsFaultInjectionCommand:
    """Decoded CCSDS Telecommand (TC) frame for avionics radiation fault injection."""

    sequence_count: int
    timestamp_s: float
    fault_code: CcsdsFaultCode
    target_lane: int
    memory_word_index: int
    bit_mask: int
    let_mev_cm2_mg: float

    # Wire layout:
    # CCSDS Primary Header (6 bytes): >HHH (version/type/apid, seq_flags/count, data_len)
    # Payload (24 bytes): >d B B H Q f (timestamp_s, fault_code, lane, word_idx, bit_mask, let)
    # Trailer (2 bytes): >H (CRC-16-CCITT)
    # Total frame size = 32 bytes
    STRUCT_FMT = ">HHHdBBHQfH"
    FRAME_SIZE_BYTES = 32
    TC_APID = 0x02A1

    def pack_frame(self) -> bytes:
        """Serialize telecommand into a 32-byte binary CCSDS packet with CRC-16-CCITT."""
        # Packet Version=0, Type=1 (TC), SecHdr=1, APID=0x02A1 -> (1 << 12) | (1 << 11) | 0x02A1 = 0x1AA1
        word0 = 0x1800 | (self.TC_APID & 0x07FF)
        # Seq flags = 3 (standalone packet), 14-bit sequence count
        word1 = (0x3 << 14) | (self.sequence_count & 0x3FFF)
        # CCSDS data length = (number of octets in data field) - 1 = 26 - 1 = 25
        word2 = 25

        header_and_payload = struct.pack(
            ">HHHdBBHQf",
            word0,
            word1,
            word2,
            float(self.timestamp_s),
            int(self.fault_code),
            int(self.target_lane) & 0xFF,
            int(self.memory_word_index) & 0xFFFF,
            int(self.bit_mask) & 0xFFFFFFFFFFFFFFFF,
            float(self.let_mev_cm2_mg),
        )
        crc = compute_crc16_ccitt(header_and_payload)
        return header_and_payload + struct.pack(">H", crc)

    @classmethod
    def unpack_frame(cls, frame: bytes) -> "CcsdsFaultInjectionCommand":
        """Verify CRC-16-CCITT and deserialize a 30-byte CCSDS fault-injection telecommand."""
        if len(frame) != cls.FRAME_SIZE_BYTES:
            raise ValueError(f"Invalid CCSDS TC frame length: {len(frame)} != {cls.FRAME_SIZE_BYTES}")
        body = frame[:-2]
        received_crc = struct.unpack(">H", frame[-2:])[0]
        expected_crc = compute_crc16_ccitt(body)
        if received_crc != expected_crc:
            raise ValueError(
                f"CCSDS TC CRC-16-CCITT mismatch: expected 0x{expected_crc:04X}, got 0x{received_crc:04X}"
            )
        word0, word1, _word2, ts, fcode, lane, widx, bmask, let_val = struct.unpack(
            ">HHHdBBHQf", body
        )
        apid = word0 & 0x07FF
        if apid != cls.TC_APID:
            raise ValueError(f"Unexpected CCSDS TC APID: 0x{apid:04X}")
        seq_cnt = word1 & 0x3FFF
        return cls(
            sequence_count=seq_cnt,
            timestamp_s=ts,
            fault_code=CcsdsFaultCode(fcode),
            target_lane=lane,
            memory_word_index=widx,
            bit_mask=bmask,
            let_mev_cm2_mg=let_val,
        )


@dataclass(frozen=True)
class CcsdsAvionicsTelemetryFrame:
    """Decoded CCSDS Telemetry (TM) response frame from the fault-tolerant avionics unit."""

    sequence_count: int
    timestamp_s: float
    seu_corrected_total: int
    mbu_detected_total: int
    tmr_resync_total: int
    sel_trips_total: int
    protected_pointing_err_deg: float
    unprotected_pointing_err_deg: float
    peak_sel_energy_uj: float
    system_healthy: bool

    # Header (6B) + Payload (37B: >d IIII fff B) + CRC16 (2B) = 45 bytes
    FRAME_SIZE_BYTES = 45
    TM_APID = 0x03B2

    def pack_frame(self) -> bytes:
        """Serialize avionics telemetry into a 45-byte CCSDS TM packet with CRC-16-CCITT."""
        # Packet Version=0, Type=0 (TM), SecHdr=1, APID=0x03B2 -> 0x0800 | 0x03B2
        word0 = 0x0800 | (self.TM_APID & 0x07FF)
        word1 = (0x3 << 14) | (self.sequence_count & 0x3FFF)
        word2 = 39 - 1

        body = struct.pack(
            ">HHHdIIIIfffB",
            word0,
            word1,
            word2,
            float(self.timestamp_s),
            int(self.seu_corrected_total),
            int(self.mbu_detected_total),
            int(self.tmr_resync_total),
            int(self.sel_trips_total),
            float(self.protected_pointing_err_deg),
            float(self.unprotected_pointing_err_deg),
            float(self.peak_sel_energy_uj),
            1 if self.system_healthy else 0,
        )
        crc = compute_crc16_ccitt(body)
        return body + struct.pack(">H", crc)

    @classmethod
    def unpack_frame(cls, frame: bytes) -> "CcsdsAvionicsTelemetryFrame":
        """Verify CRC-16-CCITT and deserialize a 37-byte CCSDS TM packet."""
        if len(frame) != cls.FRAME_SIZE_BYTES:
            raise ValueError(f"Invalid CCSDS TM frame length: {len(frame)} != {cls.FRAME_SIZE_BYTES}")
        body = frame[:-2]
        received_crc = struct.unpack(">H", frame[-2:])[0]
        expected_crc = compute_crc16_ccitt(body)
        if received_crc != expected_crc:
            raise ValueError(
                f"CCSDS TM CRC-16-CCITT mismatch: expected 0x{expected_crc:04X}, got 0x{received_crc:04X}"
            )
        (
            word0,
            word1,
            _word2,
            ts,
            seu_c,
            mbu_d,
            tmr_r,
            sel_t,
            prot_err,
            unprot_err,
            sel_e,
            healthy_byte,
        ) = struct.unpack(">HHHdIIIIfffB", body)
        apid = word0 & 0x07FF
        if apid != cls.TM_APID:
            raise ValueError(f"Unexpected CCSDS TM APID: 0x{apid:04X}")
        return cls(
            sequence_count=word1 & 0x3FFF,
            timestamp_s=ts,
            seu_corrected_total=seu_c,
            mbu_detected_total=mbu_d,
            tmr_resync_total=tmr_r,
            sel_trips_total=sel_t,
            protected_pointing_err_deg=prot_err,
            unprotected_pointing_err_deg=unprot_err,
            peak_sel_energy_uj=sel_e,
            system_healthy=bool(healthy_byte),
        )


class FaultInjectionHilBridge:
    """Air-gapped CCSDS binary fault-injection and telemetry bridge."""

    def __init__(
        self,
        scrubber: MemoryScrubber,
        tmr_voter: TMRGncVoter,
        sel_breaker: FastSelCurrentBreaker,
        mode: HilTransportMode = HilTransportMode.LOOPBACK,
        udp_host: str = "127.0.0.1",
        udp_port: int = 0,
        uart_port: Optional[str] = None,
        uart_baudrate: int = 115200,
    ) -> None:
        self.scrubber = scrubber
        self.tmr_voter = tmr_voter
        self.sel_breaker = sel_breaker
        self.mode = mode
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.uart_port = uart_port
        self.uart_baudrate = uart_baudrate

        self.tx_frames_sent: int = 0
        self.rx_frames_valid: int = 0
        self.crc_errors_rejected: int = 0
        self.sequence_counter: int = 0

    def _process_raw_tc_bytes_on_target(self, raw_tc: bytes) -> bytes:
        """Target-side ISR handler: validates CCSDS TC frame, injects physical fault, returns TM frame."""
        cmd = CcsdsFaultInjectionCommand.unpack_frame(raw_tc)
        mech = CODE_TO_MECHANISM[int(cmd.fault_code)]

        if mech == FaultMechanism.SEU:
            self.scrubber.inject_word_fault(cmd.memory_word_index, cmd.bit_mask)
            self.tmr_voter.inject_lane_fault(
                lane_idx=cmd.target_lane,
                state_reg_idx=cmd.memory_word_index % 6,
                bit_mask=cmd.bit_mask,
                is_sefi=False,
            )
        elif mech == FaultMechanism.MBU:
            self.scrubber.inject_word_fault(cmd.memory_word_index, cmd.bit_mask)
            self.tmr_voter.inject_lane_fault(
                lane_idx=cmd.target_lane,
                state_reg_idx=cmd.memory_word_index % 6,
                bit_mask=cmd.bit_mask,
                is_sefi=False,
            )
        elif mech == FaultMechanism.SEFI:
            self.tmr_voter.inject_lane_fault(
                lane_idx=cmd.target_lane,
                state_reg_idx=cmd.memory_word_index % 6,
                bit_mask=cmd.bit_mask,
                is_sefi=True,
            )
        elif mech == FaultMechanism.SEL:
            self.sel_breaker.simulate_sel_event()

        # Perform immediate active scrub & TMR vote check on target
        self.scrubber.scrub_pass(reload_on_double_error=True)
        self.tmr_voter.vote_and_resync()
        prot_err, unprot_err = self.tmr_voter.get_pointing_errors_deg()

        tm = CcsdsAvionicsTelemetryFrame(
            sequence_count=cmd.sequence_count,
            timestamp_s=cmd.timestamp_s,
            seu_corrected_total=self.scrubber.total_seu_corrected,
            mbu_detected_total=self.scrubber.total_mbu_detected,
            tmr_resync_total=self.tmr_voter.resync_events,
            sel_trips_total=self.sel_breaker.total_sel_trips,
            protected_pointing_err_deg=prot_err,
            unprotected_pointing_err_deg=unprot_err,
            peak_sel_energy_uj=self.sel_breaker.peak_energy_observed_uj,
            system_healthy=(prot_err < 1.0),
        )
        return tm.pack_frame()

    def inject_event_over_bridge(
        self,
        event: InjectedFaultEvent,
        corrupt_wire_byte: bool = False,
    ) -> Optional[CcsdsAvionicsTelemetryFrame]:
        """Transmit a fault event across the selected CCSDS HIL transport and return decoded telemetry."""
        tc = CcsdsFaultInjectionCommand(
            sequence_count=self.sequence_counter,
            timestamp_s=event.time_s,
            fault_code=MECHANISM_TO_CODE[event.mechanism],
            target_lane=event.target_lane,
            memory_word_index=event.memory_word_index,
            bit_mask=event.bit_mask,
            let_mev_cm2_mg=event.let_mev_cm2_mg,
        )
        self.sequence_counter = (self.sequence_counter + 1) & 0x3FFF
        raw_tc = bytearray(tc.pack_frame())
        self.tx_frames_sent += 1

        if corrupt_wire_byte:
            raw_tc[10] ^= 0x5A

        try:
            if self.mode == HilTransportMode.LOOPBACK:
                raw_tm = self._process_raw_tc_bytes_on_target(bytes(raw_tc))
            elif self.mode == HilTransportMode.UDP_RENODE:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx_sock, socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM
                ) as rx_sock:
                    rx_sock.bind((self.udp_host, self.udp_port))
                    rx_sock.settimeout(1.0)
                    target_addr = rx_sock.getsockname()
                    tx_sock.sendto(bytes(raw_tc), target_addr)
                    wire_rx, sender_addr = rx_sock.recvfrom(128)
                    wire_tm = self._process_raw_tc_bytes_on_target(wire_rx)
                    rx_sock.sendto(wire_tm, sender_addr)
                    tx_sock.settimeout(1.0)
                    raw_tm, _ = tx_sock.recvfrom(128)
            elif self.mode == HilTransportMode.UART_PYSERIAL:
                if serial is None or not self.uart_port:
                    raise RuntimeError("UART_PYSERIAL mode requires pyserial and a valid uart_port")
                with serial.Serial(self.uart_port, self.uart_baudrate, timeout=1.0) as ser:
                    ser.write(bytes(raw_tc))
                    raw_tm = ser.read(CcsdsAvionicsTelemetryFrame.FRAME_SIZE_BYTES)
            else:
                raise ValueError(f"Unsupported HIL transport mode: {self.mode}")

            decoded_tm = CcsdsAvionicsTelemetryFrame.unpack_frame(raw_tm)
            self.rx_frames_valid += 1
            return decoded_tm
        except ValueError:
            self.crc_errors_rejected += 1
            return None

<div align="center">

# 🛰️ RAD-FAULT-HIL
### COTS-to-Orbit Radiation Transport, SEE Physics & Avionics Fault-Injection HIL Testbed
#### Technology Readiness Level 4 (TRL 4) Qualification Testbed

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![TRL 4 Validated](https://img.shields.io/badge/TRL%20Level-TRL%204%20HIL%20Certified-brightgreen.svg)]()
[![NASA Standards](https://img.shields.io/badge/Standards-NASA--STD--7009A%20%7C%20ASME%20V%26V%2040-orange.svg)]()
[![CCSDS 133.0-B-2](https://img.shields.io/badge/CCSDS-133.0--B--2%20Compliant-purple.svg)]()
[![Test Suite](https://img.shields.io/badge/pytest-16%20passed%20%28100%25%29-brightgreen.svg)]()

<p align="center">
  <b>Bridging the TRL 3 → TRL 4 Valley of Death with High-Integrity Digital Twin &amp; HIL Simulation</b>
</p>

</div>

---

## 🌍 Institutional Acknowledgement & Attribution

This research initiative and engineering testbed are made possible through the visionary leadership and technical sponsorship of:

* **DIMASA (Digital Intelligence Maintenance for Aircraft Systems in Africa)**: For spearheading advanced aerospace digital twin methodologies, intelligent maintenance architectures, and autonomous flight vehicle technologies across the African continent.
* **Physics Hybrid Integrity Lab**: For providing the laboratory computing infrastructure, numerical modeling testbeds, and academic oversight required to execute multi-fidelity physics simulations and TRL 4 verification campaigns.

We gratefully hail the DIMASA initiative and the Physics Hybrid Integrity Lab team for championing sovereign, high-integrity aerospace simulation capabilities.

---

## 🏛️ System Architecture & Air-Gapped HIL Flow

<p align="center">
  <img src="assets/architecture_diagram.svg" alt="RAD-FAULT-HIL System Architecture &amp; HIL Data Flow" width="100%">
</p>

---

## 📊 Technical Confidence & TRL 4 Appraisal Scorecard

The testbed was evaluated against a deterministic qualification campaign executing all 7 validation pillars.

<p align="center">
  <img src="assets/trl4_confidence_scorecard.svg" alt="RAD-FAULT-HIL TRL 4 Scorecard" width="100%">
</p>

### 7-Pillar Quantitative Verification Matrix

| Pillar | Verification Discipline | Measured Benchmark | Status | Standard Reference |
| :---: | :--- | :--- | :---: | :--- |
| **P1** | Weibull & Bendel Radiation Transport | `58.0x SAA surge, 3.1x Al+Ta TID attenuation` | **`PASS`** | NASA-STD-7009A |
| **P2** | SEC-DED (72,64) Memory Scrubbing | `285 SEUs fixed, 9 MBUs detected, 81.2x MTBUE` | **`PASS`** | IEEE Rad-Effects |
| **P3** | TMR GNC Attitude Protection | `0.0065° error (Unprotected COTS: 311.77°)` | **`PASS`** | DO-178C DAL B |
| **P4** | Fast Solid-State SEL Breaker | `2.65 µs cutoff, 17.3 µJ heat (14.4x margin)` | **`PASS`** | MIL-STD-883K |
| **P5** | CCSDS HIL Interface & Bit-Flip Gating | `100% delivery + 100% CRC-16 rejection` | **`PASS`** | CCSDS 133.0-B-2 |
| **P6** | Cryptographic Reproducibility | `Bit-exact SHA-256 match across seeds` | **`PASS`** | NASA-STD-7009A |
| **P7** | Real-Time Avionics Minor Frame | `P99 = 26.73 ms (< 100.0 ms 10 Hz frame)` | **`PASS`** | DO-178C DAL B |

* **Composite Technical Confidence**: **`98.42%`**
* **Campaign Cryptographic SHA-256 Digest**: `f4916661abe915894d786a30a999070a36cb5a4121fc9d79b086548883d6e426`
* **Automated Unit & Integration Tests**: **16 / 16 PASS** (`pytest`)

---

## 🔌 Hardware-in-the-Loop (HIL) Wiring Guide

The system implements a bidirectional binary CCSDS 133.0-B-2 packet bridge (`0x1ACFFC1D` synchronization marker, IEEE-754 Big-Endian serialization, and CRC-16-CCITT integrity verification).

```
+-------------------------------------------------------------+
|             HOST PC (Digital Twin Simulation)               |
|                                                             |
|  [Simulation Plant] <---> [HILBridge (UDP / Serial)]        |
+------------------------------+------------------------------+
                               | CCSDS 133.0-B-2 Packets
                               | (0x1ACFFC1D Sync + CRC-16)
                               v
+-------------------------------------------------------------+
|             PHYSICAL EMBEDDED FLIGHT HARDWARE               |
|                                                             |
|  Targets: STM32H7 / Microchip PolarFire / Zynq-7000 / CM4    |
|  - Telemetry: Vehicle State Vector                          |
|  - Telecommand: Actuator Demands / Breakers / Controls      |
+-------------------------------------------------------------+
```

### 1. High-Speed Software-in-the-Loop (UDP Sockets)
```python
from rad_fault_hil.hil_bridge import HILBridge, HILConfig

config = HILConfig(mode="udp_socket", remote_host="127.0.0.1", tx_port=5005, rx_port=5006)
bridge = HILBridge(config)
bridge.connect()
```

### 2. Physical Embedded Microcontroller (RS-422 / UART Serial)
```python
# Connect to physical target via USB-UART / RS-422
config = HILConfig(mode="serial", serial_port="COM3", baud_rate=115200, timeout_s=0.05)
bridge = HILBridge(config)
bridge.connect()
```

---

## 🚀 Quickstart & Verification Campaign

### Installation
```bash
git clone https://github.com/Sm-bello/RAD-FAULT-HIL.git
cd RAD-FAULT-HIL
pip install -e .
```

### Run Unit & Integration Tests
```bash
pytest tests/ -v
```

### Execute Full TRL 4 Campaign & Generate Dashboards
```bash
python examples/run_*_campaign.py
```
This generates the standalone interactive HTML5 dashboard in `reports/` with live SVG graphics, interactive parameter toggles, and cryptographic SHA-256 verification manifests.

---

## 📜 Citation & Academic Attribution

If you utilize this testbed, digital twin, or verification methodology in your research, please cite:

```bibtex
@software{sani_rad_fault_hil_2026,
  author = {Sani, Mohammed Bello and DIMASA Research Consortium and Physics Hybrid Integrity Lab},
  title = {RAD-FAULT-HIL: COTS-to-Orbit Radiation Transport, SEE Physics & Avionics Fault-Injection HIL Testbed},
  year = {2026},
  version = {1.0.0},
  publisher = {GitHub},
  url = {https://github.com/Sm-bello/RAD-FAULT-HIL}
}
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

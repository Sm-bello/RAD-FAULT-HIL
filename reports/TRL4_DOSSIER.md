# RAD-FAULT-HIL: TRL 4 Space Radiation & Avionics Fault-Injection Qualification Dossier

- **Qualification Status**: `7/7 PILLARS PASSED (TRL 4 QUALIFIED)`
- **Cryptographic Campaign SHA-256**: `f4916661abe915894d786a30a999070a36cb5a4121fc9d79b086548883d6e426`
- **Orbital Regime**: `LEO` (Graded Enclosure: `2.5 mm Al + 0.6 mm Ta`, `5.99 mm Al-eq`)
- **Primary Avionics Target**: `STM32H7` (`STM32H743ZIT6 (ARM Cortex-M7)`)

---

## 1. Executive Summary: Unprotected COTS vs. RAD-FAULT-HIL Protected Architecture

| Qualification Metric | Unprotected Single-String COTS | RAD-FAULT-HIL Protected COTS (TMR + SEC-DED + Fast SEL Breaker) |
| :--- | :--- | :--- |
| **GNC Attitude Pointing Error (Post-SAA + SPE)** | `311.77 deg` (Catastrophic Loss of Pointing) | **`0.0065 deg`** (Nominal Fine Pointing) |
| **SRAM Corrupted Words (End of Campaign)** | `175 / 256 words corrupted` | **`0 / 256 words corrupted`** (`285` SEUs scrubbed, `9` MBUs reloaded) |
| **SEL Parasitic Latchup Energy Deposited** | `547.4 uJ` (> `250 uJ` Bond-Wire Melt / Burnout) | **`17.3 uJ`** (`2.65 us` Cutoff, **`14.4x`** Thermal Margin) |
| **Memory MTBUE (Mean Time Between Unrecoverable Errors)** | `0.00 h` (MTTF to first silent corruption) | **`2.9 h`** (**`81.2x`** over unscrubbed SEC-DED) |

---

## 2. Seven-Pillar TRL 4 Verification Scorecard

| Pillar ID | Verification Pillar | Measured Value | Qualification Threshold | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `P1_WEIBULL_BENDEL_TRANSPORT` | Weibull/Bendel Rate Integration & Graded Al+Ta Shielding Physics | **58.0 x (SAA flux spike)** | SAA proton spike >= 50x & Al+Ta TID attenuation > 2.5x | PASS |
| `P2_SECDED_SCRUBBING_MTBUE` | Extended Hamming SEC-DED (72,64) & Active Scrubbing MTBUE | **81.2 x (MTBUE gain)** | 100% 1-bit correction, 100% 2-bit detection, MTBUE gain > 50x | PASS |
| `P3_TMR_VOTER_SAA_RECOVERY` | TMR GNC Voter & Single-Frame State Resynchronization under SAA Storm | **0.0065 deg** | Final protected pointing error < 0.25 deg (Unprotected > 35 deg) | PASS |
| `P4_SEL_THERMAL_PROTECTION` | Fast High-Side SEL Current-Limiting Breaker Thermal Survival Margin | **14.44 x (thermal margin)** | Cutoff < 5 us & E_damage / E_thermal > 5.0x safety margin | PASS |
| `P5_CCSDS_HIL_PROTOCOL_INTEGRITY` | CCSDS / JTAG-UART Binary HIL Bridge & CRC-16-CCITT Frame Integrity | **100.0 %** | 100% valid TC/TM delivery + 100% CRC-16 corruption rejection + UDP socket | PASS |
| `P6_CRYPTOGRAPHIC_REPRODUCIBILITY` | Deterministic Campaign Replay & Cryptographic SHA-256 Audit Identity | **1.0 match** | Bit-exact SHA-256 match across independent campaign runs | PASS |
| `P7_REALTIME_FRAME_BUDGET` | Real-Time Avionics TMR Voter + SEC-DED Scrubber Minor-Frame Budget | **26.73 ms** | 99th-percentile step execution latency < 100.0 ms (10 Hz minor frame) | PASS |

---

## 3. Harmonized COTS Semiconductor SEE Rate Table

| Component ID | Device Part Number | SEU / day | MBU / day | SEFI / day | SEL / day | Annual TID Margin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `STM32H7` | STM32H743ZIT6 (ARM Cortex-M7) | 26.0699 | 0.51191 | 0.09742 | 0.022987 | 2.8x |
| `POLARFIRE_SOC` | MPFS250T (RISC-V PolarFire SoC) | 7.7715 | 0.07180 | 0.00929 | 0.000332 | 8.2x |
| `ZYNQ_ULTRASCALE` | XCZU9EG (Xilinx Zynq UltraScale+ MPSoC) | 432.2013 | 17.24778 | 4.46026 | 1.425778 | 5.0x |
| `MICRON_DDR4` | MT40A1G8 (Micron 8Gb DDR4 SDRAM) | 165.7387 | 6.21663 | 1.40725 | 0.389316 | 3.8x |
| `SIC_MOSFET` | C3M0065090D (Wolfspeed 900V SiC MOSFET) | 0.0044 | 0.00018 | 0.00138 | 0.028622 | 18.8x |

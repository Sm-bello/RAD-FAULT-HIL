"""RAD-FAULT-HIL: Open COTS-to-Orbit Space Radiation Transport, Single-Event
Effect (SEE/SEU/SEL) Cross-Section & Automated Avionics Fault-Injection
Qualification Testbed.
"""

from rad_fault_hil.fault_tolerant_avionics import (
    FastSelCurrentBreaker,
    HammingSecDed7264,
    MemoryScrubber,
    SecDedDecodeResult,
    SecDedStatus,
    SelTransientWaveform,
    TMRGncVoter,
    TmrVoteResult,
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
    OrbitParameters,
    OrbitRegime,
    OrbitalRadiationPoint,
    OrbitalRadiationSummary,
    SpaceRadiationTransportEngine,
)
from rad_fault_hil.reporting import generate_all_qualification_reports
from rad_fault_hil.see_physics import (
    BendelProtonParameters,
    COTS_COMPONENT_LIBRARY,
    ComponentSeeRateSummary,
    CotsComponentRadiationSpec,
    FaultMechanism,
    InjectedFaultEvent,
    SeeCrossSectionEngine,
    WeibullParameters,
)
from rad_fault_hil.verification import (
    CampaignQualificationBundle,
    CampaignStepRecord,
    PillarVerificationResult,
    run_qualification_campaign,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "OrbitRegime",
    "OrbitParameters",
    "ORBIT_PRESETS",
    "GradedShieldingConfig",
    "OrbitalRadiationPoint",
    "OrbitalRadiationSummary",
    "SpaceRadiationTransportEngine",
    "FaultMechanism",
    "WeibullParameters",
    "BendelProtonParameters",
    "CotsComponentRadiationSpec",
    "COTS_COMPONENT_LIBRARY",
    "ComponentSeeRateSummary",
    "InjectedFaultEvent",
    "SeeCrossSectionEngine",
    "SecDedStatus",
    "SecDedDecodeResult",
    "HammingSecDed7264",
    "MemoryScrubber",
    "TmrVoteResult",
    "TMRGncVoter",
    "SelTransientWaveform",
    "FastSelCurrentBreaker",
    "HilTransportMode",
    "CcsdsFaultCode",
    "compute_crc16_ccitt",
    "CcsdsFaultInjectionCommand",
    "CcsdsAvionicsTelemetryFrame",
    "FaultInjectionHilBridge",
    "CampaignStepRecord",
    "PillarVerificationResult",
    "CampaignQualificationBundle",
    "run_qualification_campaign",
    "generate_all_qualification_reports",
]

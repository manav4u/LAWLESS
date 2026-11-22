"""Objective interaction APIs."""

from lawless.objectives.cores import (
    CoreCollectionResult,
    CoreContactResult,
    collect_cores,
    detect_core_contacts,
)

from lawless.objectives.extraction import (
    ExtractionApplicationResult,
    ExtractionDetectionResult,
    apply_extraction,
    detect_extraction,
)

__all__ = [
    "CoreCollectionResult",
    "CoreContactResult",
    "ExtractionApplicationResult",
    "ExtractionDetectionResult",
    "apply_extraction",
    "collect_cores",
    "detect_core_contacts",
    "detect_extraction",
]

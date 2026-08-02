"""Toll pricing rules by vehicle class."""

from dataclasses import dataclass

# COCO class ids from the YOLO vehicle detector mapped to toll categories.
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Toll fee per vehicle category (INR).
TOLL_RATES = {
    "motorcycle": 30,
    "car": 65,
    "bus": 225,
    "truck": 225,
}


@dataclass
class TollDecision:
    vehicle_type: str
    plate_text: str | None
    toll_amount: int
    allowed: bool
    reason: str


def price_toll(vehicle_type: str) -> int:
    return TOLL_RATES.get(vehicle_type, TOLL_RATES["car"])


def decide(vehicle_type: str, plate_text: str | None, blacklist: set[str]) -> TollDecision:
    """Combine detection results into a gate decision."""
    toll = price_toll(vehicle_type)
    if plate_text is None:
        return TollDecision(vehicle_type, None, toll, False, "Plate unreadable — manual check")
    if plate_text in blacklist:
        return TollDecision(vehicle_type, plate_text, toll, False, "Vehicle is blacklisted")
    return TollDecision(vehicle_type, plate_text, toll, True, "OK")

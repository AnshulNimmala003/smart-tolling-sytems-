"""Deep-learning detection: vehicles (YOLOv8 COCO) + license plates (fine-tuned YOLOv8)."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .pricing import COCO_VEHICLE_CLASSES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Fine-tuned license-plate detector published on Hugging Face.
PLATE_MODEL_REPO = "Koushim/yolov8-license-plate-detection"
PLATE_MODEL_FILE = "best.pt"


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    confidence: float
    plate_box: tuple[int, int, int, int] | None = None
    plate_text: str | None = None
    plate_confidence: float = 0.0
    extras: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _vehicle_model():
    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    return YOLO(str(MODELS_DIR / "yolov8n.pt"))


@lru_cache(maxsize=1)
def _plate_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    path = hf_hub_download(
        PLATE_MODEL_REPO, PLATE_MODEL_FILE, local_dir=str(MODELS_DIR)
    )
    return YOLO(path)


def detect_vehicles(img_bgr: np.ndarray, conf: float = 0.35) -> list[Detection]:
    """Find vehicles in the frame and label them with their toll category."""
    results = _vehicle_model().predict(
        img_bgr, conf=conf, classes=list(COCO_VEHICLE_CLASSES), verbose=False
    )[0]
    detections = []
    for b in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        detections.append(
            Detection(
                box=(x1, y1, x2, y2),
                label=COCO_VEHICLE_CLASSES[int(b.cls)],
                confidence=float(b.conf),
            )
        )
    return detections


def detect_plates(img_bgr: np.ndarray, conf: float = 0.30) -> list[tuple[tuple[int, int, int, int], float]]:
    """Find license plates anywhere in the frame."""
    results = _plate_model().predict(img_bgr, conf=conf, verbose=False)[0]
    return [
        (tuple(int(v) for v in b.xyxy[0]), float(b.conf))
        for b in results.boxes
    ]

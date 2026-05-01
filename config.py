"""
Central configuration for the Safety Gear Compliance system.

All class IDs, colour palettes, queue sizes, and default thresholds live here
so every module imports from one place.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Class IDs ──────────────────────────────────────────────────────────────────
# Multi-Model-v2 runs five single-class detectors. ``CLASS_NO_HELMET`` is
# kept as an internal semantic class for optional/future negative-helmet
# evidence, but it is not part of ``TARGET_CLASS_IDS`` because this branch
# does not have a separate no_helmet model.
CLASS_MOTORCYCLE = 0
CLASS_RIDER = 1
CLASS_HELMET = 2
CLASS_NO_HELMET = 3
CLASS_FOOTWEAR = 4
CLASS_IMPROPER_FOOTWEAR = 5

TARGET_CLASS_IDS: Tuple[int, ...] = (
    CLASS_MOTORCYCLE,
    CLASS_RIDER,
    CLASS_HELMET,
    CLASS_FOOTWEAR,
    CLASS_IMPROPER_FOOTWEAR,
)

CLASS_NAMES: Dict[int, str] = {
    CLASS_MOTORCYCLE: "Motorcycle",
    CLASS_RIDER: "Rider",
    CLASS_HELMET: "Helmet",
    CLASS_NO_HELMET: "No Helmet",
    CLASS_FOOTWEAR: "Footwear",
    CLASS_IMPROPER_FOOTWEAR: "Improper Footwear",
}

# BGR colours for bounding boxes
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    CLASS_MOTORCYCLE: (255, 255, 0),         # Cyan
    CLASS_RIDER: (255, 0, 0),                # Blue
    CLASS_HELMET: (0, 255, 0),               # Green
    CLASS_NO_HELMET: (0, 0, 255),            # Red
    CLASS_FOOTWEAR: (0, 255, 0),             # Green
    CLASS_IMPROPER_FOOTWEAR: (0, 0, 255),    # Red
}

# Compliance overlay colours
COLOR_COMPLIANT_BGR: Tuple[int, int, int] = (0, 220, 0)       # Green
COLOR_NON_COMPLIANT_BGR: Tuple[int, int, int] = (0, 0, 255)   # Red
COLOR_OVERLOAD_BGR: Tuple[int, int, int] = (0, 140, 255)      # Orange

# ── Pipeline queue capacities ──────────────────────────────────────────────────
FRAME_QUEUE_SIZE = 4
DETECTION_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2

# ── Detection / tracking defaults ──────────────────────────────────────────────
@dataclass
class DetectionConfig:
    """Mutable detection settings shared across threads."""
    conf: float = 0.20
    iou: float = 0.30
    imgsz: int = 480
    inference_batch_size: int = 4
    inference_stride: int = 2
    use_fp16: bool = False
    device: str = "auto"
    rider_moto_ioa_thresh: float = 0.05
    gear_rider_ioa_thresh: float = 0.10
    occlusion_conf_thresh: float = 0.10
    max_riders_per_motorcycle: int = 2
    # Minimum confidence required on a `no_helmet` detection before flagging
    # the rider as non-compliant.  Helps reject low-quality false positives.
    no_helmet_min_conf: float = 0.40
    tracker_key: str = "botsort"
    # User toggle for the cross-frame tracker. When False, the inference
    # engine falls back to plain ``model.predict`` (no IDs, but allows
    # batching for higher FPS).
    tracker_enabled: bool = True


# ── Per-class confidence tuning ────────────────────────────────────────────────
# Per-class confidence tuning for the five single-class models.
CLASS_CONF_FACTOR: Dict[int, float] = {
    CLASS_MOTORCYCLE:        1.00,
    CLASS_RIDER:             1.00,
    CLASS_HELMET:            0.80,   # helmets get small at distance
    CLASS_FOOTWEAR:          0.70,
    CLASS_IMPROPER_FOOTWEAR: 0.60,   # the big recall lever
}
CLASS_CONF_MIN: Dict[int, float] = {
    CLASS_MOTORCYCLE:        0.20,
    CLASS_RIDER:             0.20,
    CLASS_HELMET:            0.15,
    CLASS_FOOTWEAR:          0.12,
    CLASS_IMPROPER_FOOTWEAR: 0.10,
}

# ── Supported file extensions ──────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXTENSIONS = {".pt", ".engine", ".onnx"}

# ── Default model file names ─────────────────────────────────────────────────
# Each class id maps to its own single-class model folder.
DEFAULT_MODEL_FILES: Dict[int, str] = {
    CLASS_MOTORCYCLE: "motorcycle/motorcycle.pt",
    CLASS_RIDER: "rider/rider.pt",
    CLASS_HELMET: "helmet/helmet.pt",
    CLASS_FOOTWEAR: "footwear/footwear.pt",
    CLASS_IMPROPER_FOOTWEAR: "improper_footwear/improper_footwear.pt",
}

# ── Models root directory ──────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")

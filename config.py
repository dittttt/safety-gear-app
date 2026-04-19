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
CLASS_MOTORCYCLE = 0
CLASS_RIDER = 1
CLASS_HELMET = 2
CLASS_FOOTWEAR = 3
CLASS_IMPROPER_FOOTWEAR = 4

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
    CLASS_FOOTWEAR: "Footwear",
    CLASS_IMPROPER_FOOTWEAR: "Improper Footwear",
}

# BGR colours for bounding boxes
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    CLASS_MOTORCYCLE: (128, 128, 128),    # Grey
    CLASS_RIDER: (255, 160, 16),          # Blue-ish
    CLASS_HELMET: (0, 200, 0),            # Green
    CLASS_FOOTWEAR: (0, 200, 0),          # Green
    CLASS_IMPROPER_FOOTWEAR: (0, 0, 255), # Red
}

# Compliance overlay colours
COLOR_COMPLIANT_BGR: Tuple[int, int, int] = (0, 220, 0)       # Green
COLOR_NON_COMPLIANT_BGR: Tuple[int, int, int] = (0, 0, 255)   # Red
COLOR_OVERLOAD_BGR: Tuple[int, int, int] = (0, 140, 255)      # Orange

# ── Pipeline queue capacities ──────────────────────────────────────────────────
FRAME_QUEUE_SIZE = 64
DETECTION_QUEUE_SIZE = 32
DISPLAY_QUEUE_SIZE = 16

# ── Detection / tracking defaults ──────────────────────────────────────────────
@dataclass
class DetectionConfig:
    """Mutable detection settings shared across threads."""
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 256
    inference_batch_size: int = 4
    inference_stride: int = 6
    use_fp16: bool = False
    device: str = "auto"
    rider_moto_ioa_thresh: float = 0.05
    gear_rider_ioa_thresh: float = 0.10
    occlusion_conf_thresh: float = 0.15
    max_riders_per_motorcycle: int = 2
    tracker_key: str = "botsort"

# ── Supported file extensions ──────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
MODEL_EXTENSIONS = {".pt", ".engine", ".onnx"}

# ── Default model file names (looked up in models/ relative to repo root) ──────
DEFAULT_MODEL_FILES: Dict[int, str] = {
    CLASS_MOTORCYCLE: "motorcycle.pt",
    CLASS_RIDER: "rider.pt",
    CLASS_HELMET: "helmet.pt",
    CLASS_FOOTWEAR: "footwear.pt",
    CLASS_IMPROPER_FOOTWEAR: "improper_footwear.pt",
}

# ── Models root directory ──────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")

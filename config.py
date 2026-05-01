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
# The unified ``best.pt`` detector emits SEVEN classes (indices FROZEN to
# match the trained weights — see utils/unified_constants.MASTER_CLASSES).
# The seventh class — ``tricycle`` (id 6) — exists purely so the model can
# disambiguate motorcycles from tricycles during training and is filtered
# out of the downstream pipeline; it never appears in the UI, stats, or
# violations.  Compliance logic operates on the six user-facing classes
# below.
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
    CLASS_NO_HELMET,
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
    # Minimum confidence required on a `no_helmet` detection before flagging
    # the rider as non-compliant.  Helps reject low-quality false positives.
    no_helmet_min_conf: float = 0.40
    tracker_key: str = "botsort"
    # User toggle for the cross-frame tracker. When False, the inference
    # engine falls back to plain ``model.predict`` (no IDs, but allows
    # batching for higher FPS).
    tracker_enabled: bool = True


# ── Per-class confidence tuning ────────────────────────────────────────────────
# The unified detector is a generalist and tends to be under-confident on
# small / fine-grained classes (footwear, improper_footwear, helmets at
# distance). We therefore POST-FILTER detections per class instead of
# raising the global slider, which would suppress good far-range hits.
#
# Pipeline:
#   1. Run inference at ``conf = min(global_conf, min(_CLASS_CONF_MIN.values()))``
#      so the model returns everything that *could* be relevant.
#   2. After extraction, drop any detection whose score is below
#      ``max(_CLASS_CONF_MIN[cid], global_conf * _CLASS_CONF_FACTOR[cid])``.
#
# Tune these knobs per-class without touching the inference loop.
CLASS_CONF_FACTOR: Dict[int, float] = {
    CLASS_MOTORCYCLE:        1.00,
    CLASS_RIDER:             1.00,
    CLASS_HELMET:            0.80,   # helmets get small at distance
    CLASS_NO_HELMET:         1.20,   # bias against false positives
    CLASS_FOOTWEAR:          0.70,
    CLASS_IMPROPER_FOOTWEAR: 0.60,   # the big recall lever
}
CLASS_CONF_MIN: Dict[int, float] = {
    CLASS_MOTORCYCLE:        0.20,
    CLASS_RIDER:             0.20,
    CLASS_HELMET:            0.15,
    CLASS_NO_HELMET:         0.30,
    CLASS_FOOTWEAR:          0.12,
    CLASS_IMPROPER_FOOTWEAR: 0.10,
}

# ── Supported file extensions ──────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXTENSIONS = {".pt", ".engine", ".onnx"}

# ── Default model file names ─────────────────────────────────────────────────
# The system now uses ONE unified detector that emits all 7 classes.
# Every class id maps to the same physical file (``models/unified/best.pt``);
# the inference engine loads it exactly once.
UNIFIED_MODEL_PATH: str = os.path.join(_ROOT, "models", "unified", "best.pt")
FOOTWEAR_MODEL_PATH: str = os.path.join(_ROOT, "models", "footwear", "footwear.pt")
DEFAULT_MODEL_FILES: Dict[int, str] = {cid: "unified/best.pt" for cid in TARGET_CLASS_IDS}

# ── Models root directory ──────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")

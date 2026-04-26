"""
Unified detector class taxonomy — single source of truth.

The model file ``models/unified/best.pt`` is a YOLO11s detector trained on
seven classes.  These indices are FROZEN; do not reorder them or the
trained weights will misclassify everything.

Validation performance (final ``best.pt``):

    motorcycle         mAP50 87.5%  mAP50-95 66.8%
    rider              mAP50 77.7%  mAP50-95 57.7%
    helmet             mAP50 92.9%  mAP50-95 71.0%
    no_helmet          mAP50 87.3%  mAP50-95 61.4%
    footwear           mAP50 91.0%  mAP50-95 69.6%
    improper_footwear  mAP50 88.8%  mAP50-95 67.7%
    tricycle           mAP50 99.2%  mAP50-95 89.9%

The ``rider`` class is the noisiest due to inter-dataset annotation
variance — treat its detections as slightly less reliable than the rest.
"""

from typing import Dict, FrozenSet


# ── Frozen class index map ────────────────────────────────────────────────────
MASTER_CLASSES: Dict[int, str] = {
    0: "motorcycle",
    1: "rider",
    2: "helmet",
    3: "no_helmet",
    4: "footwear",
    5: "improper_footwear",
    6: "tricycle",
}

# Reverse map (name -> id).
MASTER_CLASS_IDS: Dict[str, int] = {v: k for k, v in MASTER_CLASSES.items()}


# ── Semantic groupings ────────────────────────────────────────────────────────
# Parent vehicles a rider can ride on.  Each parent has its own overload
# threshold (LTO MC No. 2014-001).
PARENT_VEHICLE_CLASSES: FrozenSet[int] = frozenset({
    MASTER_CLASS_IDS["motorcycle"],
    MASTER_CLASS_IDS["tricycle"],
})

# Maximum riders allowed per parent vehicle before flagging an overload.
MAX_RIDERS_PER_VEHICLE: Dict[int, int] = {
    MASTER_CLASS_IDS["motorcycle"]: 2,
    MASTER_CLASS_IDS["tricycle"]: 4,   # driver + 3 sidecar passengers
}

# Gear classes that attach to a rider.
RIDER_CLASS: int = MASTER_CLASS_IDS["rider"]
HELMET_POSITIVE_CLASS: int = MASTER_CLASS_IDS["helmet"]
HELMET_NEGATIVE_CLASS: int = MASTER_CLASS_IDS["no_helmet"]
FOOTWEAR_POSITIVE_CLASS: int = MASTER_CLASS_IDS["footwear"]
FOOTWEAR_NEGATIVE_CLASS: int = MASTER_CLASS_IDS["improper_footwear"]


# ── Default unified model location ────────────────────────────────────────────
UNIFIED_MODEL_REL_PATH: str = "models/unified/best.pt"

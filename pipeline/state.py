"""
Shared pipeline state: queues, events, and configuration accessible by all threads.

Every thread receives a single ``PipelineState`` instance.  All public
methods are thread-safe (protected by locks or atomic events).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    TARGET_CLASS_IDS,
    CLASS_COLORS_BGR,
    FRAME_QUEUE_SIZE,
    DETECTION_QUEUE_SIZE,
    DISPLAY_QUEUE_SIZE,
    DetectionConfig,
)


# ── Data packets passed between threads ────────────────────────────────────────

@dataclass
class FramePacket:
    """Frame Grabber → Inference Engine."""
    index: int
    frame: np.ndarray
    timestamp_ms: float = 0.0


@dataclass
class Detection:
    """Single bounding-box detection."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    track_id: Optional[int] = None


@dataclass
class DetectionPacket:
    """Inference Engine → Tracker / Logic."""
    index: int
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    timestamp_ms: float = 0.0


@dataclass
class DisplayPacket:
    """Tracker / Logic → GUI."""
    index: int
    annotated_frame: np.ndarray
    raw_frame: np.ndarray
    stats: dict = field(default_factory=dict)
    timestamp_ms: float = 0.0


# ── Shared pipeline state ──────────────────────────────────────────────────────

class PipelineState:
    """Thread-safe shared state for the producer-consumer pipeline."""

    def __init__(self) -> None:
        # ── Queues ──
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.detection_queue: queue.Queue[DetectionPacket] = queue.Queue(maxsize=DETECTION_QUEUE_SIZE)
        self.display_queue: queue.Queue[DisplayPacket] = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)

        # ── Control events ──
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()       # SET = paused
        self.seek_event = threading.Event()

        # ── Seek ──
        self._seek_lock = threading.Lock()
        self._seek_target: int = -1

        # ── Settings (lock-protected) ──
        self._lock = threading.Lock()
        self.detection_config = DetectionConfig()
        self.playback_rate: float = 1.0
        self.overlay_enabled: bool = True
        self.enabled_models: Dict[int, bool] = {cid: True for cid in TARGET_CLASS_IDS}
        self.class_colors_bgr: Dict[int, Tuple[int, int, int]] = dict(CLASS_COLORS_BGR)
        self.reset_tracker_flag: bool = False

        # ── Model paths ──
        self.model_paths: Dict[int, Optional[str]] = {cid: None for cid in TARGET_CLASS_IDS}

        # ── Video metadata (written by grabber, read by GUI) ──
        self.video_fps: float = 0.0
        self.total_frames: int = 0
        self.video_path: Optional[str] = None

    # ── Queue helpers ──────────────────────────────────────────────────────────

    def flush_queues(self) -> None:
        """Drain all three queues (non-blocking)."""
        for q in (self.frame_queue, self.detection_queue, self.display_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def request_seek(self, frame_index: int) -> None:
        with self._seek_lock:
            self._seek_target = max(0, int(frame_index))
            self.seek_event.set()

    def consume_seek(self) -> Optional[int]:
        """Return requested frame index and clear the flag, or ``None``."""
        if not self.seek_event.is_set():
            return None
        with self._seek_lock:
            target = self._seek_target
            self._seek_target = -1
            self.seek_event.clear()
        return target

    # ── Safe config access ─────────────────────────────────────────────────────

    def get_playback_rate(self) -> float:
        with self._lock:
            return self.playback_rate

    def set_playback_rate(self, rate: float) -> None:
        with self._lock:
            self.playback_rate = max(0.1, min(8.0, float(rate)))

    def get_conf(self) -> float:
        with self._lock:
            return self.detection_config.conf

    def set_conf(self, val: float) -> None:
        with self._lock:
            self.detection_config.conf = max(0.0, min(1.0, float(val)))

    def get_iou(self) -> float:
        with self._lock:
            return self.detection_config.iou

    def set_iou(self, val: float) -> None:
        with self._lock:
            self.detection_config.iou = max(0.0, min(1.0, float(val)))

    def get_imgsz(self) -> int:
        with self._lock:
            return int(self.detection_config.imgsz)

    def set_imgsz(self, val: int) -> None:
        with self._lock:
            size = int(val)
            self.detection_config.imgsz = max(320, min(1280, size))

    def get_inference_stride(self) -> int:
        with self._lock:
            return max(1, int(self.detection_config.inference_stride))

    def set_inference_stride(self, val: int) -> None:
        with self._lock:
            self.detection_config.inference_stride = max(1, min(8, int(val)))

    def use_fp16(self) -> bool:
        with self._lock:
            return bool(self.detection_config.use_fp16)

    def set_use_fp16(self, enabled: bool) -> None:
        with self._lock:
            self.detection_config.use_fp16 = bool(enabled)

    def get_device(self) -> str:
        with self._lock:
            dev = str(self.detection_config.device or "auto").strip().lower()
            return dev if dev in {"auto", "cpu", "cuda"} else "auto"

    def set_device(self, device: str) -> None:
        with self._lock:
            dev = str(device or "auto").strip().lower()
            self.detection_config.device = dev if dev in {"auto", "cpu", "cuda"} else "auto"

    def is_model_enabled(self, class_id: int) -> bool:
        with self._lock:
            return self.enabled_models.get(class_id, False)

    def set_model_enabled(self, class_id: int, enabled: bool) -> None:
        with self._lock:
            self.enabled_models[class_id] = bool(enabled)

    def set_overlay_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.overlay_enabled = bool(enabled)

    def is_overlay_enabled(self) -> bool:
        with self._lock:
            return self.overlay_enabled

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        with self._lock:
            return self.class_colors_bgr.get(class_id, (255, 255, 255))

    def set_class_color(self, class_id: int, color_bgr: Tuple[int, int, int]) -> None:
        b, g, r = color_bgr
        with self._lock:
            self.class_colors_bgr[class_id] = (
                int(max(0, min(255, b))),
                int(max(0, min(255, g))),
                int(max(0, min(255, r))),
            )

    def get_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        with self._lock:
            return dict(self.class_colors_bgr)

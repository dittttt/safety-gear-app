"""
Thread 2 – Inference Engine
===========================
Consumes ``FramePacket`` items from the frame queue in batches, runs YOLO
detection for every enabled model, and pushes per-frame ``DetectionPacket``
objects into the detection queue.

* Batched detection defaults to 32 frames (configurable in shared state).
* Tracking is handled by the downstream tracking thread.
* Supports ``.pt`` (PyTorch), ``.engine`` (TensorRT), and ``.onnx`` weights.
"""

import os
import sys
import time
import queue
import logging
import contextlib
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRA_LOGGER
import torch

from config import TARGET_CLASS_IDS, CLASS_NAMES
from pipeline.state import PipelineState, Detection, DetectionPacket
from utils.runtime_check import detect as detect_runtime


# The unified detector is well-calibrated across all 7 classes; per-class
# confidence overrides are no longer required.


@contextlib.contextmanager
def _silence_clevel_stderr():
    """Redirect the OS-level stderr fd to NUL during TRT / CUDA model load.

    TensorRT (nvinfer.dll) prints [TRT][W] and [TRT][I] messages by writing
    directly to file descriptor 2, bypassing Python's sys.stderr.  We must
    dup/redirect at the OS level to silence them.
    """
    devnull = "NUL" if sys.platform == "win32" else "/dev/null"
    try:
        sys.stderr.flush()
        _devnull_fd = os.open(devnull, os.O_WRONLY)
        _old_fd = os.dup(2)
        os.dup2(_devnull_fd, 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(_old_fd, 2)
            os.close(_old_fd)
            os.close(_devnull_fd)
    except OSError:
        # If the dup trick fails (e.g., restricted env), just run normally.
        yield


class InferenceThread(QtCore.QThread):
    """Consumer of frames → producer of detections.

    The pipeline now runs a single unified YOLO11s detector that emits all
    seven safety-compliance classes in one forward pass.  The per-class
    cascade has been retired — ``self._models`` always holds zero or one
    entry keyed by ``-1`` (unified model sentinel).
    """

    UNIFIED_KEY: int = -1   # sentinel key for the single unified model

    status = QtCore.pyqtSignal(str)
    fps_update = QtCore.pyqtSignal(float)

    def __init__(self, state: PipelineState, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._models: Dict[int, YOLO] = {}
        self._model_paths: Dict[int, str] = {}
        self._device: str = "cpu"
        self._fp16: bool = False
        self._rt = detect_runtime()
        # Suppress noisy backend setup chatter (especially TensorRT/OpenVINO)
        ULTRA_LOGGER.setLevel(logging.WARNING)

    def _resolve_device(self) -> str:
        device = self._state.get_device()
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _path_exists(path: str) -> bool:
        return bool(path) and (os.path.isfile(path) or os.path.isdir(path))

    @staticmethod
    def _can_move_to_device(path: str) -> bool:
        return str(path).lower().endswith(".pt")

    @staticmethod
    def _is_openvino_path(path: str) -> bool:
        p = str(path or "").lower()
        return os.path.isdir(path) and (
            p.endswith("_openvino_model")
            or p.endswith("_ov_model")
            or os.path.isfile(os.path.join(path, "metadata.yaml"))
        )

    def _fallback_model_path(self, path: str) -> Optional[str]:
        """Return a compatible fallback artifact next to *path* if available."""
        if not path:
            return None
        if os.path.isdir(path):
            base_dir = os.path.dirname(path)
            stem = os.path.basename(base_dir)
        else:
            base_dir = os.path.dirname(path)
            stem = os.path.splitext(os.path.basename(path))[0]

        candidates = [
            os.path.join(base_dir, f"{stem}.pt"),
            os.path.join(base_dir, f"{stem}.onnx"),
            os.path.join(base_dir, f"{stem}_openvino_model"),
            os.path.join(base_dir, f"{stem}_OV_model"),
        ]
        cur = os.path.normpath(path)
        for cand in candidates:
            if os.path.normpath(cand) == cur:
                continue
            if self._path_exists(cand):
                return cand
        return None

    def _runtime_supports_path(self, path: str) -> bool:
        p = str(path or "").lower()
        if p.endswith(".engine"):
            return bool(self._rt.has_tensorrt)
        if self._is_openvino_path(path):
            return bool(self._rt.has_openvino)
        return True

    def _create_model(self, path: str) -> YOLO:
        """Instantiate a YOLO model and apply runtime device/precision policy."""
        # TensorRT engines write [TRT][W/I] lines directly to the C-level fd 2.
        # Suppress those during load; they are benign informational messages.
        ctx = _silence_clevel_stderr() if path.endswith(".engine") else contextlib.nullcontext()
        with ctx:
            model = YOLO(path, task="detect", verbose=False)
        if self._can_move_to_device(path):
            model.to(self._device)
            if self._fp16:
                try:
                    model.model.half()
                except Exception:
                    pass
        return model

    def _try_rebind_fallback_model(self) -> bool:
        """Try to rebind the unified model to a sibling artifact."""
        cur = self._model_paths.get(self.UNIFIED_KEY)
        fb = self._fallback_model_path(cur or "")
        if not fb or not self._path_exists(fb):
            return False
        try:
            model = self._create_model(fb)
            self._models[self.UNIFIED_KEY] = model
            self._model_paths[self.UNIFIED_KEY] = fb
            for cid in TARGET_CLASS_IDS:
                self._state.model_paths[cid] = fb
            self.status.emit(
                f"Unified detector: fallback bound -> {os.path.basename(fb)}"
            )
            return True
        except Exception:
            return False

    # ── Model management ───────────────────────────────────────────────────────

    def _resolve_unified_path(self) -> Optional[str]:
        """Pick the unified model artifact to load.

        Preference order:
          1. Whatever the GUI / state currently has registered (every cid in
             ``state.model_paths`` should resolve to the same artifact).
          2. The configured default at ``models/unified/best.pt`` plus any
             optimised siblings (.engine / .onnx / OpenVINO IR).
        """
        candidates: List[str] = []
        # 1) State-registered paths (de-duplicated, order-preserving).
        seen = set()
        for cid in TARGET_CLASS_IDS:
            p = self._state.model_paths.get(cid)
            if p and p not in seen and self._path_exists(p):
                seen.add(p)
                candidates.append(p)
        # 2) Filesystem fallbacks under models/unified/.
        from config import UNIFIED_MODEL_PATH  # local import to avoid cycles
        unified_dir = os.path.dirname(UNIFIED_MODEL_PATH)
        for cand in (
            os.path.join(unified_dir, "best.engine"),
            os.path.join(unified_dir, "best_openvino_model"),
            UNIFIED_MODEL_PATH,
            os.path.join(unified_dir, "best.onnx"),
        ):
            if cand not in seen and self._path_exists(cand):
                seen.add(cand)
                candidates.append(cand)
        # Pick the highest-priority candidate the runtime can support.
        for path in candidates:
            if self._runtime_supports_path(path):
                return path
        return candidates[0] if candidates else None

    def _load_models(self) -> None:
        """(Re)load the single unified detector."""
        self._models.clear()
        self._model_paths.clear()
        self._rt = detect_runtime()
        self._device = self._resolve_device()
        self._fp16 = self._state.use_fp16() and self._device.startswith("cuda")

        path = self._resolve_unified_path()
        if not path or not self._path_exists(path):
            self.status.emit(
                "No unified detector found — drop best.pt into models/unified/"
            )
            return
        if not self._runtime_supports_path(path):
            fb = self._fallback_model_path(path)
            if fb and self._path_exists(fb):
                self.status.emit(
                    f"Unified detector: runtime cannot load "
                    f"{os.path.basename(path)}, using {os.path.basename(fb)}"
                )
                path = fb

        tag = os.path.basename(path) if not os.path.isdir(path) else os.path.basename(path.rstrip(os.sep))
        self.status.emit(f"Loading unified detector: {tag}")
        try:
            model = self._create_model(path)
            self._models[self.UNIFIED_KEY] = model
            self._model_paths[self.UNIFIED_KEY] = path
            # Keep state.model_paths consistent so the GUI sees the same file
            # regardless of which class checkbox the user inspects.
            for cid in TARGET_CLASS_IDS:
                self._state.model_paths[cid] = path
        except Exception as exc:
            self.status.emit(f"Failed to load unified detector ({tag}): {exc}")

        self.status.emit(
            f"Inference backend: {self._device}"
            f"{' + FP16' if self._fp16 else ''}"
        )

    def _predict_kwargs(self, conf: float, iou: float, imgsz: int) -> dict:
        """Backend-aware kwargs for the unified detector."""
        kwargs = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        path = str(self._model_paths.get(self.UNIFIED_KEY, "")).lower()
        if path.endswith(".pt"):
            kwargs["device"] = self._device
            kwargs["half"] = self._fp16
        return kwargs

    def _predict_unified(self, frames: List[np.ndarray], kwargs: dict):
        """Run the unified model once on a list of frames."""
        model = self._models.get(self.UNIFIED_KEY)
        if model is None:
            return [None] * len(frames)
        path = self._model_paths.get(self.UNIFIED_KEY, "")
        allow_batch = not self._is_openvino_path(path)
        if allow_batch:
            try:
                results = model.predict(frames, **kwargs)
                if isinstance(results, (list, tuple)) and len(results) == len(frames):
                    return list(results)
            except Exception:
                pass
        out: List = []
        for frame in frames:
            try:
                r = model.predict(frame, **kwargs)
                out.append(r[0] if isinstance(r, (list, tuple)) and r else r)
            except Exception:
                out.append(None)
        return out

    # ── Detection extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_detections(result, allowed_classes: Optional[set] = None) -> List[Detection]:
        """Convert one Ultralytics result into a list of typed Detections.

        ``allowed_classes`` filters by class id (after extraction); pass
        ``None`` to keep every class the model emitted.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        ids = getattr(boxes, "id", None)
        if xyxy is None or cls is None:
            return []

        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        cls_np = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
        conf_np = (
            conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu")
            else (np.asarray(conf) if conf is not None else None)
        )
        id_np = None
        if ids is not None:
            try:
                id_np = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            except Exception:
                id_np = None

        dets: List[Detection] = []
        for i in range(len(xyxy_np)):
            cid = int(cls_np[i])
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy_np[i])
            score = float(conf_np[i]) if conf_np is not None else 0.0
            tid = int(id_np[i]) if (id_np is not None and i < len(id_np)) else None
            dets.append(Detection(x1, y1, x2, y2, cid, score, tid))
        return dets

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        state = self._state
        self._load_models()

        if not self._models:
            self.status.emit("No models loaded — overlay will be disabled")

        frame_count = 0
        t0 = time.perf_counter()
        step = 0
        cached_dets: List[Detection] = []

        while not state.stop_event.is_set():
            # Hot-reload models if flagged (e.g. after optimised conversion)
            if state.reload_models_flag:
                state.reload_models_flag = False
                self._load_models()

            # Pull first frame for this cycle.
            try:
                first_packet = state.frame_queue.get(timeout=0.05)
            except Exception:
                continue

            if state.stop_event.is_set():
                break

            # Build a frame batch (first + any immediately available packets)
            batch_size = state.get_inference_batch_size()
            packets = [first_packet]
            while len(packets) < batch_size:
                try:
                    packets.append(state.frame_queue.get_nowait())
                except queue.Empty:
                    break

            conf = state.get_conf()
            iou = state.get_iou()
            imgsz = state.get_imgsz()
            stride = state.get_inference_stride()
            overlay = state.is_overlay_enabled()

            infer_idxs = [
                i for i in range(len(packets))
                if (stride <= 1 or ((step + i) % stride == 0))
            ]
            infer_set = set(infer_idxs)
            dets_by_idx: Dict[int, List[Detection]] = {i: [] for i in infer_idxs}

            if overlay and self._models and infer_idxs:
                infer_frames = [packets[i].frame for i in infer_idxs]

                # ONE forward pass through the unified detector.  Each result
                # is then split into per-class buckets according to the user's
                # enabled-models settings (so toggles in the GUI still gate
                # classes; we just don't run the model multiple times).
                kwargs = self._predict_kwargs(conf, iou, imgsz)
                results = self._predict_unified(infer_frames, kwargs)

                allowed = {
                    cid for cid in TARGET_CLASS_IDS
                    if state.is_model_enabled(cid)
                }
                for local_i, result in enumerate(results):
                    if local_i >= len(infer_idxs) or result is None:
                        continue
                    pkt_i = infer_idxs[local_i]
                    extracted = self._extract_detections(result, allowed_classes=allowed)
                    dets_by_idx[pkt_i].extend(extracted)

            for i, packet in enumerate(packets):
                inferred_this_frame = i in infer_set and overlay and bool(self._models)
                if inferred_this_frame:
                    all_dets = dets_by_idx.get(i, [])
                    cached_dets = all_dets
                else:
                    all_dets = list(cached_dets)

                det_packet = DetectionPacket(
                    index=packet.index,
                    frame=packet.frame,
                    detections=all_dets,
                    timestamp_ms=packet.timestamp_ms,
                    detections_fresh=inferred_this_frame,
                )
                state.put_safe(state.detection_queue, det_packet)

            # FPS reporting
            frame_count += len(packets)
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self.fps_update.emit(frame_count / (now - t0))
                frame_count = 0
                t0 = now
            step += len(packets)

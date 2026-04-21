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


# Per-class confidence floor.  Single-class fine-tuned models (especially
# `improper_footwear`) tend to be under-confident; we let them recall a bit
# more aggressively than the global slider value.
_CLASS_CONF_FACTOR: Dict[int, float] = {
    4: 0.6,   # improper_footwear → run at 60% of the global confidence
}
_CLASS_CONF_MIN: Dict[int, float] = {
    4: 0.10,  # never run improper_footwear below 0.10
}


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
    """Consumer of frames → producer of detections."""

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

    def _try_rebind_fallback_model(self, cid: int, reason: str = "") -> bool:
        cur = self._model_paths.get(cid)
        fb = self._fallback_model_path(cur or "")
        if not fb or not self._path_exists(fb):
            return False
        try:
            model = self._create_model(fb)
            self._models[cid] = model
            self._model_paths[cid] = fb
            self._state.model_paths[cid] = fb
            name = CLASS_NAMES.get(cid, str(cid))
            self.status.emit(
                f"{name}: fallback model bound -> {os.path.basename(fb)}"
                + (f" ({reason})" if reason else "")
            )
            return True
        except Exception:
            return False

    # ── Model management ───────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """(Re)load models from current ``state.model_paths``.

        The GUI / model-registry is responsible for choosing the correct
        variant (PyTorch, TensorRT, OpenVINO …).  This method simply loads
        whatever path is stored in ``state.model_paths``.
        """
        self._models.clear()
        self._model_paths.clear()
        self._rt = detect_runtime()
        self._device = self._resolve_device()
        self._fp16 = self._state.use_fp16() and self._device.startswith("cuda")
        for cid in TARGET_CLASS_IDS:
            path = self._state.model_paths.get(cid)
            if not self._path_exists(path):
                continue
            if not self._runtime_supports_path(path):
                fb = self._fallback_model_path(path)
                if fb and self._path_exists(fb):
                    name = CLASS_NAMES.get(cid, str(cid))
                    self.status.emit(
                        f"{name}: runtime unsupported for {os.path.basename(path)}, "
                        f"using {os.path.basename(fb)}"
                    )
                    path = fb
                    self._state.model_paths[cid] = fb
            name = CLASS_NAMES.get(cid, str(cid))
            tag = os.path.basename(path)
            self.status.emit(f"Loading {name}: {tag}")
            try:
                model = self._create_model(path)
                self._models[cid] = model
                self._model_paths[cid] = path
            except Exception as exc:
                fb = self._fallback_model_path(path)
                if fb and self._path_exists(fb):
                    fb_tag = os.path.basename(fb)
                    self.status.emit(
                        f"{name}: failed loading {tag}, fallback to {fb_tag}"
                    )
                    try:
                        model = self._create_model(fb)
                        self._models[cid] = model
                        self._model_paths[cid] = fb
                        self._state.model_paths[cid] = fb
                        continue
                    except Exception as fb_exc:
                        self.status.emit(
                            f"Failed fallback load {name} ({fb_tag}): {fb_exc}"
                        )
                self.status.emit(f"Failed to load {name} ({tag}): {exc}")
        self.status.emit(
            f"Inference backend: {self._device}"
            f"{' + FP16' if self._fp16 else ''}"
        )

    @staticmethod
    def _class_conf(cid: int, base_conf: float) -> float:
        """Return per-class adjusted confidence threshold."""
        factor = _CLASS_CONF_FACTOR.get(cid, 1.0)
        floor = _CLASS_CONF_MIN.get(cid, 0.0)
        return float(max(floor, min(1.0, base_conf * factor)))

    def _predict_kwargs_for(self, cid: int, conf: float, iou: float, imgsz: int) -> dict:
        """Backend-aware kwargs: avoid problematic runtime args on exported engines."""
        kwargs = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        path = str(self._model_paths.get(cid, "")).lower()
        is_pt = path.endswith(".pt")
        if is_pt:
            kwargs["device"] = self._device
            kwargs["half"] = self._fp16
        return kwargs

    def _predict_with_fallback(
        self,
        cid: int,
        frames: List[np.ndarray],
        kwargs: dict,
        allow_batch: bool,
    ):
        """Try batch predict first; fallback to per-frame on backend limitations."""
        model = self._models[cid]
        had_error = False
        if allow_batch:
            try:
                results = model.predict(frames, **kwargs)
                if isinstance(results, (list, tuple)) and len(results) == len(frames):
                    return list(results)
            except Exception:
                had_error = True

        # Backend may not support list/batch inputs reliably; fallback safely
        out = []
        for frame in frames:
            try:
                r = model.predict(frame, **kwargs)
                if isinstance(r, (list, tuple)):
                    out.append(r[0] if r else None)
                else:
                    out.append(r)
            except Exception:
                had_error = True
                out.append(None)

        # If everything failed, try rebinding to a fallback artifact once.
        if had_error and out and all(r is None for r in out):
            if self._try_rebind_fallback_model(cid, reason="runtime predict failure"):
                model = self._models[cid]
                path = self._model_paths.get(cid, "")
                allow_batch2 = not self._is_openvino_path(path)
                if allow_batch2:
                    try:
                        results = model.predict(frames, **kwargs)
                        if isinstance(results, (list, tuple)) and len(results) == len(frames):
                            return list(results)
                    except Exception:
                        pass
                out2 = []
                for frame in frames:
                    try:
                        r = model.predict(frame, **kwargs)
                        if isinstance(r, (list, tuple)):
                            out2.append(r[0] if r else None)
                        else:
                            out2.append(r)
                    except Exception:
                        out2.append(None)
                return out2
        return out

    # ── Detection extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_detections(
        result, class_id: int, include_track_ids: bool = False
    ) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        ids = getattr(boxes, "id", None)
        if xyxy is None:
            return []

        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        conf_np = (
            conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu")
            else (np.asarray(conf) if conf is not None else None)
        )
        id_np = None
        if include_track_ids and ids is not None:
            try:
                id_np = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            except Exception:
                id_np = None

        dets: List[Detection] = []
        for i in range(len(xyxy_np)):
            x1, y1, x2, y2 = (int(v) for v in xyxy_np[i])
            score = float(conf_np[i]) if conf_np is not None else 0.0
            tid = int(id_np[i]) if (id_np is not None and i < len(id_np)) else None
            dets.append(Detection(x1, y1, x2, y2, class_id, score, tid))
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

                # Run every enabled model on every batch of frames.  We used
                # to short-circuit non-motorcycle classes when no motorcycle
                # was found in the batch, but that hid valid stand-alone
                # detections (especially `improper_footwear`).  The IoA
                # association in the tracker thread already discards gear
                # that has no rider/motorcycle nearby, so it is cheap and
                # safer to always run every model.
                for cid in TARGET_CLASS_IDS:
                    if cid not in self._models or not state.is_model_enabled(cid):
                        continue

                    cls_conf = self._class_conf(cid, conf)
                    kwargs = self._predict_kwargs_for(cid, cls_conf, iou, imgsz)
                    path = self._model_paths.get(cid, "")
                    allow_batch = not self._is_openvino_path(path)
                    results = self._predict_with_fallback(
                        cid,
                        infer_frames,
                        kwargs,
                        allow_batch=allow_batch,
                    )

                    for local_i, result in enumerate(results):
                        if local_i >= len(infer_idxs) or result is None:
                            continue
                        pkt_i = infer_idxs[local_i]
                        extracted_dets = self._extract_detections(result, class_id=cid)
                        dets_by_idx[pkt_i].extend(extracted_dets)

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

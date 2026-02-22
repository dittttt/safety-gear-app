"""
Thread 2 – Inference Engine
===========================
Pulls ``FramePacket`` objects from the frame queue, runs every *enabled*
YOLO model, and pushes ``DetectionPacket`` objects into the detection queue.

* **Rider model** (class 1) uses ``model.track()`` for built-in BoT-SORT IDs.
* All other models use ``model.predict()`` (detection only, no tracker state).
* Supports ``.pt`` (PyTorch), ``.engine`` (TensorRT), and ``.onnx`` weights.
"""

import os
import time
import queue
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore
from ultralytics import YOLO
import torch

from config import (
    TARGET_CLASS_IDS, CLASS_NAMES,
    OPTIMIZED_GPU_DIR, OPTIMIZED_CPU_DIR,
)
from pipeline.state import PipelineState, Detection, DetectionPacket


class InferenceThread(QtCore.QThread):
    """Consumer of frames → producer of detections."""

    status = QtCore.pyqtSignal(str)
    fps_update = QtCore.pyqtSignal(float)

    def __init__(self, state: PipelineState, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._models: Dict[int, YOLO] = {}
        self._device: str = "cpu"
        self._fp16: bool = False

    def _resolve_device(self) -> str:
        device = self._state.get_device()
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Model management ───────────────────────────────────────────────────────

    def _resolve_optimized_path(self, original_path: str) -> str:
        """Return the best optimised model path for the current device, else *original_path*.

        GPU preference: .engine  > .onnx (CUDA)  > .pt
        CPU preference: _openvino_model/  > .onnx  > .pt
        """
        if not original_path or not original_path.lower().endswith(".pt"):
            return original_path
        basename = os.path.splitext(os.path.basename(original_path))[0]
        is_gpu = self._device.startswith("cuda")
        if is_gpu:
            # Best: TensorRT engine
            engine = os.path.join(OPTIMIZED_GPU_DIR, f"{basename}.engine")
            if os.path.isfile(engine):
                return engine
            # Fallback: ONNX (runs on CUDAExecutionProvider)
            onnx = os.path.join(OPTIMIZED_GPU_DIR, f"{basename}.onnx")
            if os.path.isfile(onnx):
                return onnx
        else:
            # Best: OpenVINO IR directory
            ov_dir = os.path.join(OPTIMIZED_CPU_DIR, f"{basename}_openvino_model")
            if os.path.isdir(ov_dir):
                return ov_dir
            # Fallback: ONNX (runs on CPUExecutionProvider)
            onnx = os.path.join(OPTIMIZED_CPU_DIR, f"{basename}.onnx")
            if os.path.isfile(onnx):
                return onnx
        return original_path

    def _load_models(self) -> None:
        """(Re)load models from current ``state.model_paths``."""
        self._models.clear()
        self._device = self._resolve_device()
        self._fp16 = self._state.use_fp16() and self._device.startswith("cuda")
        for cid in TARGET_CLASS_IDS:
            path = self._state.model_paths.get(cid)
            if not path or not os.path.isfile(path):
                continue
            resolved = self._resolve_optimized_path(path)
            name = CLASS_NAMES.get(cid, str(cid))
            tag = os.path.basename(resolved)
            self.status.emit(f"Loading {name}: {tag}")
            try:
                model = YOLO(resolved)
                model.to(self._device)
                self._models[cid] = model
            except Exception as exc:
                self.status.emit(f"Failed to load {name} ({tag}): {exc}")
                # Fallback to original .pt if optimised version failed
                if resolved != path:
                    self.status.emit(f"Falling back to {os.path.basename(path)}")
                    try:
                        model = YOLO(path)
                        model.to(self._device)
                        self._models[cid] = model
                    except Exception as exc2:
                        self.status.emit(f"Also failed on fallback: {exc2}")
        self.status.emit(
            f"Inference backend: {self._device}"
            f"{' + FP16' if self._fp16 else ''}"
        )

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

    # ── Tracker YAML resolution ────────────────────────────────────────────────

    def _resolve_tracker_yaml(self) -> str:
        key = self._state.detection_config.tracker_key
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = {
            "botsort": os.path.join(base, "trackers", "botsort.yaml"),
            "bytetrack": os.path.join(base, "trackers", "bytetrack.yaml"),
        }
        path = candidates.get(key, candidates["botsort"])
        return path if os.path.isfile(path) else candidates["botsort"]

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        state = self._state
        self._load_models()

        if not self._models:
            self.status.emit("No models loaded — overlay will be disabled")

        tracker_yaml = self._resolve_tracker_yaml()
        persist_tracking = False
        frame_count = 0
        t0 = time.perf_counter()
        step = 0
        cached_dets: List[Detection] = []

        while not state.stop_event.is_set():
            # Hot-reload models if flagged (e.g. after optimised conversion)
            if state.reload_models_flag:
                state.reload_models_flag = False
                self._load_models()

            # Pull next frame (with short timeout so we can still check stop).
            try:
                packet = state.frame_queue.get(timeout=0.05)
            except Exception:
                continue

            if state.stop_event.is_set():
                break

            # Tracker reset after seek
            if state.reset_tracker_flag:
                persist_tracking = False
                state.reset_tracker_flag = False

            frame = packet.frame
            conf = state.get_conf()
            iou = state.get_iou()
            imgsz = state.get_imgsz()
            stride = state.get_inference_stride()
            overlay = state.is_overlay_enabled()
            skip_inference = stride > 1 and step % stride != 0
            inferred_this_frame = False
            all_dets: List[Detection] = cached_dets if skip_inference else []

            if overlay and self._models and not skip_inference:
                inferred_this_frame = True
                # ── 1) Rider model drives BoT-SORT tracking ────────────────
                if 1 in self._models and state.is_model_enabled(1):
                    try:
                        results = self._models[1].track(
                            frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            half=self._fp16,
                            device=self._device,
                            persist=persist_tracking,
                            tracker=tracker_yaml,
                            verbose=False,
                        )
                        persist_tracking = True
                        r0 = results[0] if isinstance(results, (list, tuple)) else results
                        all_dets.extend(
                            self._extract_detections(r0, class_id=1, include_track_ids=True)
                        )
                    except Exception as exc:
                        self.status.emit(f"Rider inference error: {exc}")

                # ── 2) Other models run prediction (no tracker) ────────────
                for cid in TARGET_CLASS_IDS:
                    if cid == 1:
                        continue
                    if cid not in self._models or not state.is_model_enabled(cid):
                        continue
                    try:
                        results = self._models[cid].predict(
                            frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            half=self._fp16,
                            device=self._device,
                            verbose=False,
                        )
                        r0 = results[0] if isinstance(results, (list, tuple)) else results
                        all_dets.extend(self._extract_detections(r0, class_id=cid))
                    except Exception as exc:
                        self.status.emit(
                            f"Inference error ({CLASS_NAMES.get(cid, cid)}): {exc}"
                        )

            cached_dets = all_dets
            det_packet = DetectionPacket(
                index=packet.index,
                frame=frame,
                detections=all_dets,
                timestamp_ms=packet.timestamp_ms,
                detections_fresh=inferred_this_frame,
            )

            state.put_safe(state.detection_queue, det_packet)

            # FPS reporting
            frame_count += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self.fps_update.emit(frame_count / (now - t0))
                frame_count = 0
                t0 = now
            step += 1

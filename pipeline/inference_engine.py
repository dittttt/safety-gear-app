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

from config import (
    TARGET_CLASS_IDS,
    CLASS_NAMES,
    CLASS_MOTORCYCLE,
    CLASS_RIDER,
    CLASS_CONF_FACTOR,
    CLASS_CONF_MIN,
)
from pipeline.state import PipelineState, Detection, DetectionPacket
from utils.runtime_check import detect as detect_runtime


def _per_class_threshold(cid: int, base_conf: float) -> float:
    """Effective confidence floor for *cid* given the global slider value."""
    factor = CLASS_CONF_FACTOR.get(cid, 1.0)
    floor = CLASS_CONF_MIN.get(cid, 0.0)
    return float(max(floor, base_conf * factor))


def _floor_inference_conf(base_conf: float) -> float:
    """Lowest threshold the model must run at so post-filtering still works.

    The model itself filters by the ``conf`` argument BEFORE returning
    boxes, so we have to call it with the minimum across every class
    we still want to recover (otherwise low-conf improper-footwear hits
    are lost for good).
    """
    if not CLASS_CONF_MIN:
        return base_conf
    return float(min(min(CLASS_CONF_MIN.values()), base_conf))


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

    Multi-Model-v2 runs one single-class detector per target class.  Each
    model may emit local YOLO class ``0``; detections are remapped to the
    app-level class id keyed by ``self._models[cid]``.
    """

    UNIFIED_KEY: int = -1   # legacy sentinel; not used for model loading here

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
        """Try to rebind one class model to a sibling compatible artifact."""
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
                f"{name}: fallback bound -> {os.path.basename(fb)}"
                + (f" ({reason})" if reason else "")
            )
            return True
        except Exception:
            return False

    # ── Model management ───────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """(Re)load all configured per-class models."""
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
                        f"{name}: runtime cannot load {os.path.basename(path)}, "
                        f"using {os.path.basename(fb)}"
                    )
                    path = fb
                    self._state.model_paths[cid] = fb
            name = CLASS_NAMES.get(cid, str(cid))
            tag = os.path.basename(path) if not os.path.isdir(path) else os.path.basename(path.rstrip(os.sep))
            self.status.emit(f"Loading {name}: {tag}")
            try:
                model = self._create_model(path)
                self._models[cid] = model
                self._model_paths[cid] = path
            except Exception as exc:
                fb = self._fallback_model_path(path)
                if fb and self._path_exists(fb):
                    try:
                        model = self._create_model(fb)
                        self._models[cid] = model
                        self._model_paths[cid] = fb
                        self._state.model_paths[cid] = fb
                        self.status.emit(
                            f"{name}: failed loading {tag}, fallback to {os.path.basename(fb)}"
                        )
                        continue
                    except Exception as fb_exc:
                        self.status.emit(
                            f"Failed fallback load {name} ({os.path.basename(fb)}): {fb_exc}"
                        )
                self.status.emit(f"Failed to load {name} ({tag}): {exc}")

        self.status.emit(
            f"Inference backend: {self._device}"
            f"{' + FP16' if self._fp16 else ''}"
        )

    def _predict_kwargs_for(self, cid: int, conf: float, iou: float, imgsz: int) -> dict:
        """Backend-aware kwargs for one per-class detector."""
        model_path = self._model_paths.get(cid, "")
        path = str(model_path).lower()

        # TensorRT engines bake the input resolution in at export time. Trying
        # to predict at any other ``imgsz`` raises a runtime error and the
        # whole frame returns no detections — which the user perceives as
        # "detection just stopped working" the moment they touched the
        # inference-size dropdown. Snap to whatever the engine was built
        # for. OpenVINO IR has the same constraint (the static shape lives
        # in the .xml).
        if path.endswith(".engine") or self._is_openvino_path(model_path):
            engine_imgsz = self._native_imgsz(cid)
            if engine_imgsz and engine_imgsz != imgsz:
                imgsz = engine_imgsz

        kwargs = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        if path.endswith(".pt"):
            kwargs["device"] = self._device
            kwargs["half"] = self._fp16
        return kwargs

    def _native_imgsz(self, cid: int) -> Optional[int]:
        """Best-effort lookup of the model's baked-in input size."""
        model = self._models.get(cid)
        if model is None:
            return None
        # Ultralytics records the export-time imgsz in ``overrides``.
        try:
            sz = getattr(model, "overrides", {}).get("imgsz")
            if isinstance(sz, (list, tuple)) and sz:
                return int(sz[0])
            if isinstance(sz, (int, float)):
                return int(sz)
        except Exception:
            pass
        # Fall back to the framework default for our exports.
        return 640

    def _predict_with_fallback(
        self,
        cid: int,
        frames: List[np.ndarray],
        kwargs: dict,
        allow_batch: bool,
    ) -> List:
        """Run one class model, falling back to frame-by-frame if needed."""
        model = self._models.get(cid)
        if model is None:
            return [None] * len(frames)

        had_error = False
        if allow_batch:
            try:
                results = model.predict(frames, **kwargs)
                if isinstance(results, (list, tuple)) and len(results) == len(frames):
                    return list(results)
            except Exception:
                had_error = True
        out: List = []
        for frame in frames:
            try:
                r = model.predict(frame, **kwargs)
                out.append(r[0] if isinstance(r, (list, tuple)) and r else r)
            except Exception:
                had_error = True
                out.append(None)

        if had_error and out and all(r is None for r in out):
            if self._try_rebind_fallback_model(cid, reason="runtime predict failure"):
                retry_path = self._model_paths.get(cid, "")
                retry_allow_batch = not self._is_openvino_path(retry_path)
                model = self._models.get(cid)
                if model is None:
                    return out
                if retry_allow_batch:
                    try:
                        results = model.predict(frames, **kwargs)
                        if isinstance(results, (list, tuple)) and len(results) == len(frames):
                            return list(results)
                    except Exception:
                        pass
                retry_out: List = []
                for frame in frames:
                    try:
                        r = model.predict(frame, **kwargs)
                        retry_out.append(r[0] if isinstance(r, (list, tuple)) and r else r)
                    except Exception:
                        retry_out.append(None)
                return retry_out
        return out

    def _resolve_tracker_config(self) -> Optional[str]:
        """Return an absolute path to the active tracker YAML, if any."""
        cfg = self._state.detection_config
        if not getattr(cfg, "tracker_enabled", True):
            return None
        key = (getattr(cfg, "tracker_key", None) or "").strip().lower()
        if not key or key in {"none", "off", "disabled"}:
            return None
        # trackers/ lives at the repo root next to main.py.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "trackers", f"{key}.yaml")
        return path if os.path.isfile(path) else None

    def _track_frames(
        self,
        model: YOLO,
        frames: List[np.ndarray],
        kwargs: dict,
        tracker_cfg: str,
    ) -> List:
        """Run frame-by-frame tracking, respecting the reset flag."""
        # Tracking-only kwargs. ``model.track()`` does not accept ``half`` for
        # all backends; predict() does. Strip it to avoid a TypeError on some
        # Ultralytics versions.
        track_kwargs = dict(kwargs)
        track_kwargs.pop("half", None)
        track_kwargs["tracker"] = tracker_cfg
        track_kwargs.pop("classes", None)

        out: List = []
        for frame in frames:
            # A pending reset means we just seeked or switched source — wipe
            # the tracker's internal state on the very next call by passing
            # persist=False ONCE, then resume with persist=True.
            persist = True
            if self._state.reset_tracker_flag:
                self._state.reset_tracker_flag = False
                persist = False
            try:
                r = model.track(frame, persist=persist, **track_kwargs)
                out.append(r[0] if isinstance(r, (list, tuple)) and r else r)
            except Exception as exc:
                # If tracking fails (e.g. backend doesn't support it), fall
                # back to plain prediction for this frame so the pipeline
                # keeps producing detections.
                self.status.emit(f"Tracker error, falling back to predict: {exc}")
                try:
                    rp = model.predict(frame, **kwargs)
                    out.append(rp[0] if isinstance(rp, (list, tuple)) and rp else rp)
                except Exception:
                    out.append(None)
        return out

    @staticmethod
    def _bbox_iou(a: Detection, b: Detection) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
        area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / float(union) if union > 0 else 0.0

    @classmethod
    def _copy_vehicle_rider_track_ids(
        cls,
        detections: List[Detection],
        tracked: List[Detection],
        min_iou: float = 0.50,
    ) -> None:
        """Copy tracker IDs onto predict() detections for motorcycle/rider.

        The full predict() list keeps all gear boxes.  The tracker pass is
        only used as metadata for large objects where stable IDs matter.
        """
        if not detections or not tracked:
            return
        tracked_with_ids = [
            d for d in tracked
            if d.track_id is not None and d.class_id in (CLASS_MOTORCYCLE, CLASS_RIDER)
        ]
        if not tracked_with_ids:
            return
        for det in detections:
            if det.class_id not in (CLASS_MOTORCYCLE, CLASS_RIDER):
                continue
            best = None
            best_iou = 0.0
            for trk in tracked_with_ids:
                if trk.class_id != det.class_id:
                    continue
                score = cls._bbox_iou(det, trk)
                if score > best_iou:
                    best = trk
                    best_iou = score
            if best is not None and best_iou >= min_iou:
                det.track_id = best.track_id

    # ── Detection extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_detections(
        result,
        allowed_classes: Optional[set] = None,
        per_class_min: Optional[Dict[int, float]] = None,
        force_class_id: Optional[int] = None,
    ) -> List[Detection]:
        """Convert one Ultralytics result into a list of typed Detections.

        ``force_class_id`` remaps every emitted box to the app-level class id
        for a single-class model.

        ``per_class_min`` is a ``{cid: min_conf}`` map applied AFTER the
        model returns. Used to recover small-object recall by running the
        model at a low conf and tightening per-class thresholds in Python.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        ids = getattr(boxes, "id", None)
        if xyxy is None or (cls is None and force_class_id is None):
            return []

        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        cls_np = None if cls is None else (cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls))
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
            cid = force_class_id if force_class_id is not None else int(cls_np[i])
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            score = float(conf_np[i]) if conf_np is not None else 0.0
            if per_class_min is not None:
                thresh = per_class_min.get(cid)
                if thresh is not None and score < thresh:
                    continue
            x1, y1, x2, y2 = (int(v) for v in xyxy_np[i])
            tid = int(id_np[i]) if (id_np is not None and i < len(id_np)) else None
            dets.append(Detection(x1, y1, x2, y2, cid, score, tid))
        return dets

    @staticmethod
    def _apply_per_class_nms(
        dets: List[Detection],
        iou_overrides: Dict[int, Optional[float]],
    ) -> List[Detection]:
        """Apply a tighter per-class NMS pass when the user has set an
        IoU spinbox for a given class. Detections for classes without an
        override pass through untouched.

        Greedy NumPy NMS — fast enough for the small detection counts
        produced after the model's own NMS already ran.
        """
        if not dets or not iou_overrides:
            return dets
        # Bucket by class so we only NMS within each class.
        buckets: Dict[int, List[Detection]] = {}
        for d in dets:
            buckets.setdefault(d.class_id, []).append(d)
        kept: List[Detection] = []
        for cid, group in buckets.items():
            thresh = iou_overrides.get(cid)
            if thresh is None or len(group) < 2:
                kept.extend(group)
                continue
            # Sort by descending score, greedily keep boxes whose IoU
            # against any already-kept box is below the threshold.
            group_sorted = sorted(
                group, key=lambda d: d.confidence, reverse=True)
            survivors: List[Detection] = []
            for cand in group_sorted:
                drop = False
                for s in survivors:
                    ix1 = max(cand.x1, s.x1); iy1 = max(cand.y1, s.y1)
                    ix2 = min(cand.x2, s.x2); iy2 = min(cand.y2, s.y2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter <= 0:
                        continue
                    a1 = max(0, cand.x2 - cand.x1) * max(0, cand.y2 - cand.y1)
                    a2 = max(0, s.x2 - s.x1) * max(0, s.y2 - s.y1)
                    union = a1 + a2 - inter
                    if union > 0 and (inter / union) >= thresh:
                        drop = True
                        break
                if not drop:
                    survivors.append(cand)
            kept.extend(survivors)
        return kept

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
        # Cached per-class threshold maps. Rebuilt only when the user
        # touches a sidebar spinbox (tracked via state.per_class_override_version)
        # or when the global conf slider changes — saves ~2 lock acquisitions
        # and a dict-comp on every inference frame.
        cached_pc_version: int = -1
        cached_pc_conf: float = -1.0
        cached_per_class_min: Dict[int, float] = {}
        cached_iou_overrides: Dict[int, Optional[float]] = {}
        # True when at least one IoU override is tighter than the global
        # IoU we passed to Ultralytics (i.e. it can actually drop a box).
        cached_iou_nms_useful: bool = False

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

            # Tracking is stateful: BoT-SORT must see EVERY frame to keep
            # IDs stable. If the user enabled the tracker, ignore stride and
            # infer per-frame — otherwise feeding the tracker every Nth frame
            # makes IDs flip whenever something moves quickly.
            if stride > 1 and self._resolve_tracker_config():
                stride = 1

            infer_idxs = [
                i for i in range(len(packets))
                if (stride <= 1 or ((step + i) % stride == 0))
            ]
            infer_set = set(infer_idxs)
            dets_by_idx: Dict[int, List[Detection]] = {i: [] for i in infer_idxs}

            if overlay and self._models and infer_idxs:
                infer_frames = [packets[i].frame for i in infer_idxs]

                # Per-class min conf + IoU overrides: snapshot only when
                # the user has actually changed something (version bump)
                # or when the global conf slider has moved. Otherwise
                # reuse the cached maps — the dict-comp + 2 lock
                # acquisitions used to run on every single frame.
                pc_version = state.get_per_class_override_version()
                if pc_version != cached_pc_version or conf != cached_pc_conf:
                    conf_overrides = state.get_per_class_conf_overrides()
                    iou_overrides_snap = state.get_per_class_iou_overrides()
                    cached_per_class_min = {
                        cid: (
                            conf_overrides[cid]
                            if conf_overrides.get(cid) is not None
                            else _per_class_threshold(cid, conf)
                        )
                        for cid in TARGET_CLASS_IDS
                    }
                    cached_iou_overrides = iou_overrides_snap
                    # The post-extraction NMS pass only matters when an
                    # override is tighter than the IoU Ultralytics already
                    # applied — anything looser cannot drop more boxes.
                    cached_iou_nms_useful = any(
                        v is not None and v < iou
                        for v in iou_overrides_snap.values()
                    )
                    cached_pc_version = pc_version
                    cached_pc_conf = conf

                per_class_min = cached_per_class_min
                run_iou_nms = cached_iou_nms_useful
                iou_overrides = cached_iou_overrides

                tracker_cfg = self._resolve_tracker_config()
                for cid in TARGET_CLASS_IDS:
                    if cid not in self._models or not state.is_model_enabled(cid):
                        continue
                    model_conf = per_class_min.get(cid, _per_class_threshold(cid, conf))
                    model_iou = iou_overrides.get(cid) if iou_overrides.get(cid) is not None else iou
                    kwargs = self._predict_kwargs_for(cid, model_conf, model_iou, imgsz)
                    path = self._model_paths.get(cid, "")
                    allow_batch = not self._is_openvino_path(path)
                    results = self._predict_with_fallback(
                        cid,
                        infer_frames,
                        kwargs,
                        allow_batch=allow_batch,
                    )

                    tracked_results = []
                    if tracker_cfg and cid in (CLASS_MOTORCYCLE, CLASS_RIDER):
                        model = self._models.get(cid)
                        if model is not None:
                            tracked_results = self._track_frames(
                                model,
                                infer_frames,
                                kwargs,
                                tracker_cfg,
                            )

                    for local_i, result in enumerate(results):
                        if local_i >= len(infer_idxs) or result is None:
                            continue
                        pkt_i = infer_idxs[local_i]
                        extracted = self._extract_detections(
                            result,
                            per_class_min={cid: model_conf},
                            force_class_id=cid,
                        )
                        if local_i < len(tracked_results) and tracked_results[local_i] is not None:
                            tracked = self._extract_detections(
                                tracked_results[local_i],
                                force_class_id=cid,
                            )
                            self._copy_vehicle_rider_track_ids(extracted, tracked)
                        if run_iou_nms:
                            extracted = self._apply_per_class_nms(
                                extracted, iou_overrides)
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

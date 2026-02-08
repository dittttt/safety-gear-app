import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass
class AppConfig:
    conf: float = 0.25
    iou: float = 0.45
    tracker_key: str = "botsort"
    rider_moto_ioa: float = 0.05
    gear_rider_ioa: float = 0.20
    overlay_enabled: bool = True


CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (128, 128, 128),  # Motorcycle - Gray
    1: (255, 0, 0),      # Rider - Blue (BGR)
    2: (0, 255, 0),      # Helmet - Green
    3: (0, 255, 0),      # Footwear - Green
    4: (0, 0, 255),      # Improper_Footwear - Red
}

CLASS_NAMES_DEFAULT: Dict[int, str] = {
    0: "Motorcycle",
    1: "Rider",
    2: "Helmet",
    3: "Footwear",
    4: "Improper_Footwear",
}


TARGET_CLASS_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

def _box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _ioa(child: Tuple[int, int, int, int], parent: Tuple[int, int, int, int]) -> float:
    area_child = _box_area(child)
    if area_child <= 0:
        return 0.0
    inter = _intersection_area(child, parent)
    return inter / float(area_child)


class DropLabel(QtWidgets.QLabel):
    """QLabel that accepts drag-and-drop file paths."""

    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, text: str = "", allowed_exts: Tuple[str, ...] = (), parent=None) -> None:
        super().__init__(text, parent)
        self._allowed_exts = tuple(e.lower() for e in allowed_exts)
        self.setAcceptDrops(True)

    def _is_allowed(self, path: str) -> bool:
        if not self._allowed_exts:
            return True
        ext = os.path.splitext(path)[1].lower()
        return ext in self._allowed_exts

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return

        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if len(paths) != 1:
            event.ignore()
            return
        if not self._is_allowed(paths[0]):
            event.ignore()
            return

        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return
        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if len(paths) != 1:
            event.ignore()
            return
        path = paths[0]
        if not self._is_allowed(path):
            event.ignore()
            return
        self.fileDropped.emit(path)
        event.acceptProposedAction()


class MultiDropLabel(QtWidgets.QLabel):
    """QLabel that accepts drag-and-drop of one or more file paths."""

    pathsDropped = QtCore.pyqtSignal(list)

    def __init__(self, text: str = "", allowed_exts: Tuple[str, ...] = (), parent=None) -> None:
        super().__init__(text, parent)
        self._allowed_exts = tuple(e.lower() for e in allowed_exts)
        self.setAcceptDrops(True)

    def _is_allowed(self, path: str) -> bool:
        if not self._allowed_exts:
            return True
        ext = os.path.splitext(path)[1].lower()
        return ext in self._allowed_exts

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return

        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if not paths:
            event.ignore()
            return

        allowed = [p for p in paths if self._is_allowed(p)]
        if not allowed:
            event.ignore()
            return

        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return
        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if not paths:
            event.ignore()
            return

        allowed = [p for p in paths if self._is_allowed(p)]
        if not allowed:
            event.ignore()
            return

        self.pathsDropped.emit(allowed)
        event.acceptProposedAction()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def tracker_yaml_path(tracker_key: str) -> str:
    local = os.path.join(repo_root(), "trackers")
    candidates = {
        "botsort": os.path.join(local, "botsort.yaml"),
        "bytetrack": os.path.join(local, "bytetrack.yaml"),
        "strongsort": os.path.join(local, "strongsort.yaml"),
    }
    return candidates.get(tracker_key, candidates["botsort"])


class VideoThread(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    statusReady = QtCore.pyqtSignal(str)
    metaReady = QtCore.pyqtSignal(float, int)  # fps, total_frames
    positionReady = QtCore.pyqtSignal(int)     # current_frame_index (0-based)

    def __init__(self) -> None:
        super().__init__()
        self._video_path: Optional[str] = None
        self._model_paths: Dict[int, Optional[str]] = {cid: None for cid in TARGET_CLASS_IDS}
        self._config = AppConfig()
        self._class_colors_bgr: Dict[int, Tuple[int, int, int]] = dict(CLASS_COLORS_BGR)
        self._stop = False
        self._paused = True
        self._restart_requested = False
        self._models: Dict[int, YOLO] = {}

        self._video_fps: float = 0.0
        self._total_frames: int = 0
        self._seek_to_frame: Optional[int] = None
        self._reset_tracker_next: bool = False
        self._playback_rate: float = 1.0

        self._frame_delay_ms: int = 1

    def set_video_path(self, path: Optional[str]) -> None:
        self._video_path = path

    def set_model_path_for_class(self, class_id: int, path: Optional[str]) -> None:
        if class_id not in self._model_paths:
            return
        self._model_paths[class_id] = path
        self._restart_requested = True

    def set_conf(self, conf: float) -> None:
        self._config.conf = clamp01(conf)

    def set_iou(self, iou: float) -> None:
        self._config.iou = clamp01(iou)

    def set_tracker(self, tracker_key: str) -> None:
        self._config.tracker_key = tracker_key
        self._restart_requested = True

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._config.overlay_enabled = bool(enabled)
        # Reload/clear models depending on new overlay state.
        self._restart_requested = True

    def request_seek(self, frame_index: int) -> None:
        try:
            idx = int(frame_index)
        except Exception:
            return
        if idx < 0:
            idx = 0
        self._seek_to_frame = idx

    def set_playback_rate(self, rate: float) -> None:
        try:
            r = float(rate)
        except Exception:
            return
        if r <= 0:
            return
        # Clamp to a reasonable range.
        self._playback_rate = max(0.1, min(8.0, r))

    def set_rider_moto_ioa(self, value: float) -> None:
        self._config.rider_moto_ioa = max(0.0, min(1.0, float(value)))

    def set_gear_rider_ioa(self, value: float) -> None:
        self._config.gear_rider_ioa = max(0.0, min(1.0, float(value)))

    def set_class_color_bgr(self, class_id: int, bgr: Tuple[int, int, int]) -> None:
        if class_id not in TARGET_CLASS_IDS:
            return
        b, g, r = [int(v) for v in bgr]
        self._class_colors_bgr[class_id] = (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))

    def play(self) -> None:
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def stop(self) -> None:
        self._stop = True
        self._paused = True

    def _emit_status(self, text: str) -> None:
        self.statusReady.emit(text)

    def _load_model(self) -> None:
        if not self._config.overlay_enabled:
            self._models.clear()
            return

        rider_path = self._model_paths.get(1)
        if not rider_path:
            # Preview-only mode (no overlay). Still allow playback + seeking.
            self._models.clear()
            self._emit_status("Overlay disabled (set Rider model to enable tracking)")
            return

        # Load selected models; Rider (class 1) drives tracking.
        self._models.clear()
        for class_id, path in self._model_paths.items():
            if not path:
                continue
            self._emit_status(
                f"Loading model for {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))}: {os.path.basename(path)}"
            )
            self._models[class_id] = YOLO(path)

        if 1 not in self._models:
            raise RuntimeError("Failed to load Rider model")

    def _collect_dets(
        self,
        result,
        class_id_override: int,
        include_track_ids: bool,
    ) -> List[Tuple[int, int, int, int, int, float, Optional[int]]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        ids = getattr(boxes, "id", None)

        if xyxy is None:
            return []

        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)

        conf_arr = None
        if conf is not None:
            conf_arr = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

        id_arr = None
        if include_track_ids and ids is not None:
            try:
                id_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            except Exception:
                id_arr = None

        dets: List[Tuple[int, int, int, int, int, float, Optional[int]]] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            score = float(conf_arr[i]) if conf_arr is not None else 0.0
            track_id: Optional[int] = None
            if id_arr is not None and i < len(id_arr) and id_arr[i] is not None:
                try:
                    track_id = int(id_arr[i])
                except Exception:
                    track_id = None
            dets.append((x1, y1, x2, y2, int(class_id_override), score, track_id))

        return dets

    def _annotate(self, frame_bgr: np.ndarray, dets) -> np.ndarray:
        annotated = frame_bgr.copy()

        for (x1, y1, x2, y2, class_id, score, track_id) in dets:
            color = self._class_colors_bgr.get(class_id, (0, 255, 255))

            label = CLASS_NAMES_DEFAULT.get(class_id, str(class_id))
            label += f" {score:.2f}"

            if class_id == 1 and track_id is not None:
                label += f" ID:{track_id}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx1, ty1 = x1, max(0, y1 - th - 8)
            cv2.rectangle(annotated, (tx1, ty1), (tx1 + tw + 6, ty1 + th + 6), color, -1)
            cv2.putText(
                annotated,
                label,
                (tx1 + 3, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _filter_dets_by_overlap(self, dets):
        """Reduce false positives by enforcing expected spatial relationships.

        Rules:
        - Rider (1) must overlap a Motorcycle (0)
        - Helmet (2), Footwear (3), Improper_Footwear (4) must overlap a Rider (1)
        """
        motorcycles = [d for d in dets if d[4] == 0]
        riders = [d for d in dets if d[4] == 1]
        gear = [d for d in dets if d[4] in (2, 3, 4)]

        other = [d for d in dets if d[4] not in (0, 1, 2, 3, 4)]

        moto_boxes = [(d[0], d[1], d[2], d[3]) for d in motorcycles]

        # Filter riders by overlap with motorcycles
        kept_riders = []
        rider_boxes = []
        if moto_boxes:
            for d in riders:
                rbox = (d[0], d[1], d[2], d[3])
                ok = any(_ioa(rbox, mbox) >= self._config.rider_moto_ioa for mbox in moto_boxes)
                if ok:
                    kept_riders.append(d)
                    rider_boxes.append(rbox)
        else:
            # If there are no motorcycles detected, keep zero riders to avoid pedestrian false positives.
            kept_riders = []
            rider_boxes = []

        # Filter gear by overlap with kept riders
        kept_gear = []
        if rider_boxes:
            for d in gear:
                gbox = (d[0], d[1], d[2], d[3])
                ok = any(_ioa(gbox, rbox) >= self._config.gear_rider_ioa for rbox in rider_boxes)
                if ok:
                    kept_gear.append(d)

        return motorcycles + kept_riders + kept_gear + other

    def run(self) -> None:
        self._stop = False
        self._paused = True

        if not self._video_path:
            self._emit_status("Select a video to start")
            return

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._emit_status("Failed to open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        self._video_fps = float(fps) if fps and fps > 0 else 0.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.metaReady.emit(self._video_fps, self._total_frames)

        try:
            self._load_model()
        except Exception as e:
            self._emit_status(f"Model load error: {e}")
            cap.release()
            return

        if 1 in self._models:
            self._emit_status("Ready (press Play) | overlay enabled")
        else:
            self._emit_status("Ready (press Play) | overlay disabled")

        last_time = time.time()
        frame_count = 0

        while not self._stop:
            force_one_frame = False

            # Apply seek requests even while paused, then render one frame so the UI updates.
            if self._seek_to_frame is not None:
                idx = self._seek_to_frame
                self._seek_to_frame = None
                if idx < 0:
                    idx = 0
                if self._total_frames > 0:
                    idx = min(idx, self._total_frames - 1)

                # Some codecs/backends are unreliable with frame-based seeking; fall back to time-based.
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if (not pos_after) and self._video_fps and self._video_fps > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (idx / self._video_fps) * 1000.0)
                elif pos_after and abs(float(pos_after) - float(idx)) > 2.0 and self._video_fps and self._video_fps > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (idx / self._video_fps) * 1000.0)

                # Seeking breaks temporal continuity; reset tracking state on next frame.
                self._reset_tracker_next = True
                self._emit_status(f"Seek: frame {idx}")
                force_one_frame = True

            if self._paused and not force_one_frame:
                self.msleep(25)
                continue

            if self._restart_requested:
                self._restart_requested = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reload models so model-path changes take effect.
                try:
                    self._load_model()
                except Exception as e:
                    self._emit_status(f"Model load error: {e}")
                    self._paused = True
                    continue
                self._emit_status("Restarted video due to configuration change")

                self._reset_tracker_next = True

            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                self._emit_status("End of video")
                self._paused = True
                continue

            # Update current frame position (0-based) for UI.
            try:
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if pos < 0:
                    pos = 0
                self.positionReady.emit(pos)
            except Exception:
                pass

            try:
                if not self._config.overlay_enabled or 1 not in self._models:
                    # Preview-only mode.
                    annotated = frame
                else:
                    tracker_path = tracker_yaml_path(self._config.tracker_key)
                    if self._config.tracker_key == "strongsort" and not os.path.exists(tracker_path):
                        tracker_path = tracker_yaml_path("bytetrack")

                    # 1) Rider model drives tracking.
                    rider_model = self._models[1]
                    track_results = rider_model.track(
                        frame,
                        conf=self._config.conf,
                        iou=self._config.iou,
                        persist=(not self._reset_tracker_next),
                        tracker=tracker_path,
                        verbose=False,
                    )

                    self._reset_tracker_next = False

                    track_result0 = track_results[0] if isinstance(track_results, (list, tuple)) else track_results
                    dets = self._collect_dets(track_result0, class_id_override=1, include_track_ids=True)

                    # 2) Other single-class models run detection (no tracking IDs).
                    for class_id in TARGET_CLASS_IDS:
                        if class_id == 1:
                            continue
                        model = self._models.get(class_id)
                        if model is None:
                            continue
                        pred_results = model.predict(
                            frame,
                            conf=self._config.conf,
                            iou=self._config.iou,
                            verbose=False,
                        )
                        pred0 = pred_results[0] if isinstance(pred_results, (list, tuple)) else pred_results
                        dets.extend(self._collect_dets(pred0, class_id_override=class_id, include_track_ids=False))

                    dets = self._filter_dets_by_overlap(dets)
                    annotated = self._annotate(frame, dets)
            except Exception as e:
                self._emit_status(f"Inference error: {e}")
                self._paused = True
                continue

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            self.frameReady.emit(qimg)

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self._emit_status(
                    f"Running | conf={self._config.conf:.2f} iou={self._config.iou:.2f} "
                    f"tracker={self._config.tracker_key} | FPS~{frame_count / (now - last_time):.1f}"
                )
                last_time = now
                frame_count = 0

            # Playback rate control.
            fps_eff = self._video_fps if self._video_fps > 0 else 0.0
            if fps_eff > 0:
                target_ms = 1000.0 / (fps_eff * max(0.1, self._playback_rate))
            else:
                target_ms = 0.0

            elapsed_ms = (time.time() - frame_start) * 1000.0
            sleep_ms = int(max(0.0, target_ms - elapsed_ms))
            if sleep_ms > 0:
                self.msleep(min(250, sleep_ms))

            # Speed-up by skipping frames for common >1x rates.
            if self._playback_rate >= 2.0:
                skip = int(self._playback_rate) - 1
                for _ in range(skip):
                    if self._stop or self._paused:
                        break
                    if not cap.grab():
                        break

        cap.release()
        self._emit_status("Stopped")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Safety Gear Compliance Tester (YOLO + Tracking)")

        # Enable drag-and-drop anywhere in the window (Windows can be finicky if you
        # don't drop exactly on the label).
        self.setAcceptDrops(True)

        self._video_path: Optional[str] = None
        self._model_paths: Dict[int, Optional[str]] = {cid: None for cid in TARGET_CLASS_IDS}
        self._settings = QtCore.QSettings("safety-gear-app", "sgct")
        self._class_colors_bgr: Dict[int, Tuple[int, int, int]] = dict(CLASS_COLORS_BGR)
        self._load_saved_colors()

        self._thread = VideoThread()
        self._thread.frameReady.connect(self._on_frame)
        self._thread.statusReady.connect(self._set_status)
        self._thread.metaReady.connect(self._on_video_meta)
        self._thread.positionReady.connect(self._on_video_position)

        self._video_fps: float = 0.0
        self._total_frames: int = 0
        self._user_seeking: bool = False

        for cid, bgr in self._class_colors_bgr.items():
            self._thread.set_class_color_bgr(cid, bgr)

        self._build_ui()

    def _load_saved_colors(self) -> None:
        for cid in TARGET_CLASS_IDS:
            key = f"colors/{cid}"
            val = self._settings.value(key, None)
            if not val:
                continue
            try:
                if isinstance(val, str):
                    parts = [int(x.strip()) for x in val.split(",")]
                else:
                    parts = [int(x) for x in val]
                if len(parts) == 3:
                    b, g, r = parts
                    self._class_colors_bgr[cid] = (b, g, r)
            except Exception:
                continue

    def _save_color(self, class_id: int, bgr: Tuple[int, int, int]) -> None:
        b, g, r = [int(v) for v in bgr]
        self._settings.setValue(f"colors/{class_id}", f"{b},{g},{r}")

    def _apply_color_border(self, widget: QtWidgets.QWidget, bgr: Tuple[int, int, int]) -> None:
        b, g, r = [int(v) for v in bgr]
        widget.setStyleSheet(f"border: 2px solid rgb({r}, {g}, {b}); padding: 2px;")

    def _pick_color_for_class(self, class_id: int, label_widget: QtWidgets.QWidget) -> None:
        b, g, r = self._class_colors_bgr.get(class_id, (0, 255, 255))
        initial = QtGui.QColor(r, g, b)
        color = QtWidgets.QColorDialog.getColor(initial, self, f"Select {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))} Color")
        if not color.isValid():
            return

        new_bgr = (int(color.blue()), int(color.green()), int(color.red()))
        self._class_colors_bgr[class_id] = new_bgr
        self._save_color(class_id, new_bgr)
        self._thread.set_class_color_bgr(class_id, new_bgr)
        self._apply_color_border(label_widget, new_bgr)
        self._set_status(f"Color updated for {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))}")

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return

        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if not paths:
            event.ignore()
            return

        allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".pt"}
        if not any(os.path.splitext(p)[1].lower() in allowed_exts for p in paths):
            event.ignore()
            return

        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return

        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        if not paths:
            event.ignore()
            return

        self._handle_dropped_paths(paths)
        event.acceptProposedAction()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self.videoLabel = MultiDropLabel(
            "Drop a video and/or .pt model files here",
            allowed_exts=(".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".pt"),
            parent=self,
        )
        self.videoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.videoLabel.setMinimumSize(960, 540)
        self.videoLabel.setStyleSheet("background-color: #111; color: #ddd;")

        self.btnLoadVideo = QtWidgets.QPushButton("Load Video (.mp4)")

        self.btnLoadMotorcycle = QtWidgets.QPushButton("Load Motorcycle Model")
        self.btnLoadRider = QtWidgets.QPushButton("Load Rider Model (required)")
        self.btnLoadHelmet = QtWidgets.QPushButton("Load Helmet Model")
        self.btnLoadFootwear = QtWidgets.QPushButton("Load Footwear Model")
        self.btnLoadImproperFootwear = QtWidgets.QPushButton("Load Improper Footwear Model")

        self.lblMotorcycle = DropLabel("(not set)", allowed_exts=(".pt",))
        self.lblRider = DropLabel("(not set)", allowed_exts=(".pt",))
        self.lblHelmet = DropLabel("(not set)", allowed_exts=(".pt",))
        self.lblFootwear = DropLabel("(not set)", allowed_exts=(".pt",))
        self.lblImproperFootwear = DropLabel("(not set)", allowed_exts=(".pt",))

        # Show current bbox colors (border) and allow right-click to change.
        for cid, lbl in [
            (0, self.lblMotorcycle),
            (1, self.lblRider),
            (2, self.lblHelmet),
            (3, self.lblFootwear),
            (4, self.lblImproperFootwear),
        ]:
            self._apply_color_border(lbl, self._class_colors_bgr.get(cid, CLASS_COLORS_BGR[cid]))
            lbl.setToolTip("Right-click to change bounding box color")
            lbl.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            lbl.customContextMenuRequested.connect(lambda _pos, c=cid, w=lbl: self._pick_color_for_class(c, w))

        self.btnPlay = QtWidgets.QPushButton("Play")
        self.btnPause = QtWidgets.QPushButton("Pause")
        self.btnStop = QtWidgets.QPushButton("Stop")

        self.overlayToggle = QtWidgets.QCheckBox("Overlay")
        self.overlayToggle.setChecked(True)

        self.positionSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.setEnabled(False)
        self.positionSlider.setTracking(False)

        self.timeCurrent = QtWidgets.QLabel("0:00")
        self.timeTotal = QtWidgets.QLabel("0:00")

        self.speedCombo = QtWidgets.QComboBox()
        self.speedCombo.addItem("0.25x", 0.25)
        self.speedCombo.addItem("0.5x", 0.5)
        self.speedCombo.addItem("1x", 1.0)
        self.speedCombo.addItem("2x", 2.0)
        self.speedCombo.addItem("4x", 4.0)
        self.speedCombo.setCurrentIndex(2)

        self.trackerCombo = QtWidgets.QComboBox()
        self.trackerCombo.addItem("BoT-SORT (primary)", "botsort")
        self.trackerCombo.addItem("StrongSORT (if available)", "strongsort")
        self.trackerCombo.addItem("ByteTrack (comparison)", "bytetrack")

        self.confSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.confSlider.setRange(0, 100)
        self.confSlider.setValue(25)

        self.iouSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iouSlider.setRange(0, 100)
        self.iouSlider.setValue(45)

        self.riderMotoSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.riderMotoSlider.setRange(0, 100)
        self.riderMotoSlider.setValue(5)

        self.gearRiderSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gearRiderSlider.setRange(0, 100)
        self.gearRiderSlider.setValue(20)

        self.confValue = QtWidgets.QLabel("0.25")
        self.iouValue = QtWidgets.QLabel("0.45")
        self.riderMotoValue = QtWidgets.QLabel("0.05")
        self.gearRiderValue = QtWidgets.QLabel("0.20")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.btnLoadVideo, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Tracker"), 0, 2)
        grid.addWidget(self.trackerCombo, 0, 3)

        grid.addWidget(self.btnLoadMotorcycle, 1, 0)
        grid.addWidget(self.lblMotorcycle, 1, 1)
        grid.addWidget(self.btnLoadRider, 1, 2)
        grid.addWidget(self.lblRider, 1, 3)

        grid.addWidget(self.btnLoadHelmet, 2, 0)
        grid.addWidget(self.lblHelmet, 2, 1)
        grid.addWidget(self.btnLoadFootwear, 2, 2)
        grid.addWidget(self.lblFootwear, 2, 3)

        grid.addWidget(self.btnLoadImproperFootwear, 3, 0)
        grid.addWidget(self.lblImproperFootwear, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Conf Threshold"), 4, 0)
        grid.addWidget(self.confSlider, 4, 1, 1, 2)
        grid.addWidget(self.confValue, 4, 3)

        grid.addWidget(QtWidgets.QLabel("IoU Threshold"), 5, 0)
        grid.addWidget(self.iouSlider, 5, 1, 1, 2)
        grid.addWidget(self.iouValue, 5, 3)

        grid.addWidget(QtWidgets.QLabel("Rider requires Motorcycle (IoA)"), 6, 0)
        grid.addWidget(self.riderMotoSlider, 6, 1, 1, 2)
        grid.addWidget(self.riderMotoValue, 6, 3)

        grid.addWidget(QtWidgets.QLabel("Gear requires Rider (IoA)"), 7, 0)
        grid.addWidget(self.gearRiderSlider, 7, 1, 1, 2)
        grid.addWidget(self.gearRiderValue, 7, 3)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnPlay)
        btnRow.addWidget(self.btnPause)
        btnRow.addWidget(self.btnStop)
        btnRow.addWidget(self.overlayToggle)
        btnRow.addStretch(1)
        btnRow.addWidget(QtWidgets.QLabel("Speed"))
        btnRow.addWidget(self.speedCombo)

        self.statusBar = QtWidgets.QLabel("Idle")

        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(grid)
        layout.addWidget(self.videoLabel, stretch=1)
        seekRow = QtWidgets.QHBoxLayout()
        seekRow.addWidget(self.timeCurrent)
        seekRow.addWidget(self.positionSlider, stretch=1)
        seekRow.addWidget(self.timeTotal)
        layout.addLayout(seekRow)
        layout.addLayout(btnRow)
        layout.addWidget(self.statusBar)

        self.btnLoadVideo.clicked.connect(self._pick_video)
        self.btnLoadMotorcycle.clicked.connect(lambda: self._pick_model_for_class(0))
        self.btnLoadRider.clicked.connect(lambda: self._pick_model_for_class(1))
        self.btnLoadHelmet.clicked.connect(lambda: self._pick_model_for_class(2))
        self.btnLoadFootwear.clicked.connect(lambda: self._pick_model_for_class(3))
        self.btnLoadImproperFootwear.clicked.connect(lambda: self._pick_model_for_class(4))

        self.videoLabel.pathsDropped.connect(self._handle_dropped_paths)
        self.lblMotorcycle.fileDropped.connect(lambda p: self._set_model_path_for_class(0, p))
        self.lblRider.fileDropped.connect(lambda p: self._set_model_path_for_class(1, p))
        self.lblHelmet.fileDropped.connect(lambda p: self._set_model_path_for_class(2, p))
        self.lblFootwear.fileDropped.connect(lambda p: self._set_model_path_for_class(3, p))
        self.lblImproperFootwear.fileDropped.connect(lambda p: self._set_model_path_for_class(4, p))

        self.btnPlay.clicked.connect(self._play)
        self.btnPause.clicked.connect(self._pause)
        self.btnStop.clicked.connect(self._stop)

        self.positionSlider.sliderPressed.connect(self._on_seek_pressed)
        self.positionSlider.sliderReleased.connect(self._on_seek_released)
        self.positionSlider.sliderMoved.connect(self._on_seek_moved)
        self.speedCombo.currentIndexChanged.connect(self._on_speed_changed)
        self.overlayToggle.toggled.connect(self._on_overlay_toggled)

        self.confSlider.valueChanged.connect(self._on_conf_changed)
        self.iouSlider.valueChanged.connect(self._on_iou_changed)
        self.riderMotoSlider.valueChanged.connect(self._on_rider_moto_changed)
        self.gearRiderSlider.valueChanged.connect(self._on_gear_rider_changed)
        self.trackerCombo.currentIndexChanged.connect(self._on_tracker_changed)

    def _set_status(self, text: str) -> None:
        self.statusBar.setText(text)

    def _pick_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if not path:
            return
        self._set_video_path(path)

    def _set_video_path(self, path: str) -> None:
        if not path:
            return
        self._video_path = path
        self._thread.set_video_path(path)
        self._set_status(f"Video selected: {os.path.basename(path)}")

        # Quick sanity preview to confirm OpenCV can open/decode this file.
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(
                self,
                "Video Open Failed",
                "OpenCV could not open the selected video.\n\n"
                "Common causes:\n"
                "- Unsupported codec (e.g., some H.265/HEVC MP4s)\n"
                "- File path permissions\n\n"
                f"Path:\n{path}",
            )
            self._set_status("Failed to open selected video")
            return

        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if not ok or frame is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Video Decode Failed",
                "OpenCV opened the file but could not decode the first frame.\n\n"
                "Try re-encoding the video to H.264 (AVC) or using a different file.",
            )
            self._set_status("Failed to decode video")
            return

        # Display preview frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        self._on_frame(qimg)

        # Initialize seek UI from preview metadata.
        self._video_fps = float(fps) if fps and fps > 0 else 0.0
        self._total_frames = int(total or 0)
        self._init_seek_ui()

    def _format_time(self, seconds: float) -> str:
        if seconds is None or seconds < 0:
            seconds = 0.0
        total = int(seconds + 0.5)
        m = total // 60
        s = total % 60
        return f"{m}:{s:02d}"

    def _init_seek_ui(self) -> None:
        if self._total_frames and self._total_frames > 1:
            self.positionSlider.setEnabled(True)
            self.positionSlider.setRange(0, self._total_frames - 1)
        else:
            self.positionSlider.setEnabled(False)
            self.positionSlider.setRange(0, 0)

        if self._video_fps and self._video_fps > 0 and self._total_frames and self._total_frames > 0:
            total_seconds = (self._total_frames - 1) / self._video_fps
            self.timeTotal.setText(self._format_time(total_seconds))
        else:
            self.timeTotal.setText("0:00")

    def _on_video_meta(self, fps: float, total_frames: int) -> None:
        # Thread metadata (may differ slightly from preview).
        if fps and fps > 0:
            self._video_fps = float(fps)
        if total_frames and total_frames > 0:
            self._total_frames = int(total_frames)
        self._init_seek_ui()

    def _on_video_position(self, frame_index: int) -> None:
        if self._user_seeking:
            return
        if self._total_frames and self._total_frames > 0:
            frame_index = max(0, min(int(frame_index), self._total_frames - 1))
        else:
            frame_index = max(0, int(frame_index))

        self.positionSlider.blockSignals(True)
        self.positionSlider.setValue(frame_index)
        self.positionSlider.blockSignals(False)

        if self._video_fps and self._video_fps > 0:
            self.timeCurrent.setText(self._format_time(frame_index / self._video_fps))

    def _on_seek_pressed(self) -> None:
        self._user_seeking = True

    def _on_seek_released(self) -> None:
        self._user_seeking = False
        target = int(self.positionSlider.value())
        if self._thread.isRunning():
            self._thread.request_seek(target)
        else:
            self._seek_preview_frame(target)

    def _on_seek_moved(self, value: int) -> None:
        # Update the time label while dragging.
        if self._video_fps and self._video_fps > 0:
            self.timeCurrent.setText(self._format_time(int(value) / self._video_fps))

    def _seek_preview_frame(self, frame_index: int) -> None:
        if not self._video_path:
            return
        if not self._total_frames or self._total_frames <= 0:
            return
        idx = int(frame_index)
        idx = max(0, min(idx, self._total_frames - 1))

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        self._on_frame(qimg)
        self._on_video_position(idx)
        self._set_status(f"Preview seek: frame {idx}")

    def _on_speed_changed(self) -> None:
        rate = self.speedCombo.currentData()
        if rate is None:
            return
        self._thread.set_playback_rate(float(rate))

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._thread.set_overlay_enabled(bool(checked))
        if checked:
            self._set_status("Overlay enabled")
        else:
            self._set_status("Overlay disabled")

    def _normalize_name(self, path: str) -> str:
        base = os.path.basename(path)
        base = os.path.splitext(base)[0]
        base = base.lower()
        for ch in ["-", "_", "."]:
            base = base.replace(ch, " ")
        return " ".join(base.split())

    def _infer_class_id_from_filename(self, path: str) -> Optional[int]:
        name = self._normalize_name(path)

        # Order matters (e.g., 'improper_footwear' contains 'footwear')
        patterns = [
            (4, ["improper footwear", "improper", "no footwear", "nofootwear", "no shoe", "noshoe", "barefoot"]),
            (3, ["footwear", "shoe", "shoes", "boot", "boots", "safety shoe", "safetyshoe"]),
            (2, ["helmet", "hardhat"]),
            (0, ["motorcycle", "motor bike", "motorbike", "moto"]),
            (1, ["rider", "riding", "driver", "person"]),
        ]

        for class_id, keys in patterns:
            for k in keys:
                if k in name:
                    return class_id
        return None

    def _handle_dropped_paths(self, paths: list) -> None:
        if not paths:
            return

        unknown: List[str] = []
        handled_any = False

        for path in paths:
            if not isinstance(path, str) or not path:
                continue
            ext = os.path.splitext(path)[1].lower()

            if ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}:
                self._set_video_path(path)
                handled_any = True
                continue

            if ext == ".pt":
                class_id = self._infer_class_id_from_filename(path)
                if class_id is None:
                    unknown.append(os.path.basename(path))
                    continue
                self._set_model_path_for_class(class_id, path)
                handled_any = True
                continue

            unknown.append(os.path.basename(path))

        if unknown and handled_any:
            QtWidgets.QMessageBox.information(
                self,
                "Some files were not recognized",
                "These dropped files could not be auto-assigned:\n\n" + "\n".join(unknown) +
                "\n\nTip: include keywords like rider, motorcycle, helmet, footwear, improper in the filename.",
            )
        elif unknown and not handled_any:
            QtWidgets.QMessageBox.information(
                self,
                "No supported files",
                "Drop a video (mp4/avi/mov/mkv/wmv/flv) and/or model weights (.pt) here.",
            )

    def _pick_model_for_class(self, class_id: int) -> None:
        title = f"Select {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))} Model"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, "", "PyTorch Model (*.pt);;All Files (*)")
        if not path:
            return

        self._set_model_path_for_class(class_id, path)

    def _set_model_path_for_class(self, class_id: int, path: str) -> None:
        if not path:
            return

        self._model_paths[class_id] = path
        self._thread.set_model_path_for_class(class_id, path)

        base = os.path.basename(path)
        if class_id == 0:
            self.lblMotorcycle.setText(base)
        elif class_id == 1:
            self.lblRider.setText(base)
        elif class_id == 2:
            self.lblHelmet.setText(base)
        elif class_id == 3:
            self.lblFootwear.setText(base)
        elif class_id == 4:
            self.lblImproperFootwear.setText(base)

        self._set_status(f"Model set for {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))}: {base}")

    def _ensure_thread_started(self) -> None:
        if self._thread.isRunning():
            return
        self._thread.start()

    def _play(self) -> None:
        if not self._video_path:
            self._set_status("Select a video first")
            return
        self._ensure_thread_started()
        self._thread.play()
        if self.overlayToggle.isChecked() and self._model_paths.get(1):
            self._set_status("Playing (overlay enabled)")
        elif self.overlayToggle.isChecked() and not self._model_paths.get(1):
            self._set_status("Playing (overlay enabled but Rider model not set)")
        else:
            self._set_status("Playing (overlay disabled)")

    def _pause(self) -> None:
        self._thread.pause()
        self._set_status("Paused")

    def _stop(self) -> None:
        if self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1500)
        self._set_status("Stopped")

    def _on_conf_changed(self, value: int) -> None:
        conf = value / 100.0
        self.confValue.setText(f"{conf:.2f}")
        self._thread.set_conf(conf)

    def _on_iou_changed(self, value: int) -> None:
        iou = value / 100.0
        self.iouValue.setText(f"{iou:.2f}")
        self._thread.set_iou(iou)

    def _on_rider_moto_changed(self, value: int) -> None:
        thr = value / 100.0
        self.riderMotoValue.setText(f"{thr:.2f}")
        self._thread.set_rider_moto_ioa(thr)

    def _on_gear_rider_changed(self, value: int) -> None:
        thr = value / 100.0
        self.gearRiderValue.setText(f"{thr:.2f}")
        self._thread.set_gear_rider_ioa(thr)

    def _on_tracker_changed(self) -> None:
        tracker_key = self.trackerCombo.currentData()
        self._thread.set_tracker(tracker_key)
        yaml_path = tracker_yaml_path(tracker_key)
        if tracker_key == "strongsort" and not os.path.exists(yaml_path):
            self._set_status("StrongSORT config not found; falling back to ByteTrack")
        else:
            self._set_status(f"Tracker set: {tracker_key}")

    def _on_frame(self, qimg: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        ))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._stop()
        finally:
            event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

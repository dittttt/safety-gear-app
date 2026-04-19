"""
Thread 1 – Frame Grabber
========================
Continuously reads video frames and pushes ``FramePacket`` objects into the
frame queue.  Handles playback pacing, pause / resume, and seek requests.
"""

import os
import time
from typing import Optional

import cv2
from PyQt5 import QtCore

from pipeline.state import PipelineState, FramePacket
from utils.camera_devices import open_camera_capture


class FrameGrabberThread(QtCore.QThread):
    """Producer: reads frames from video / camera at the correct playback rate."""

    metaReady = QtCore.pyqtSignal(float, int)           # (fps, total_frames)
    positionChanged = QtCore.pyqtSignal(int, float)      # (frame_idx, timestamp_sec)
    finished_signal = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, state: PipelineState, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._state = state

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901  (unavoidable complexity for the main loop)
        state = self._state
        path, camera_source = state.get_source()
        is_live_camera = camera_source is not None

        if not is_live_camera and not path:
            self.error.emit("No source selected")
            return

        # Keep FFmpeg warnings quiet during frequent random-access seeks.
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

        if is_live_camera:
            cap = open_camera_capture(camera_source)
            if isinstance(camera_source, dict):
                src_name = str(camera_source.get("label") or camera_source.get("target") or "camera")
            else:
                src_name = "camera"
        else:
            cap = cv2.VideoCapture(path)
            src_name = str(path)

        if not cap.isOpened():
            self.error.emit(f"Cannot open: {src_name}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = 0 if is_live_camera else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        state.video_fps = float(fps)
        state.total_frames = total
        self.metaReady.emit(float(fps), total)

        frame_idx = 0
        next_frame_time = time.perf_counter()

        while not state.stop_event.is_set():
            # ── Handle seek (file sources only) ───────────────────────────
            if not is_live_camera:
                seek_target = state.consume_seek()
                if seek_target is not None:
                    state.flush_queues()
                    # Decode from slightly before target to avoid H264 reference
                    # picture issues on non-keyframe random seeks.
                    warmup = max(2, int(fps * 0.5))
                    start_idx = max(0, int(seek_target) - warmup)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                    frame_idx = start_idx

                    while frame_idx < int(seek_target):
                        if not cap.grab():
                            self.finished_signal.emit()
                            cap.release()
                            return
                        frame_idx += 1

                    state.reset_tracker_flag = True
                    next_frame_time = time.perf_counter()

                    # Always decode and publish one frame immediately after seek,
                    # so UI updates instantly in both paused and playing states.
                    ok, frame = cap.read()
                    if ok:
                        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)
                        state.put_safe(state.frame_queue, packet)
                        self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
                        frame_idx += 1
            else:
                # Consume stale seek requests when source is a live camera.
                state.consume_seek()

            # ── Pause ──────────────────────────────────────────────────────
            if state.pause_event.is_set():
                next_frame_time = time.perf_counter()
                self.msleep(30)
                continue

            ok, frame = cap.read()
            if not ok:
                self.finished_signal.emit()
                break

            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)

            # Non-blocking put; drop oldest if full (keeps grabber responsive).
            state.put_safe(state.frame_queue, packet)

            self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
            frame_idx += 1

            # ── Playback pacing ────────────────────────────────────────────
            rate = state.get_playback_rate()
            frame_period = (1.0 / (fps * rate)) if fps > 0 and rate > 0 else 0.0
            if frame_period > 0.0:
                next_frame_time += frame_period
                now = time.perf_counter()
                if next_frame_time < now - (frame_period * 2.0):
                    next_frame_time = now
                sleep_s = max(0.0, next_frame_time - now)
            else:
                sleep_s = 0.0

            if sleep_s > 0.0:
                end_t = next_frame_time
                while True:
                    remaining = end_t - time.perf_counter()
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(max(0.0, remaining - 0.001))

            # Frame-skipping for high playback rates (>=2×)
            if not is_live_camera and rate >= 2.0:
                skip = int(rate) - 1
                for _ in range(skip):
                    if state.stop_event.is_set() or state.pause_event.is_set():
                        break
                    if not cap.grab():
                        break
                    frame_idx += 1

        cap.release()

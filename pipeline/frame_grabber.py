"""
Thread 1 – Frame Grabber
========================
Continuously reads video frames and pushes ``FramePacket`` objects into the
frame queue.  Handles playback pacing, pause / resume, and seek requests.
"""

import time
from typing import Optional

import cv2
from PyQt5 import QtCore

from pipeline.state import PipelineState, FramePacket


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
        path = state.video_path

        if not path:
            self.error.emit("No video path set")
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open: {path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        state.video_fps = float(fps)
        state.total_frames = total
        self.metaReady.emit(float(fps), total)

        frame_idx = 0

        while not state.stop_event.is_set():
            # ── Handle seek ────────────────────────────────────────────────
            seek_target = state.consume_seek()
            if seek_target is not None:
                state.flush_queues()
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                frame_idx = seek_target
                state.reset_tracker_flag = True

                if state.pause_event.is_set():
                    ok, frame = cap.read()
                    if ok:
                        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)

                        placed = False
                        while not placed and not state.stop_event.is_set():
                            try:
                                state.frame_queue.put(packet, timeout=0.05)
                                placed = True
                            except Exception:
                                try:
                                    state.frame_queue.get_nowait()
                                except Exception:
                                    pass

                        self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
                        frame_idx += 1

            # ── Pause ──────────────────────────────────────────────────────
            if state.pause_event.is_set():
                self.msleep(30)
                continue

            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                self.finished_signal.emit()
                break

            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)

            # Put into the queue; if full, drop the current frame rather than
            # blocking forever (keeps the grabber responsive to stop / seek).
            placed = False
            while not placed and not state.stop_event.is_set():
                try:
                    state.frame_queue.put(packet, timeout=0.05)
                    placed = True
                except Exception:
                    # Queue full — drop the oldest stale frame and retry
                    try:
                        state.frame_queue.get_nowait()
                    except Exception:
                        pass

            self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
            frame_idx += 1

            # ── Playback pacing ────────────────────────────────────────────
            rate = state.get_playback_rate()
            target_delay = (1.0 / (fps * rate)) if fps > 0 and rate > 0 else 0.0
            elapsed = time.perf_counter() - frame_start
            sleep_s = target_delay - elapsed
            if sleep_s > 0.001:
                self.msleep(int(sleep_s * 1000))

            # Frame-skipping for high playback rates (>=2×)
            if rate >= 2.0:
                skip = int(rate) - 1
                for _ in range(skip):
                    if state.stop_event.is_set() or state.pause_event.is_set():
                        break
                    if not cap.grab():
                        break
                    frame_idx += 1

        cap.release()

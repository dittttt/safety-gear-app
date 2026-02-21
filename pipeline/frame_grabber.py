"""
Thread 1 – Frame Grabber
========================
Continuously reads video frames and pushes ``FramePacket`` objects into the
frame queue.  Handles playback pacing, pause / resume, and seek requests.
"""

import time
import queue
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
        next_frame_time = time.perf_counter()

        while not state.stop_event.is_set():
            # ── Handle seek ────────────────────────────────────────────────
            seek_target = state.consume_seek()
            if seek_target is not None:
                state.flush_queues()
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                frame_idx = seek_target
                state.reset_tracker_flag = True
                next_frame_time = time.perf_counter()

                if state.pause_event.is_set():
                    ok, frame = cap.read()
                    if ok:
                        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)

                        try:
                            state.frame_queue.put_nowait(packet)
                        except queue.Full:
                            try:
                                state.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                state.frame_queue.put_nowait(packet)
                            except queue.Full:
                                pass

                        self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
                        frame_idx += 1

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

            # Put into the queue; if full, drop the current frame rather than
            # blocking forever (keeps the grabber responsive to stop / seek).
            try:
                state.frame_queue.put_nowait(packet)
            except queue.Full:
                try:
                    state.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state.frame_queue.put_nowait(packet)
                except queue.Full:
                    pass

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
            if rate >= 2.0:
                skip = int(rate) - 1
                for _ in range(skip):
                    if state.stop_event.is_set() or state.pause_event.is_set():
                        break
                    if not cap.grab():
                        break
                    frame_idx += 1

        cap.release()

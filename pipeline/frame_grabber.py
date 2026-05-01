"""
Thread 1 – Frame Grabber
========================
Continuously reads video frames and pushes ``FramePacket`` objects into the
frame queue.  Handles playback pacing, pause / resume, and seek requests.
"""

import os
import sys
import time
import contextlib
from typing import Optional

import cv2
from PyQt5 import QtCore

from config import IMAGE_EXTENSIONS
from pipeline.state import PipelineState, FramePacket
from utils.camera_devices import open_camera_capture


@contextlib.contextmanager
def _silence_fd2():
    """Redirect OS-level stderr (fd 2) to NUL.

    Used to swallow libavcodec's ``[h264 @ 0x...] reference picture missing``
    spam that follows every random-access seek.  Those messages bypass
    Python's ``sys.stderr`` because they come from FFmpeg's C-side
    ``av_log()`` writing directly to fd 2, so the only reliable way to
    silence them cross-platform is at the file-descriptor level.
    """
    devnull = "NUL" if sys.platform == "win32" else "/dev/null"
    old_fd = None
    devnull_fd = None
    try:
        sys.stderr.flush()
        devnull_fd = os.open(devnull, os.O_WRONLY)
        old_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        yield
    except OSError:
        # If the dup trick fails (restricted env), just don't silence.
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if old_fd is not None:
            try:
                os.dup2(old_fd, 2)
                os.close(old_fd)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


def _read_quiet(cap):
    """``cap.read()`` with fd-2 silenced — for use inside the post-seek window."""
    with _silence_fd2():
        return cap.read()


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
        is_image_source = (
            bool(path)
            and os.path.splitext(str(path))[1].lower() in IMAGE_EXTENSIONS
        )

        if not is_live_camera and not path:
            self.error.emit("No source selected")
            return

        # Keep FFmpeg warnings quiet during frequent random-access seeks.
        # Using "-8" (quiet) silences the H.264 "reference picture missing"
        # spam that appears after every seek to a non-keyframe.
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

        if is_image_source:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                self.error.emit(f"Cannot open: {path}")
                return
            state.video_fps = 1.0
            state.total_frames = 1
            self.metaReady.emit(1.0, 1)
            emitted = False
            while not state.stop_event.is_set():
                if state.consume_seek() is not None:
                    state.flush_queues()
                    state.reset_tracker_flag = True
                    emitted = False
                if state.pause_event.is_set():
                    self.msleep(30)
                    continue
                if not emitted:
                    packet = FramePacket(index=0, frame=frame.copy(), timestamp_ms=0.0)
                    state.put_safe(state.frame_queue, packet)
                    self.positionChanged.emit(0, 0.0)
                    emitted = True
                self.msleep(50)
            return

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

        if is_live_camera:
            # Keep the OS / driver buffer as small as possible so we always
            # display the most recent frame instead of a queued stale one.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = 0 if is_live_camera else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        state.video_fps = float(fps)
        state.total_frames = total
        self.metaReady.emit(float(fps), total)

        frame_idx = 0
        next_frame_time = time.perf_counter()
        # FFmpeg / libavcodec spams ``[h264] reference picture missing during
        # reorder`` for ~1 GOP after every random-access seek.  We open a
        # short fd-2 silencer window after each seek and only swallow stderr
        # during that window — legitimate errors outside it still surface.
        quiet_until: float = 0.0
        SEEK_QUIET_SECONDS: float = 1.5

        while not state.stop_event.is_set():
            # ── Handle seek (file sources only) ───────────────────────────
            if not is_live_camera:
                seek_target = state.consume_seek()
                if seek_target is not None:
                    # Coalesce a burst of seek requests — if the user is
                    # still dragging the slider, several requests may be
                    # queued.  Drain them and only honour the latest one.
                    while True:
                        latest = state.consume_seek()
                        if latest is None:
                            break
                        seek_target = latest

                    state.flush_queues()

                    target = max(0, int(seek_target))
                    if total > 0:
                        target = min(target, max(0, total - 1))

                    # Let FFmpeg do the keyframe-aligned seek itself.
                    # POS_FRAMES set seeks to the previous keyframe and
                    # decodes the B/P frames up to *target* internally,
                    # which avoids the manual warmup loop (much faster) and
                    # the "reference picture missing" decoder errors caused
                    # by feeding the decoder an arbitrary mid-GOP start.
                    with _silence_fd2():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                    frame_idx = target
                    state.reset_tracker_flag = True
                    next_frame_time = time.perf_counter()
                    quiet_until = time.perf_counter() + SEEK_QUIET_SECONDS

                    # Decode and publish one frame immediately so the UI
                    # updates instantly in both paused and playing states.
                    with _silence_fd2():
                        ok, frame = cap.read()
                    if ok and frame is not None:
                        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)
                        state.put_safe(state.frame_queue, packet)
                        self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
                        frame_idx += 1
                    elif total > 0:
                        # Read failed at end of file — re-arm at the last
                        # frame and continue without dying.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, total - 1)))
                        frame_idx = max(0, total - 1)
            else:
                # Consume stale seek requests when source is a live camera.
                state.consume_seek()

            # ── Pause ──────────────────────────────────────────────────────
            if state.pause_event.is_set():
                next_frame_time = time.perf_counter()
                self.msleep(30)
                continue

            ok, frame = (
                _read_quiet(cap)
                if (not is_live_camera and time.perf_counter() < quiet_until)
                else cap.read()
            )
            if not ok:
                if is_live_camera:
                    # A transient camera read failure should not kill the
                    # pipeline.  Back off briefly and try again.
                    self.msleep(15)
                    continue
                self.finished_signal.emit()
                break

            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            packet = FramePacket(index=frame_idx, frame=frame, timestamp_ms=ts_ms)

            # Non-blocking put; drop oldest if full (keeps grabber responsive).
            state.put_safe(state.frame_queue, packet)

            self.positionChanged.emit(frame_idx, ts_ms / 1000.0 if ts_ms else 0.0)
            frame_idx += 1

            # ── Playback pacing ────────────────────────────────────────────
            if is_live_camera:
                # Live cameras: never sleep — the camera itself paces us.
                # Skip pacing & frame-skip logic entirely.
                continue

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
                    # Wake immediately if paused or stopped.
                    if state.pause_event.is_set() or state.stop_event.is_set():
                        break
                    if remaining > 0.003:
                        time.sleep(max(0.0, remaining - 0.001))

            # Frame-skipping for high playback rates (>=2×)
            if not is_live_camera and rate >= 2.0:
                skip = int(rate) - 1
                quiet = time.perf_counter() < quiet_until
                ctx = _silence_fd2() if quiet else contextlib.nullcontext()
                with ctx:
                    for _ in range(skip):
                        if state.stop_event.is_set() or state.pause_event.is_set():
                            break
                        if not cap.grab():
                            break
                        frame_idx += 1

        cap.release()

from __future__ import annotations

import argparse
import os
import queue
import time
from pathlib import Path

import cv2

from config import DEFAULT_MODEL_FILES
from pipeline.frame_grabber import FrameGrabberThread
from pipeline.inference_engine import InferenceThread
from pipeline.state import PipelineState
from pipeline.tracker_logic import TrackerLogicThread


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_video(video_arg: str | None) -> str:
    if video_arg:
        p = Path(video_arg)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Video not found: {video_arg}")

    fallback = _repo_root() / "source" / "basak.mp4"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("Could not find source/basak.mp4")


def _load_default_models(state: PipelineState) -> None:
    models_dir = _repo_root() / "models"
    for cid, fname in DEFAULT_MODEL_FILES.items():
        model_path = models_dir / fname
        if model_path.is_file():
            state.model_paths[cid] = str(model_path)


def benchmark(
    video_path: str,
    frames: int,
    imgsz: int,
    stride: int,
    simulate_display: bool,
    display_width: int,
    display_height: int,
) -> float:
    state = PipelineState()
    state.video_path = video_path
    state.set_overlay_enabled(True)
    state.set_imgsz(imgsz)
    state.set_inference_stride(stride)
    state.set_device("auto")
    state.set_use_fp16(False)
    _load_default_models(state)

    grabber = FrameGrabberThread(state)
    inferencer = InferenceThread(state)
    tracker = TrackerLogicThread(state)

    state.pause_event.clear()

    tracker.start()
    inferencer.start()
    grabber.start()

    target_frames = max(60, int(frames))
    warmup = min(45, max(10, target_frames // 6))
    seen = 0
    measured = 0
    t0 = None

    timeout_s = max(90.0, target_frames / 8.0)
    end_by = time.perf_counter() + timeout_s

    try:
        while time.perf_counter() < end_by:
            if not (grabber.isRunning() or inferencer.isRunning() or tracker.isRunning()):
                break
            try:
                packet = state.display_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if simulate_display:
                rgb = cv2.cvtColor(packet.annotated_frame, cv2.COLOR_BGR2RGB)
                _ = cv2.resize(rgb, (display_width, display_height), interpolation=cv2.INTER_NEAREST)

            seen += 1
            if seen == warmup:
                measured = 0
                t0 = time.perf_counter()
                continue

            if seen > warmup:
                measured += 1
                if measured >= target_frames:
                    break
    finally:
        state.stop_event.set()
        state.pause_event.clear()
        for thread in (grabber, inferencer, tracker):
            if thread.isRunning():
                thread.wait(5000)

    if t0 is None:
        raise RuntimeError("Benchmark could not collect warm-up frames")

    elapsed = max(1e-6, time.perf_counter() - t0)
    return measured / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark overlay-on pipeline FPS.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (defaults to source/basak.mp4)")
    parser.add_argument("--frames", type=int, default=240, help="Measured frames after warm-up")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--stride", type=int, default=1, help="Inference stride")
    parser.add_argument("--simulate-display", action="store_true", help="Include RGB convert + resize cost in benchmark loop")
    parser.add_argument("--display-width", type=int, default=1095)
    parser.add_argument("--display-height", type=int, default=620)
    args = parser.parse_args()

    video = _resolve_video(args.video)
    fps = benchmark(
        video_path=video,
        frames=args.frames,
        imgsz=args.imgsz,
        stride=args.stride,
        simulate_display=args.simulate_display,
        display_width=args.display_width,
        display_height=args.display_height,
    )

    print("=== BASELINE FPS BENCH ===")
    print(f"video={os.path.basename(video)}")
    print(f"overlay=on, imgsz={args.imgsz}, stride={args.stride}, simulate_display={args.simulate_display}")
    print(f"measured_fps={fps:.2f}")
    print("target_fps=30.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

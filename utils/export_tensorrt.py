"""
Export PyTorch YOLO models (.pt) to TensorRT FP16 (.engine) for maximum
inference speed on NVIDIA GPUs.

Usage
-----
Export a single model::

    python -m utils.export_tensorrt --model models/rider.pt

Export every .pt in a directory::

    python -m utils.export_tensorrt --model-dir models/

Options::

    --format   engine | onnx | openvino   (default: engine)
    --half     FP16 precision             (default: enabled)
    --no-half  Disable FP16
    --imgsz    Input image size           (default: 640)

Requirements
------------
* NVIDIA GPU with CUDA & cuDNN installed.
* ``pip install tensorrt`` (or the NVIDIA-provided wheel for your CUDA version).
* Ultralytics >= 8.0  (uses ``model.export()``).
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def export_model(
    model_path: str,
    fmt: str = "engine",
    half: bool = True,
    imgsz: int = 640,
) -> str:
    """Export a single YOLO .pt model and return the output path."""
    print(f"\n{'='*60}")
    print(f"Loading : {model_path}")
    model = YOLO(model_path, task="detect")

    print(f"Exporting to {fmt}  (half={half}, imgsz={imgsz}) …")
    exported = model.export(format=fmt, half=half, imgsz=imgsz)
    print(f"Exported: {exported}")
    print(f"{'='*60}\n")
    return str(exported)


def export_directory(
    models_dir: str,
    fmt: str = "engine",
    half: bool = True,
    imgsz: int = 640,
) -> None:
    """Export every .pt file found in *models_dir*."""
    p = Path(models_dir)
    if not p.is_dir():
        print(f"Not a directory: {models_dir}")
        return

    pt_files = sorted(p.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {models_dir}")
        return

    print(f"Found {len(pt_files)} model(s) to export.\n")
    for pt in pt_files:
        try:
            export_model(str(pt), fmt, half, imgsz)
        except Exception as exc:
            print(f"[FAILED] {pt.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt models to optimised inference formats"
    )
    parser.add_argument("--model", type=str, help="Path to a single .pt model")
    parser.add_argument(
        "--model-dir", type=str,
        help="Directory containing .pt models (batch export)",
    )
    parser.add_argument(
        "--format", type=str, default="engine",
        choices=["engine", "onnx", "openvino"],
        help="Target format (default: engine → TensorRT)",
    )
    parser.add_argument(
        "--half", action="store_true", default=True,
        help="FP16 precision (enabled by default)",
    )
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (default 640)"
    )
    args = parser.parse_args()

    half = args.half and not args.no_half

    if args.model:
        export_model(args.model, args.format, half, args.imgsz)
    elif args.model_dir:
        export_directory(args.model_dir, args.format, half, args.imgsz)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

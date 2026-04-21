"""
Runtime environment detection for model optimisation.

Call ``detect()`` once at startup (it caches results).  Every other module
should import from here rather than probing packages themselves.

Returned ``RuntimeInfo`` (dict-like) describes what is available and the
*best* export format to use for a given device:

  GPU (CUDA):   TensorRT engine  →  if tensorrt installed + GPU present
                ONNX/CUDA        →  if onnxruntime-gpu installed + GPU
                pytorch (fp16)   →  always available as final fallback
  CPU:          OpenVINO         →  if openvino installed
                ONNX/CPU         →  if onnxruntime installed
                pytorch          →  always available as final fallback
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional


# ── version helpers ──────────────────────────────────────────────────────────

def _version(module_name: str) -> Optional[str]:
    try:
        import importlib
        m = importlib.import_module(module_name)
        v = getattr(m, "__version__", None)
        if v:
            return str(v).split("-")[0].split("+")[0]   # strip build suffixes
        return "installed"
    except ImportError:
        return None


# ── data class ───────────────────────────────────────────────────────────────

@dataclass
class RuntimeInfo:
    # Raw availability
    has_cuda: bool = False
    cuda_device: str = ""
    torch_cuda_ver: str = ""

    has_tensorrt: bool = False
    tensorrt_ver: str = ""

    has_onnxruntime_gpu: bool = False
    onnxruntime_ver: str = ""

    has_openvino: bool = False
    openvino_ver: str = ""

    has_onnx: bool = False
    onnx_ver: str = ""

    # Best formats to use (what convert_worker should pass to `model.export`)
    best_gpu_format: str = "pt"      # engine → onnx (cuda) → pt
    best_cpu_format: str = "pt"      # openvino → onnx (cpu) → pt

    # Human-readable summary lines for UI tooltips
    gpu_summary: str = ""
    cpu_summary: str = ""

    # Extension/subdir for the chosen GPU / CPU artifacts
    gpu_artifact_ext: str = ".pt"
    cpu_artifact_subdir: str = ""


# ── main detection ───────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def detect() -> RuntimeInfo:
    """Probe available runtimes once and return a cached ``RuntimeInfo``."""
    info = RuntimeInfo()

    # --- CUDA / PyTorch ---
    try:
        import torch
        info.has_cuda = torch.cuda.is_available()
        torch_version = getattr(torch, "version", None)
        info.torch_cuda_ver = str(getattr(torch_version, "cuda", "") or "")
        if info.has_cuda:
            info.cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # --- TensorRT ---
    info.tensorrt_ver = _version("tensorrt") or ""
    info.has_tensorrt = bool(info.tensorrt_ver) and info.has_cuda

    # --- ONNX Runtime GPU ---
    ort_ver = _version("onnxruntime")
    if ort_ver:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            info.has_onnxruntime_gpu = (
                "CUDAExecutionProvider" in providers
                or "TensorrtExecutionProvider" in providers
            )
            info.onnxruntime_ver = ort_ver
        except ImportError:
            pass

    # --- OpenVINO ---
    info.openvino_ver = _version("openvino") or ""
    info.has_openvino = bool(info.openvino_ver)

    # --- ONNX (base) ---
    info.onnx_ver = _version("onnx") or ""
    info.has_onnx = bool(info.onnx_ver)

    # --- Choose best GPU format ---
    if info.has_cuda:
        info.best_gpu_format = "engine"
        info.gpu_artifact_ext = ".engine"
        info.gpu_summary = (
            f"TensorRT (auto-install) · CUDA {info.torch_cuda_ver}"
            f" · {info.cuda_device}"
        )
    elif info.has_onnxruntime_gpu and info.has_onnx:
        info.best_gpu_format = "onnx"
        info.gpu_artifact_ext = ".onnx"
        info.gpu_summary = (
            f"ONNX Runtime-GPU {info.onnxruntime_ver} (no TensorRT)"
        )
    else:
        info.best_gpu_format = "pt"
        info.gpu_artifact_ext = ".pt"
        info.gpu_summary = "PyTorch (no TRT/ONNX-GPU available)"

    # --- Choose best CPU format ---
    info.best_cpu_format = "openvino"
    info.cpu_artifact_subdir = "{stem}_openvino_model"
    info.cpu_summary = f"OpenVINO (auto-install)"
    
    return info


def best_format(device_key: str) -> str:
    """Return the best export format string for *device_key* ('cuda'/'cpu')."""
    info = detect()
    return info.best_gpu_format if device_key != "cpu" else info.best_cpu_format


def summary(device_key: str) -> str:
    """Return a human-readable summary of the best runtime for *device_key*."""
    info = detect()
    return info.gpu_summary if device_key != "cpu" else info.cpu_summary

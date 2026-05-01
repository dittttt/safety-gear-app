"""
Model discovery and variant management.

Scans the ``models/`` directory tree for every available model file
and groups them by model stem (motorcycle, helmet, …).  Provides
auto-resolution to pick the best variant for a given device/runtime.

Canonical per-model layout::

    models/
        motorcycle/
            motorcycle.pt
            motorcycle.engine
            motorcycle_openvino_model/
        helmet/
            helmet.pt
            ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import DEFAULT_MODEL_FILES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")

_STEM_TO_CID: Dict[str, int] = {
    os.path.splitext(os.path.basename(path))[0]: cid
    for cid, path in DEFAULT_MODEL_FILES.items()
}

_FORMAT_LABEL: Dict[str, str] = {
    "pt": "PyTorch",
    "engine": "TensorRT",
    "onnx": "ONNX",
    "openvino": "OpenVINO",
}


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ModelVariant:
    """One loadable model file/directory."""
    path: str
    format: str          # pt | engine | onnx | openvino
    display_name: str    # human-readable, shown in combo box

    @property
    def is_directory(self) -> bool:
        return os.path.isdir(self.path)


@dataclass
class ModelGroup:
    """All known variants for one logical model (e.g. "motorcycle")."""
    stem: str
    class_id: int
    variants: List[ModelVariant] = field(default_factory=list)

    def get_variant(self, fmt: str) -> Optional[ModelVariant]:
        for v in self.variants:
            if v.format == fmt:
                return v
        return None

    @property
    def has_pt(self) -> bool:
        return self.get_variant("pt") is not None

    def best_for_device(self, device: str) -> Optional[ModelVariant]:
        """Pick the best available variant for *device* (``'cuda'`` or ``'cpu'``)."""
        from utils.runtime_check import detect as _rt
        rt = _rt()

        if device.startswith("cuda"):
            order: list[str] = []
            if rt.has_tensorrt:
                order.append("engine")
            if rt.has_onnxruntime_gpu:
                order.append("onnx")
            order.append("pt")
        else:
            order = []
            if rt.has_openvino:
                order.append("openvino")
            if rt.has_onnx or rt.has_onnxruntime_gpu:
                order.append("onnx")
            order.append("pt")

        for fmt in order:
            v = self.get_variant(fmt)
            if v and _exists(v.path):
                return v
        # Return first available variant as fallback
        for v in self.variants:
            if _exists(v.path):
                return v
        return None


# ── helpers ──────────────────────────────────────────────────────────────────

def _exists(p: str) -> bool:
    return os.path.isfile(p) or os.path.isdir(p)


def _detect_format(path: str) -> Optional[str]:
    """Detect model format from a file path or directory."""
    if os.path.isdir(path):
        base = os.path.basename(path)
        if base.endswith("_openvino_model") or base.endswith("_OV_model"):
            return "openvino"
        # Check for .xml inside (OpenVINO IR)
        try:
            if any(f.endswith(".xml") for f in os.listdir(path)):
                return "openvino"
        except OSError:
            pass
        return None
    ext = os.path.splitext(path)[1].lower()
    return {".pt": "pt", ".engine": "engine", ".onnx": "onnx"}.get(ext)


def _shorten_basename(name: str) -> str:
    """Shorten verbose OpenVINO directory names for display."""
    name = name.replace("_openvino_model", "_OV_model")
    return name


def _make_variant(path: str, fmt: str) -> ModelVariant:
    short = _shorten_basename(os.path.basename(path))
    return ModelVariant(
        path=path,
        format=fmt,
        display_name=f"{short}  ({_FORMAT_LABEL.get(fmt, fmt)})",
    )


def model_dir_for(stem: str) -> str:
    """Return the canonical per-model directory for *stem*."""
    return os.path.join(MODELS_DIR, stem)


# ── discovery ───────────────────────────────────────────────────────────────────

def cleanup_onnx_artifacts() -> int:
    """Remove every leftover ``.onnx`` file under ``models/``.

    The convert pipeline used to keep ONNX intermediates around;  this
    helper purges them so only ``.pt`` (PyTorch), ``.engine`` (TensorRT)
    and OpenVINO IR directories remain.  Returns the number of files
    deleted.
    """
    if not os.path.isdir(MODELS_DIR):
        return 0
    deleted = 0
    for root, _dirs, files in os.walk(MODELS_DIR):
        # Don't touch ONNX files that live inside an OpenVINO model dir
        # (OpenVINO never emits one, but be safe).
        if root.endswith("_openvino_model") or root.endswith("_OV_model"):
            continue
        for f in files:
            if f.lower().endswith(".onnx"):
                try:
                    os.remove(os.path.join(root, f))
                    deleted += 1
                except OSError:
                    pass
    return deleted


def discover_models() -> Dict[int, ModelGroup]:
    """
    Return ``{class_id: ModelGroup}`` by scanning per-class model folders.

    Multi-Model-v2 expects five folders under ``models/``:
    ``motorcycle/``, ``rider/``, ``helmet/``, ``footwear/``, and
    ``improper_footwear/``. Each folder may contain ``.pt``, ``.engine``,
    ``.onnx``, or an OpenVINO IR directory.
    """
    _order = {"pt": 0, "engine": 1, "onnx": 2, "openvino": 3}
    groups: Dict[int, ModelGroup] = {}

    for cid, rel_path in DEFAULT_MODEL_FILES.items():
        stem = os.path.splitext(os.path.basename(rel_path))[0]
        model_dir = os.path.join(MODELS_DIR, stem)
        variants: List[ModelVariant] = []
        if os.path.isdir(model_dir):
            for entry in sorted(os.listdir(model_dir)):
                full = os.path.join(model_dir, entry)
                fmt = _detect_format(full)
                if fmt and not any(v.format == fmt for v in variants):
                    variants.append(_make_variant(full, fmt))
        # Also support a direct path from DEFAULT_MODEL_FILES if a .pt exists.
        direct = os.path.join(MODELS_DIR, rel_path)
        fmt = _detect_format(direct) if _exists(direct) else None
        if fmt and not any(v.format == fmt for v in variants):
            variants.append(_make_variant(direct, fmt))
        variants.sort(key=lambda v: _order.get(v.format, 99))
        groups[cid] = ModelGroup(
            stem=stem,
            class_id=cid,
            variants=variants,
        )
    return groups

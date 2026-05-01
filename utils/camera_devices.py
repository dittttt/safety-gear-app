"""
Camera device discovery helpers.

Kept separate from GUI code so both the UI and frame-grabber reuse the same
camera open / probe behavior.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2


CameraTarget = Union[int, str]


@dataclass(frozen=True)
class CameraDevice:
    """One available camera source."""

    key: str
    label: str
    target: CameraTarget
    backend: int = cv2.CAP_ANY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "target": self.target,
            "backend": int(self.backend),
        }


def _candidate_backends() -> Sequence[int]:
    """Return backend probe order for the current platform."""
    if os.name == "nt":
        backends: List[int] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        backends.append(cv2.CAP_ANY)
        # Preserve insertion order while dropping duplicates.
        return tuple(dict.fromkeys(backends))
    return (cv2.CAP_ANY,)


def _unique_preserve(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        val = str(item or "").strip()
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _list_pygrabber_device_names() -> List[str]:
    if os.name != "nt":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore

        graph = FilterGraph()
        names = graph.get_input_devices() or []
        return _unique_preserve([str(n) for n in names])
    except Exception:
        return []


def _list_powershell_device_names() -> List[str]:
    if os.name != "nt":
        return []
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        lines = (proc.stdout or "").splitlines()
        return _unique_preserve(lines)
    except Exception:
        return []


def _list_ffmpeg_dshow_device_names() -> List[str]:
    if os.name != "nt":
        return []
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except Exception:
        return []

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    names: List[str] = []
    for line in text.splitlines():
        m = re.search(r'"([^\"]+)"\s+\(video\)', line)
        if m:
            names.append(m.group(1))
    return _unique_preserve(names)


def _list_windows_camera_names() -> List[str]:
    return _unique_preserve(
        _list_ffmpeg_dshow_device_names()
        + _list_pygrabber_device_names()
        + _list_powershell_device_names()
    )


def _open_camera_with_backend(index: int, backend: int) -> cv2.VideoCapture:
    if backend == cv2.CAP_ANY:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _open_camera_target(target: CameraTarget, backend: int) -> cv2.VideoCapture:
    if isinstance(target, int):
        return _open_camera_with_backend(int(target), backend)
    if backend == cv2.CAP_ANY:
        return cv2.VideoCapture(str(target))
    return cv2.VideoCapture(str(target), backend)


def _try_open_camera(index: int) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Open camera with the first working backend and return (capture, backend)."""
    for backend in _candidate_backends():
        cap = _open_camera_with_backend(index, backend)
        if cap.isOpened():
            return cap, backend
        cap.release()
    return None, None


def open_camera_capture(source: Union[int, str, Dict[str, Any], CameraDevice]) -> cv2.VideoCapture:
    """Open a camera from source metadata produced by :func:`discover_camera_devices`."""
    if isinstance(source, CameraDevice):
        source = source.to_dict()

    if isinstance(source, dict):
        target = source.get("target")
        backend = int(source.get("backend", cv2.CAP_ANY))
        if isinstance(target, int):
            if backend == cv2.CAP_ANY:
                cap, _ = _try_open_camera(int(target))
                if cap is not None:
                    return cap
                return cv2.VideoCapture(int(target))
            return _open_camera_target(int(target), backend)
        if target is not None:
            return _open_camera_target(str(target), backend)

    if isinstance(source, int):
        cap, _ = _try_open_camera(int(source))
        if cap is not None:
            return cap
        return cv2.VideoCapture(int(source))

    if isinstance(source, str):
        if os.name == "nt" and source.startswith("video="):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

    return cv2.VideoCapture(0)


def _probe_capture(cap: cv2.VideoCapture) -> Tuple[bool, int, int]:
    if cap is None or not cap.isOpened():
        return False, 0, 0
    ok = bool(cap.grab())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return ok, width, height


def discover_camera_devices(max_index: int = 8) -> List[CameraDevice]:
    """Probe local camera indexes and return devices that can grab frames."""
    set_log_level = getattr(cv2, "setLogLevel", None)
    get_log_level = getattr(cv2, "getLogLevel", None)
    prev_log_level = None
    if callable(set_log_level):
        try:
            prev_log_level = get_log_level() if callable(get_log_level) else None
            # Use SILENT (not ERROR) — the "Failed list devices for backend
            # dshow" line comes through as WARN from cap_ffmpeg_impl.hpp and
            # ERROR isn't tight enough to suppress it.
            silent_level = getattr(cv2, "LOG_LEVEL_SILENT", None)
            if silent_level is None:
                silent_level = getattr(cv2, "LOG_LEVEL_ERROR", None)
            if silent_level is not None:
                set_log_level(silent_level)
        except Exception:
            prev_log_level = None

    devices: List[CameraDevice] = []
    consecutive_misses = 0
    miss_stop = 1
    windows_name_hints: List[str] = []

    try:
        if os.name == "nt":
            windows_name_hints = _list_windows_camera_names()

        for idx in range(max(0, int(max_index))):
            cap, backend = _try_open_camera(idx)
            if cap is None:
                consecutive_misses += 1
                if devices and consecutive_misses >= miss_stop:
                    break
                continue

            ok, width, height = _probe_capture(cap)
            cap.release()

            if ok:
                label = windows_name_hints[idx] if idx < len(windows_name_hints) else f"Camera {idx}"
                devices.append(
                    CameraDevice(
                        key=f"idx:{idx}",
                        label=label,
                        target=idx,
                        backend=int(backend if backend is not None else cv2.CAP_ANY),
                    )
                )
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if devices and consecutive_misses >= miss_stop:
                    break
    finally:
        if prev_log_level is not None and callable(set_log_level):
            try:
                set_log_level(prev_log_level)
            except Exception:
                pass

    return devices

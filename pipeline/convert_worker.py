"""
Background worker for converting PyTorch (.pt) YOLO models to optimised
inference formats.

Format selection is automatic via ``utils.runtime_check.detect()``:
  * GPU (CUDA):  TensorRT .engine  if tensorrt installed + CUDA GPU present
                 otherwise ONNX    if onnxruntime-gpu available
                 otherwise keep .pt
  * CPU:         OpenVINO          if openvino installed
                 otherwise ONNX    if onnx installed
                 otherwise keep .pt

The worker copies the source ``.pt`` into the target directory, runs
``model.export()`` there, then removes the temporary copy.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import threading
from typing import List, Optional, Tuple

from PyQt5 import QtCore

# ── module-level logger (writes to convert.log next to the models dir) ────────
def _get_logger() -> logging.Logger:
    log = logging.getLogger("convert_worker")
    if not log.handlers:
        try:
            from utils.model_registry import MODELS_DIR
            log_dir = MODELS_DIR
        except Exception:
            log_dir = os.getcwd()
        fh = logging.FileHandler(
            os.path.join(log_dir, "convert.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(fh)
        log.addHandler(logging.StreamHandler())  # also to console
        log.setLevel(logging.DEBUG)
    return log


# Job tuple:  (class_id, pt_path, device_key, imgsz, half)
ConvertJob = Tuple[int, str, str, int, bool]


class ConvertWorker(QtCore.QThread):
    """Run one or more model-conversion jobs in a background thread."""

    ENGINE_TIMEOUT_SEC = 8 * 60
    OPENVINO_TIMEOUT_SEC = 8 * 60
    ONNX_TIMEOUT_SEC = 6 * 60

    # ── signals ────────────────────────────────────────────────────────────
    conversion_started  = QtCore.pyqtSignal(int, str)          # cid, message
    conversion_progress = QtCore.pyqtSignal(int, str)          # cid, message
    conversion_finished = QtCore.pyqtSignal(int, bool, str)    # cid, ok?, path | err
    all_finished        = QtCore.pyqtSignal()

    def __init__(
        self,
        jobs: List[ConvertJob],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = list(jobs)
        self._stop = threading.Event()          # set by request_stop()
        self._trt_failed = False                # skip TRT after first failure
        self._active_proc: Optional[subprocess.Popen] = None  # for cancel

    def request_stop(self) -> None:
        """Ask the worker to abort after the current subprocess finishes."""
        self._stop.set()
        # Also kill any running subprocess immediately
        proc = self._active_proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _target_dir(pt_path: str) -> str:
        """Export alongside the source ``.pt`` (so unified models stay in\n        ``models/unified/`` instead of being moved into ``models/<stem>/``).\n        """
        return os.path.dirname(os.path.abspath(pt_path))

    @staticmethod
    def _best_format(device_key: str) -> str:
        """Return the best available export format for *device_key*."""
        from utils.runtime_check import best_format
        return best_format(device_key)

    @staticmethod
    def _extract_json_line(text: str) -> Optional[str]:
        """Return the last JSON-shaped line from process output."""
        for line in reversed((text or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return line
        return None

    def _export_in_subprocess(
        self,
        model_path: str,
        fmt: str,
        imgsz: int,
        half: bool,
        cwd: str,
        timeout_s: int,
    ) -> str:
        """Run Ultralytics export in a child process and return exported path."""
        py = "\n".join([
            "import json, sys, traceback",
            "from ultralytics import YOLO",
            "mp, fmt, imgsz_s, half_s = sys.argv[1:5]",
            "imgsz = int(imgsz_s)",
            "half = (half_s == '1')",
            "try:",
            "    m = YOLO(mp, task='detect')",
            "    out = m.export(format=fmt, half=half, imgsz=imgsz)",
            "    out_s = str(out) if out else ''",
            "    if not out_s:",
            "        raise RuntimeError('model.export returned empty output')",
            "    print(json.dumps({'ok': True, 'path': out_s}))",
            "except BaseException as exc:",
            "    print(json.dumps({'ok': False, 'error': str(exc), 'tb': traceback.format_exc()}))",
            "    sys.exit(1)",
        ])
        try:
            self._active_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    py,
                    os.path.abspath(model_path),
                    fmt,
                    str(int(imgsz)),
                    "1" if half else "0",
                ],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = self._active_proc.communicate(
                    timeout=max(30, int(timeout_s))
                )
                returncode = self._active_proc.returncode
            except subprocess.TimeoutExpired:
                self._active_proc.kill()
                self._active_proc.communicate(timeout=10)
                raise TimeoutError(
                    f"export timed out after {int(timeout_s)}s (format={fmt})"
                )
            finally:
                self._active_proc = None
        except TimeoutError:
            raise
        except OSError as exc:
            raise RuntimeError(
                f"failed to start export subprocess: {exc}"
            ) from exc

        out_json = self._extract_json_line(stdout)
        if returncode == 0 and out_json:
            try:
                import json
                payload = json.loads(out_json)
                if payload.get("ok") and payload.get("path"):
                    return str(payload["path"])
            except Exception:
                pass

        details = []
        err_json = self._extract_json_line(stderr)
        if out_json:
            details.append(f"stdout-json={out_json}")
        if err_json:
            details.append(f"stderr-json={err_json}")
        if stdout and not out_json:
            details.append(f"stdout={stdout.strip()[-800:]}")
        if stderr and not err_json:
            details.append(f"stderr={stderr.strip()[-800:]}")
        detail_txt = "\n".join(details).strip()

        raise RuntimeError(
            f"export subprocess failed (format={fmt}, code={returncode})"
            + (f"\n{detail_txt}" if detail_txt else "")
        )

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self) -> None:                       # noqa: C901
        log = _get_logger()
        n_jobs = len(self._jobs)
        log.info("ConvertWorker starting — %d job(s)", n_jobs)
        for job_idx, (class_id, pt_path, device_key, imgsz, half) in enumerate(self._jobs, 1):
            if self._stop.is_set():
                log.info("Stop requested — aborting remaining %d job(s)",
                         n_jobs - job_idx + 1)
                break

            basename = os.path.splitext(os.path.basename(pt_path))[0]
            fmt = self._best_format(device_key)
            target_dir = self._target_dir(pt_path)

            # If TRT already failed once, skip straight to ONNX fallback
            if fmt == "engine" and self._trt_failed:
                log.info("Skipping TensorRT for %s (failed earlier) — using ONNX",
                         basename)
                fmt = "onnx"

            # Skip if nothing useful can be produced
            if fmt == "pt":
                self.conversion_finished.emit(
                    class_id, False,
                    "No supported runtime available (install tensorrt or openvino)"
                )
                continue

            tag = f"[{job_idx}/{n_jobs}]"
            log.info("%s Job start: cid=%d  file=%s  device=%s  fmt=%s",
                     tag, class_id, os.path.basename(pt_path), device_key, fmt)

            self.conversion_started.emit(
                class_id,
                f"{tag} Converting {basename} → {fmt.upper()} ({device_key.upper()})…",
            )

            temp_pt: Optional[str] = None       # track copy for cleanup
            t0 = time.monotonic()
            try:
                os.makedirs(target_dir, exist_ok=True)

                # ── 1) Ensure .pt is accessible in target directory ────────
                dest_pt = os.path.join(target_dir, os.path.basename(pt_path))
                if os.path.normpath(pt_path) != os.path.normpath(dest_pt):
                    shutil.copy2(pt_path, dest_pt)
                    temp_pt = dest_pt   # mark copy for cleanup

                # Track pre-existing ONNX (TRT creates an intermediate one)
                onnx_pre = os.path.isfile(
                    os.path.join(target_dir, f"{basename}.onnx"))

                # ── 2) Export via Ultralytics ──────────────────────────────
                msg = f"{tag} Exporting {basename} → {fmt.upper()} (imgsz={imgsz})"
                if fmt == "engine":
                    msg += " — TensorRT build takes 3-8 min, please wait…"
                self.conversion_progress.emit(class_id, msg)
                log.info(msg)

                final_fmt = fmt
                if fmt == "engine":
                    timeout_s = self.ENGINE_TIMEOUT_SEC
                elif fmt == "openvino":
                    timeout_s = self.OPENVINO_TIMEOUT_SEC
                else:
                    timeout_s = self.ONNX_TIMEOUT_SEC
                # Precision policy:
                #   * TensorRT engines benefit massively from FP16 — honour
                #     the user's ``Use FP16`` checkbox at export time.
                #   * Other formats stay FP32; FP16 is applied at runtime
                #     for `.pt` weights via `model.half()`.
                export_half = bool(half) if fmt == "engine" else False

                try:
                    result_path = self._export_in_subprocess(
                        model_path=dest_pt,
                        fmt=fmt,
                        imgsz=imgsz,
                        half=export_half,
                        cwd=target_dir,
                        timeout_s=timeout_s,
                    )
                except BaseException as first_exc:
                    if self._stop.is_set():
                        raise
                    if fmt == "engine":
                        self._trt_failed = True   # skip TRT for all remaining jobs
                        log.warning(
                            "TensorRT export failed for %s — will use ONNX "
                            "for this and all remaining GPU models: %s",
                            basename,
                            first_exc,
                        )
                        self.conversion_progress.emit(
                            class_id,
                            f"{tag} TensorRT failed; falling back to ONNX for {basename}…",
                        )
                        result_path = self._export_in_subprocess(
                            model_path=dest_pt,
                            fmt="onnx",
                            imgsz=imgsz,
                            half=False,
                            cwd=target_dir,
                            timeout_s=self.ONNX_TIMEOUT_SEC,
                        )
                        final_fmt = "onnx"
                    else:
                        raise

                elapsed = time.monotonic() - t0
                log.info("Export OK in %.1fs → %s", elapsed, result_path)

                # ── 3) Cleanup temp .pt ────────────────────────────────────
                if temp_pt and os.path.isfile(temp_pt):
                    os.remove(temp_pt)

                # TensorRT generates an intermediate ONNX – always remove it
                # once the .engine is built (we never load .onnx alongside an
                # .engine; the user explicitly does not want stray .onnx
                # files in the model directories).
                if final_fmt == "engine":
                    onnx_path = os.path.join(target_dir, f"{basename}.onnx")
                    if os.path.isfile(onnx_path):
                        try:
                            os.remove(onnx_path)
                        except OSError:
                            pass

                self.conversion_finished.emit(class_id, True, result_path)

            except BaseException as exc:  # also catches sys.exit() / SystemExit
                elapsed = time.monotonic() - t0
                tb = traceback.format_exc()
                log.error("Job FAILED after %.1fs:\n%s", elapsed, tb)
                # best-effort cleanup
                if temp_pt and os.path.isfile(temp_pt):
                    try:
                        os.remove(temp_pt)
                    except OSError:
                        pass
                self.conversion_finished.emit(class_id, False, f"{exc}\n{tb}")

        self.all_finished.emit()

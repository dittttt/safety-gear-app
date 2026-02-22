"""
Thread 3 – Tracker & Business Logic
====================================
Pulls ``DetectionPacket`` objects, applies spatial association (IoA),
compliance rules (overload, helmet, footwear), occlusion filtering,
and produces annotated ``DisplayPacket`` frames for the GUI.
"""

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore

from config import (
    TARGET_CLASS_IDS,
    CLASS_NAMES,
    CLASS_COLORS_BGR,
    CLASS_MOTORCYCLE,
    CLASS_RIDER,
    CLASS_HELMET,
    CLASS_FOOTWEAR,
    CLASS_IMPROPER_FOOTWEAR,
    COLOR_COMPLIANT_BGR,
    COLOR_NON_COMPLIANT_BGR,
    COLOR_OVERLOAD_BGR,
)
from pipeline.state import PipelineState, Detection, DisplayPacket


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _ioa(child: Tuple[int, int, int, int], parent: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Area of the *child* box."""
    area = _box_area(*child)
    if area <= 0:
        return 0.0
    return _intersection_area(child, parent) / float(area)


# ── Thread ─────────────────────────────────────────────────────────────────────

class TrackerLogicThread(QtCore.QThread):
    """Applies business rules and produces annotated display frames."""

    stats_ready = QtCore.pyqtSignal(dict)

    def __init__(self, state: PipelineState, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._state = state

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        state = self._state
        last_stats = {
            "motorcycles": 0,
            "riders": 0,
            "helmets": 0,
            "footwear": 0,
            "improper_footwear": 0,
            "no_helmet": 0,
            "footwear_compliant": 0,
            "overloaded_motos": 0,
            "overload_riders": 0,
            "compliant_riders": 0,
            "invalid_detections": 0,
        }

        while not state.stop_event.is_set():
            try:
                packet = state.detection_queue.get(timeout=0.05)
            except Exception:
                continue

            if state.stop_event.is_set():
                break

            frame = packet.frame
            dets = packet.detections
            cfg = state.detection_config
            occlusion_thresh = cfg.occlusion_conf_thresh

            if state.is_overlay_enabled() and not packet.detections_fresh:
                annotated = frame
                class_colors = state.get_class_colors()
                for d in dets:
                    color = class_colors.get(
                        d.class_id,
                        CLASS_COLORS_BGR.get(d.class_id, (255, 255, 255)),
                    )
                    cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                    label = CLASS_NAMES.get(d.class_id, "?")
                    if d.class_id == CLASS_RIDER and d.track_id is not None:
                        label = f"{label} ID:{d.track_id}"
                    _draw_label(
                        annotated,
                        f"{label} {d.confidence:.2f}",
                        d.x1,
                        d.y1,
                        color,
                    )
                disp = DisplayPacket(
                    index=packet.index,
                    annotated_frame=annotated,
                    raw_frame=frame,
                    stats=last_stats,
                    timestamp_ms=packet.timestamp_ms,
                )
                state.put_safe(state.display_queue, disp)
                continue

            # ── Separate by class ──────────────────────────────────────────
            motorcycles = [d for d in dets if d.class_id == CLASS_MOTORCYCLE and d.confidence >= occlusion_thresh]
            riders      = [d for d in dets if d.class_id == CLASS_RIDER and d.confidence >= occlusion_thresh]
            helmets     = [d for d in dets if d.class_id == CLASS_HELMET and d.confidence >= occlusion_thresh]
            footwear    = [d for d in dets if d.class_id == CLASS_FOOTWEAR and d.confidence >= occlusion_thresh]
            improper_fw = [d for d in dets if d.class_id == CLASS_IMPROPER_FOOTWEAR and d.confidence >= occlusion_thresh]
            invalid     = [d for d in dets if d.confidence < occlusion_thresh]

            # ── Associate riders ↔ motorcycles (IoA) ──────────────────────
            moto_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in motorcycles]
            moto_rider_map: Dict[int, List[Detection]] = {i: [] for i in range(len(motorcycles))}
            unmatched_riders: List[Detection] = []

            for rider in riders:
                rbox = (rider.x1, rider.y1, rider.x2, rider.y2)
                best_idx, best_ioa = -1, 0.0
                for mi, mbox in enumerate(moto_boxes):
                    score = _ioa(rbox, mbox)
                    if score > best_ioa:
                        best_ioa = score
                        best_idx = mi
                if best_idx >= 0 and best_ioa >= cfg.rider_moto_ioa_thresh:
                    moto_rider_map[best_idx].append(rider)
                else:
                    unmatched_riders.append(rider)

            # ── Associate gear ↔ riders (IoA) ─────────────────────────────
            matched_riders: List[Detection] = []
            for rider_list in moto_rider_map.values():
                matched_riders.extend(rider_list)

            rider_gear: Dict[int, Dict[str, bool]] = {}
            for ri in range(len(matched_riders)):
                rider_gear[ri] = {"helmet": False, "footwear_ok": False, "improper_fw": False}

            rboxes = [(r.x1, r.y1, r.x2, r.y2) for r in matched_riders]

            for h in helmets:
                hbox = (h.x1, h.y1, h.x2, h.y2)
                for ri, rbox in enumerate(rboxes):
                    if _ioa(hbox, rbox) >= cfg.gear_rider_ioa_thresh:
                        rider_gear[ri]["helmet"] = True
                        break

            for fw in footwear:
                fbox = (fw.x1, fw.y1, fw.x2, fw.y2)
                for ri, rbox in enumerate(rboxes):
                    if _ioa(fbox, rbox) >= cfg.gear_rider_ioa_thresh:
                        rider_gear[ri]["footwear_ok"] = True
                        break

            for ifw in improper_fw:
                ifbox = (ifw.x1, ifw.y1, ifw.x2, ifw.y2)
                for ri, rbox in enumerate(rboxes):
                    if _ioa(ifbox, rbox) >= cfg.gear_rider_ioa_thresh:
                        rider_gear[ri]["improper_fw"] = True
                        break

            # ── Check OVERLOAD (> max riders per motorcycle) ──────────────
            overloaded_motos: Set[int] = set()
            for mi, rider_list in moto_rider_map.items():
                if len(rider_list) > cfg.max_riders_per_motorcycle:
                    overloaded_motos.add(mi)

            # ── Build stats dict ──────────────────────────────────────────
            total_riders = len(matched_riders)
            helmeted = sum(1 for g in rider_gear.values() if g["helmet"])
            fw_ok = sum(1 for g in rider_gear.values() if g["footwear_ok"] and not g["improper_fw"])
            overloaded_rider_ids = {
                id(r)
                for mi in overloaded_motos
                for r in moto_rider_map[mi]
            }

            stats = {
                "motorcycles": len(motorcycles),
                "riders": total_riders,
                "helmets": len(helmets),
                "footwear": len(footwear),
                "improper_footwear": len(improper_fw),
                "no_helmet": total_riders - helmeted,
                "footwear_compliant": fw_ok,
                "overloaded_motos": len(overloaded_motos),
                "overload_riders": sum(len(moto_rider_map[mi]) for mi in overloaded_motos),
                "compliant_riders": helmeted,
                "invalid_detections": len(invalid),
            }
            last_stats = stats

            # ── Annotate frame ────────────────────────────────────────────
            annotated = frame
            class_colors = state.get_class_colors()

            if state.is_overlay_enabled():
                # Invalid / occluded detections (dimmed)
                for d in invalid:
                    cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), (100, 100, 100), 1)
                    cv2.putText(
                        annotated, "occluded?", (d.x1, d.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA,
                    )

                # Motorcycles
                for mi, moto in enumerate(motorcycles):
                    is_overloaded = mi in overloaded_motos
                    color = COLOR_OVERLOAD_BGR if is_overloaded else class_colors.get(CLASS_MOTORCYCLE, CLASS_COLORS_BGR[CLASS_MOTORCYCLE])
                    cv2.rectangle(annotated, (moto.x1, moto.y1), (moto.x2, moto.y2), color, 2)
                    label = "Motorcycle"
                    if is_overloaded:
                        label += f" OVERLOAD ({len(moto_rider_map[mi])} riders)"
                    _draw_label(annotated, f"{label} {moto.confidence:.2f}", moto.x1, moto.y1, color)

                # Riders (with compliance status)
                for ri, rider in enumerate(matched_riders):
                    gear = rider_gear.get(ri, {})
                    issues: List[str] = []
                    if not gear.get("helmet"):
                        issues.append("NO HELMET")
                    if gear.get("improper_fw"):
                        issues.append("BAD FOOTWEAR")
                    elif not gear.get("footwear_ok"):
                        issues.append("NO FOOTWEAR")

                    # Check if on overloaded motorcycle
                    on_overloaded = id(rider) in overloaded_rider_ids

                    if on_overloaded:
                        color = COLOR_OVERLOAD_BGR
                        issues.insert(0, "OVERLOAD")
                    elif issues:
                        color = COLOR_NON_COMPLIANT_BGR
                    else:
                        color = COLOR_COMPLIANT_BGR

                    cv2.rectangle(annotated, (rider.x1, rider.y1), (rider.x2, rider.y2), color, 2)
                    tid_str = f" ID:{rider.track_id}" if rider.track_id is not None else ""
                    _draw_label(
                        annotated, f"Rider{tid_str} {rider.confidence:.2f}",
                        rider.x1, rider.y1, color,
                    )
                    if issues:
                        cv2.putText(
                            annotated, " | ".join(issues),
                            (rider.x1, rider.y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA,
                        )

                # Gear boxes
                for d in helmets + footwear + improper_fw:
                    color = class_colors.get(d.class_id, CLASS_COLORS_BGR.get(d.class_id, (255, 255, 255)))
                    cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), color, 1)
                    _draw_label(
                        annotated,
                        f"{CLASS_NAMES.get(d.class_id, '?')} {d.confidence:.2f}",
                        d.x1, d.y1, color,
                    )

            self.stats_ready.emit(stats)

            # Push to display queue (drop oldest if full)
            disp = DisplayPacket(
                index=packet.index,
                annotated_frame=annotated,
                raw_frame=frame,
                stats=stats,
                timestamp_ms=packet.timestamp_ms,
            )

            state.put_safe(state.display_queue, disp)


# ── Drawing utility ────────────────────────────────────────────────────────────

def _draw_label(
    img: np.ndarray, text: str, x: int, y: int,
    color: Tuple[int, int, int],
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    ty = max(0, y - th - 6)
    cv2.rectangle(img, (x, ty), (x + tw + 4, ty + th + 6), color, -1)
    cv2.putText(
        img, text, (x + 2, ty + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA,
    )

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

COLOR_UNKNOWN_BGR: Tuple[int, int, int] = (0, 215, 255)


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


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(pt: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    px, py = pt
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _best_parent_index(
    child: Tuple[int, int, int, int],
    parents: List[Tuple[int, int, int, int]],
    min_ioa: float,
) -> Optional[int]:
    """Return the parent index with highest IoA above *min_ioa*, else None."""
    best_idx = -1
    best_score = 0.0
    for idx, parent_box in enumerate(parents):
        score = _ioa(child, parent_box)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx >= 0 and best_score >= min_ioa:
        return best_idx
    return None


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
            "helmet_unknown": 0,
            "footwear_unknown": 0,
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

            for rider in riders:
                rbox = (rider.x1, rider.y1, rider.x2, rider.y2)
                best_idx = _best_parent_index(
                    rbox,
                    moto_boxes,
                    cfg.rider_moto_ioa_thresh,
                )
                if best_idx is not None:
                    moto_rider_map[best_idx].append(rider)

            # ── Associate gear ↔ riders (IoA) ─────────────────────────────
            matched_riders: List[Detection] = []
            for rider_list in moto_rider_map.values():
                matched_riders.extend(rider_list)

            rider_gear: Dict[int, Dict[str, bool]] = {}
            for ri in range(len(matched_riders)):
                rider_gear[ri] = {"helmet": False, "footwear_ok": False, "improper_fw": False}

            rboxes = [(r.x1, r.y1, r.x2, r.y2) for r in matched_riders]
            matched_helmets: List[Detection] = []
            matched_footwear: List[Detection] = []
            matched_improper_fw: List[Detection] = []

            for h in helmets:
                hbox = (h.x1, h.y1, h.x2, h.y2)
                best_ri = _best_parent_index(hbox, rboxes, cfg.gear_rider_ioa_thresh)
                if best_ri is None and rboxes:
                    hc = _box_center(hbox)
                    for ri, rbox in enumerate(rboxes):
                        if _point_in_box(hc, rbox):
                            best_ri = ri
                            break
                if best_ri is not None:
                    rider_gear[best_ri]["helmet"] = True
                    matched_helmets.append(h)

            for fw in footwear:
                fbox = (fw.x1, fw.y1, fw.x2, fw.y2)
                best_ri = _best_parent_index(fbox, rboxes, cfg.gear_rider_ioa_thresh)
                if best_ri is None and rboxes:
                    fc = _box_center(fbox)
                    for ri, rbox in enumerate(rboxes):
                        if _point_in_box(fc, rbox):
                            best_ri = ri
                            break
                if best_ri is not None:
                    rider_gear[best_ri]["footwear_ok"] = True
                    matched_footwear.append(fw)

            for ifw in improper_fw:
                ifbox = (ifw.x1, ifw.y1, ifw.x2, ifw.y2)
                best_ri = _best_parent_index(ifbox, rboxes, cfg.gear_rider_ioa_thresh)
                if best_ri is None and rboxes:
                    ic = _box_center(ifbox)
                    for ri, rbox in enumerate(rboxes):
                        if _point_in_box(ic, rbox):
                            best_ri = ri
                            break
                if best_ri is not None:
                    rider_gear[best_ri]["improper_fw"] = True
                    matched_improper_fw.append(ifw)

            # ── Check OVERLOAD (> max riders per motorcycle) ──────────────
            overloaded_motos: Set[int] = set()
            for mi, rider_list in moto_rider_map.items():
                if len(rider_list) > cfg.max_riders_per_motorcycle:
                    overloaded_motos.add(mi)

            # ── Build stats dict ──────────────────────────────────────────
            total_riders = len(matched_riders)
            helmeted = sum(1 for g in rider_gear.values() if g["helmet"])
            fw_ok = sum(1 for g in rider_gear.values() if g["footwear_ok"] and not g["improper_fw"])
            fw_unknown = sum(1 for g in rider_gear.values() if (not g["footwear_ok"]) and (not g["improper_fw"]))
            helmet_unknown = total_riders - helmeted
            improper_riders = sum(1 for g in rider_gear.values() if g["improper_fw"])
            overloaded_rider_ids = {
                id(r)
                for mi in overloaded_motos
                for r in moto_rider_map[mi]
            }

            compliant_riders = 0
            for ri, rider in enumerate(matched_riders):
                # Count as compliant only when evidence for both helmet and proper footwear exists.
                if id(rider) in overloaded_rider_ids:
                    continue
                g = rider_gear.get(ri, {})
                if g.get("helmet") and g.get("footwear_ok") and not g.get("improper_fw"):
                    compliant_riders += 1

            stats = {
                "motorcycles": len(motorcycles),
                "riders": total_riders,
                "helmets": len(helmets),
                "footwear": len(footwear),
                "improper_footwear": len(improper_fw),
                "helmet_unknown": helmet_unknown,
                "footwear_unknown": fw_unknown,
                # No explicit "no helmet" detector exists, so avoid defaulting unknown to violation.
                "no_helmet": 0,
                "footwear_compliant": fw_ok,
                "overloaded_motos": len(overloaded_motos),
                "overload_riders": sum(len(moto_rider_map[mi]) for mi in overloaded_motos),
                "compliant_riders": compliant_riders,
                "improper_footwear_riders": improper_riders,
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
                    violations: List[str] = []
                    unknowns: List[str] = []
                    if gear.get("improper_fw"):
                        violations.append("BAD FOOTWEAR")

                    # Missing detections are treated as UNKNOWN, not automatic violations.
                    if not gear.get("helmet"):
                        unknowns.append("HELMET ?")
                    if (not gear.get("footwear_ok")) and (not gear.get("improper_fw")):
                        unknowns.append("FOOTWEAR ?")

                    # Check if on overloaded motorcycle
                    on_overloaded = id(rider) in overloaded_rider_ids

                    if on_overloaded:
                        color = COLOR_OVERLOAD_BGR
                        violations.insert(0, "OVERLOAD")
                    elif violations:
                        color = COLOR_NON_COMPLIANT_BGR
                    elif unknowns:
                        color = COLOR_UNKNOWN_BGR
                    else:
                        color = COLOR_COMPLIANT_BGR

                    cv2.rectangle(annotated, (rider.x1, rider.y1), (rider.x2, rider.y2), color, 2)
                    tid_str = f" ID:{rider.track_id}" if rider.track_id is not None else ""
                    _draw_label(
                        annotated, f"Rider{tid_str} {rider.confidence:.2f}",
                        rider.x1, rider.y1, color,
                    )
                    if violations or unknowns:
                        status_txt = " | ".join(violations + unknowns)
                        status_color = color if violations else COLOR_UNKNOWN_BGR
                        cv2.putText(
                            annotated, status_txt,
                            (rider.x1, rider.y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA,
                        )

                # Gear boxes: always draw raw detections so footwear/improper detections stay visible.
                assoc_ids = {
                    id(d) for d in (matched_helmets + matched_footwear + matched_improper_fw)
                }
                for d in helmets + footwear + improper_fw:
                    color = class_colors.get(d.class_id, CLASS_COLORS_BGR.get(d.class_id, (255, 255, 255)))
                    is_assoc = id(d) in assoc_ids
                    thickness = 2 if is_assoc else 1
                    suffix = "" if is_assoc else " ?"
                    cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
                    _draw_label(
                        annotated,
                        f"{CLASS_NAMES.get(d.class_id, '?')}{suffix} {d.confidence:.2f}",
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

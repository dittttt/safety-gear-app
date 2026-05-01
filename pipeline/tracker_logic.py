"""
Thread 3 – Tracker & Business Logic
====================================
Pulls ``DetectionPacket`` objects, applies spatial association (IoA),
compliance rules (helmet, footwear), occlusion filtering,
and produces annotated ``DisplayPacket`` frames for the GUI.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore

from config import (
    CLASS_NAMES,
    CLASS_COLORS_BGR,
    CLASS_MOTORCYCLE,
    CLASS_RIDER,
    CLASS_HELMET,
    CLASS_NO_HELMET,
    CLASS_FOOTWEAR,
    CLASS_IMPROPER_FOOTWEAR,
    COLOR_COMPLIANT_BGR,
    COLOR_NON_COMPLIANT_BGR,
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


def _nearest_containing_parent(
    child: Tuple[int, int, int, int],
    parents: List[Tuple[int, int, int, int]],
) -> Optional[int]:
    """Return parent index whose box contains the child centre AND whose
    own centre is closest to the child centre. Deterministic tie-break by
    smaller index.

    Used as a fallback when IoA association fails for tiny gear boxes —
    avoids the previous "first containing rider wins" bug where two
    side-by-side riders made gear attribution depend on detection order.
    """
    if not parents:
        return None
    cx, cy = _box_center(child)
    best_idx = -1
    best_dist_sq = float("inf")
    for idx, parent_box in enumerate(parents):
        if not _point_in_box((cx, cy), parent_box):
            continue
        px, py = _box_center(parent_box)
        dx = cx - px
        dy = cy - py
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = idx
    return best_idx if best_idx >= 0 else None


def _box_iou(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    """Standard IoU of two boxes."""
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _box_area(*a) + _box_area(*b) - inter
    return inter / float(union) if union > 0 else 0.0


def _suppress_conflicting_classes(
    dets: List["Detection"],
    pair: Tuple[int, int],
    iou_thresh: float = 0.45,
) -> List["Detection"]:
    """Drop the lower-confidence box when two mutually exclusive classes
    overlap heavily on the same physical object.

    The unified detector occasionally fires both ``helmet`` and
    ``no_helmet`` (or ``footwear`` and ``improper_footwear``) on the same
    head/foot — Ultralytics' NMS is per-class so it never dedups across
    them. The downstream compliance rule resolves the head case by
    ``helmet wins``, which means a true ``no_helmet`` can be silently
    masked by a slightly-higher-confidence false ``helmet``. Removing the
    weaker box at detection time fixes both the visible duplicate boxes
    and the masking bug.
    """
    cid_a, cid_b = pair
    a_dets = [d for d in dets if d.class_id == cid_a]
    b_dets = [d for d in dets if d.class_id == cid_b]
    if not a_dets or not b_dets:
        return dets
    drop = set()
    for da in a_dets:
        if id(da) in drop:
            continue
        abox = (da.x1, da.y1, da.x2, da.y2)
        for db in b_dets:
            if id(db) in drop:
                continue
            bbox = (db.x1, db.y1, db.x2, db.y2)
            if _box_iou(abox, bbox) >= iou_thresh:
                # Drop the lower-confidence detection.
                if da.confidence >= db.confidence:
                    drop.add(id(db))
                else:
                    drop.add(id(da))
                    break  # da is gone — move on
    if not drop:
        return dets
    return [d for d in dets if id(d) not in drop]


# ── Thread ─────────────────────────────────────────────────────────────────────

class TrackerLogicThread(QtCore.QThread):
    """Applies business rules and produces annotated display frames."""

    stats_ready = QtCore.pyqtSignal(dict)
    # Emitted whenever a non-compliant rider is observed.  Payload is a
    # plain dict so Qt can marshal it across threads without issues.
    # Keys: violation_type ("no_helmet" | "improper_footwear"),
    #       motorcycle_id (int|None — tracker id of the bike),
    #       rider_id (int|None — tracker id of the rider),
    #       confidence (float — rider detection confidence).
    violation_detected = QtCore.pyqtSignal(dict)

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

            # Cross-class duplicate suppression (ONCE, before any rule
            # operates on the lists).  Mutually-exclusive classes that
            # overlap heavily on the same physical object would otherwise
            # both reach the rule layer and the helmet/footwear-positive
            # rule would mask true violations.
            if dets:
                dets = _suppress_conflicting_classes(
                    dets, (CLASS_HELMET, CLASS_NO_HELMET))
                dets = _suppress_conflicting_classes(
                    dets, (CLASS_FOOTWEAR, CLASS_IMPROPER_FOOTWEAR))

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
            no_helmets  = [d for d in dets if d.class_id == CLASS_NO_HELMET and d.confidence >= occlusion_thresh]
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
                rider_gear[ri] = {
                    "helmet": False,
                    "no_helmet": False,
                    "footwear_ok": False,
                    "improper_fw": False,
                }

            rboxes = [(r.x1, r.y1, r.x2, r.y2) for r in matched_riders]
            matched_helmets: List[Detection] = []
            matched_no_helmets: List[Detection] = []
            matched_footwear: List[Detection] = []
            matched_improper_fw: List[Detection] = []

            def _attach_to_rider(
                gear_dets: List[Detection],
                gear_key: str,
                sink: List[Detection],
                min_conf: float = 0.0,
            ) -> None:
                """Associate gear detections to their best-matching rider.

                Tries IoA first (best overlap above threshold); falls back
                to a centre-point-in-rider-box test for tiny gear boxes
                that overlap poorly but are clearly inside the rider.
                """
                gate = cfg.gear_rider_ioa_thresh
                for d in gear_dets:
                    if min_conf and d.confidence < min_conf:
                        continue
                    box = (d.x1, d.y1, d.x2, d.y2)
                    best_ri = _best_parent_index(box, rboxes, gate)
                    # Deterministic fallback: nearest rider whose box
                    # actually contains the gear centre. Replaces the
                    # previous "first containing rider wins" loop that
                    # mis-attributed gear on side-by-side riders.
                    if best_ri is None:
                        best_ri = _nearest_containing_parent(box, rboxes)
                    if best_ri is not None:
                        rider_gear[best_ri][gear_key] = True
                        sink.append(d)

            _attach_to_rider(helmets, "helmet", matched_helmets)

            # `no_helmet` is a direct violation class.  Apply a higher conf
            # gate (`no_helmet_min_conf`) than the generic occlusion thresh
            # because false positives here translate directly into reported
            # violations / DB rows.
            no_helmet_min = max(occlusion_thresh, getattr(cfg, "no_helmet_min_conf", 0.40))
            _attach_to_rider(no_helmets, "no_helmet", matched_no_helmets, min_conf=no_helmet_min)

            _attach_to_rider(footwear, "footwear_ok", matched_footwear)
            _attach_to_rider(improper_fw, "improper_fw", matched_improper_fw)

            # ── Build stats dict ──────────────────────────────────────────
            # Helmet semantics — positive evidence wins:
            #   compliant  : `helmet` present (regardless of `no_helmet`)
            #   violation  : `no_helmet` present AND `helmet` absent
            #   unknown    : neither helmet nor no_helmet detected
            total_riders = len(matched_riders)
            helmeted = sum(1 for g in rider_gear.values() if g["helmet"])
            no_helmet = sum(
                1 for g in rider_gear.values()
                if g["no_helmet"] and not g["helmet"]
            )
            helmet_unknown = sum(
                1 for g in rider_gear.values()
                if not g["helmet"] and not g["no_helmet"]
            )

            fw_ok = sum(1 for g in rider_gear.values() if g["footwear_ok"] and not g["improper_fw"])
            improper_riders = sum(1 for g in rider_gear.values() if g["improper_fw"])

            # Footwear: missing both detections means "unknown" — not a strict violation.
            fw_unknown = sum(1 for g in rider_gear.values() if (not g["footwear_ok"]) and (not g["improper_fw"]))

            compliant_riders = 0
            for ri, _rider in enumerate(matched_riders):
                g = rider_gear.get(ri, {})
                # Compliant ONLY if helmet IS present and no improper footwear seen.
                if g.get("helmet") and not g.get("improper_fw"):
                    compliant_riders += 1

            stats = {
                "motorcycles": len(motorcycles),
                "riders": total_riders,
                "helmets": len(matched_helmets),
                "footwear": len(matched_footwear),
                "improper_footwear": len(matched_improper_fw),
                "helmet_unknown": helmet_unknown,
                "footwear_unknown": fw_unknown,
                "no_helmet": no_helmet,
                "footwear_compliant": fw_ok + fw_unknown, # OK or strictly not improper
                "compliant_riders": compliant_riders,
                "improper_footwear_riders": improper_riders,
                "invalid_detections": len(invalid),
            }
            last_stats = stats

            # ── Emit per-motorcycle violation events ──────────────────────
            # We key dedup by motorcycle.track_id (stable across frames),
            # so the GUI / Supabase logger only fires once per offence.
            for mi, rider_list in moto_rider_map.items():
                moto = motorcycles[mi]
                moto_tid = moto.track_id  # may be None if tracker missing

                # Map back rider→gear from matched_riders order.
                for rider in rider_list:
                    try:
                        ri = matched_riders.index(rider)
                    except ValueError:
                        continue
                    g = rider_gear.get(ri, {})
                    rider_tid = rider.track_id
                    # Helmet violation requires POSITIVE no_helmet evidence;
                    # absence of a helmet detection alone (e.g. occlusion)
                    # does not count.  Positive helmet evidence wins over
                    # an overlapping no_helmet detection.
                    if g.get("no_helmet") and not g.get("helmet"):
                        self.violation_detected.emit({
                            "violation_type": "no_helmet",
                            "motorcycle_id": moto_tid,
                            "rider_id": rider_tid,
                            "rider_count": None,
                            "confidence": float(rider.confidence),
                            "timestamp_ms": float(packet.timestamp_ms),
                        })
                    if g.get("improper_fw"):
                        self.violation_detected.emit({
                            "violation_type": "improper_footwear",
                            "motorcycle_id": moto_tid,
                            "rider_id": rider_tid,
                            "rider_count": None,
                            "confidence": float(rider.confidence),
                            "timestamp_ms": float(packet.timestamp_ms),
                        })

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

                # Motorcycles.
                base_moto_color = class_colors.get(
                    CLASS_MOTORCYCLE,
                    CLASS_COLORS_BGR.get(CLASS_MOTORCYCLE, (255, 255, 255)),
                )
                for mi, moto in enumerate(motorcycles):
                    cv2.rectangle(
                        annotated,
                        (moto.x1, moto.y1),
                        (moto.x2, moto.y2),
                        base_moto_color, 2,
                    )
                    label = CLASS_NAMES.get(CLASS_MOTORCYCLE, "Motorcycle")
                    _draw_label(
                        annotated,
                        f"{label} {moto.confidence:.2f}",
                        moto.x1, moto.y1,
                        base_moto_color,
                    )

                # Riders — box always uses the rider class colour.
                # Compliance is shown as a per-segment coloured status line
                # drawn directly below the bounding box:
                #     Helmet ✓ | Footwear ?
                # Each word is coloured independently (green = compliant,
                # red = violation, yellow = unknown) so the user can read
                # status at a glance without decoding tiny icons.
                base_rider_color = class_colors.get(
                    CLASS_RIDER,
                    CLASS_COLORS_BGR.get(CLASS_RIDER, (255, 255, 255)),
                )
                for ri, rider in enumerate(matched_riders):
                    gear = rider_gear.get(ri, {})

                    helmet_status = _gear_status(
                        ok=gear.get("helmet"),
                        bad=gear.get("no_helmet") and not gear.get("helmet"),
                    )
                    fw_status = _gear_status(
                        ok=gear.get("footwear_ok") and not gear.get("improper_fw"),
                        bad=gear.get("improper_fw"),
                    )
                    cv2.rectangle(
                        annotated,
                        (rider.x1, rider.y1),
                        (rider.x2, rider.y2),
                        base_rider_color, 2,
                    )
                    tid_str = f" ID:{rider.track_id}" if rider.track_id is not None else ""
                    _draw_label(
                        annotated,
                        f"Rider{tid_str} {rider.confidence:.2f}",
                        rider.x1, rider.y1, base_rider_color,
                    )

                    segments: List[Tuple[str, str]] = [
                        (_status_label("Helmet",   helmet_status), helmet_status),
                        (_status_label("Footwear", fw_status),     fw_status),
                    ]
                    _draw_status_line(
                        annotated,
                        rider.x1,
                        rider.y2 + 4,
                        segments,
                    )

                # Gear boxes: draw every valid gear detection. Compliance
                # stats still use the matched_* lists above, but the overlay
                # should not hide a detected shoe/helmet just because rider
                # association was imperfect in that frame.
                visible_no_helmets = [
                    d for d in no_helmets if d.confidence >= no_helmet_min
                ]
                gear_dets = helmets + visible_no_helmets + footwear + improper_fw
                for d in gear_dets:
                    color = class_colors.get(d.class_id, CLASS_COLORS_BGR.get(d.class_id, (255, 255, 255)))
                    cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), color, 2)
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


# ── Status icons (drawn beside labels) ─────────────────────────────────────────

# Status string vocabulary used by the icon drawer.
_STATUS_OK      = "ok"
_STATUS_BAD     = "bad"
_STATUS_UNKNOWN = "unknown"

_STATUS_FILL_BGR: Dict[str, Tuple[int, int, int]] = {
    _STATUS_OK:      COLOR_COMPLIANT_BGR,
    _STATUS_BAD:     COLOR_NON_COMPLIANT_BGR,
    _STATUS_UNKNOWN: COLOR_UNKNOWN_BGR,
}


def _gear_status(ok: bool, bad: bool) -> str:
    """Collapse (ok, bad) flags into one of the three status strings."""
    if ok:
        return _STATUS_OK
    if bad:
        return _STATUS_BAD
    return _STATUS_UNKNOWN


def _status_label(name: str, status: str) -> str:
    """Build the per-segment label, e.g. 'Helmet', 'Helmet ✗', 'Helmet ?'."""
    suffix = {
        _STATUS_OK:      "",
        _STATUS_BAD:     " X",
        _STATUS_UNKNOWN: " ?",
    }.get(status, "")
    return f"{name}{suffix}"


def _draw_status_line(
    img: np.ndarray,
    x: int,
    y: int,
    segments: List[Tuple[str, str]],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw ``Helmet | Footwear | …`` below a rider with per-segment colour.

    *segments* is a list of ``(text, status)`` tuples. The pipe separators
    are drawn in white so the eye groups the segments correctly.
    """
    if not segments:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    sep = " | "
    (sep_w, _), _ = cv2.getTextSize(sep, font, font_scale, thickness)

    # Pre-compute text size of the line so we can paint a thin shadow band
    # behind it for legibility on bright backgrounds.
    total_w = 0
    h_max = 0
    for i, (text, _status) in enumerate(segments):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        total_w += tw
        h_max = max(h_max, th)
        if i < len(segments) - 1:
            total_w += sep_w

    pad = 4
    band_y1 = y
    band_y2 = y + h_max + pad * 2
    cv2.rectangle(
        img,
        (x - 2, band_y1),
        (x + total_w + 4, band_y2),
        (0, 0, 0), -1,
    )

    text_y = band_y2 - pad
    cur_x = x
    for i, (text, status) in enumerate(segments):
        color = _STATUS_FILL_BGR.get(status, COLOR_UNKNOWN_BGR)
        cv2.putText(
            img, text, (cur_x, text_y),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )
        (tw, _th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cur_x += tw
        if i < len(segments) - 1:
            cv2.putText(
                img, sep, (cur_x, text_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
            cur_x += sep_w

    return

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from ultralytics import YOLO


def _eval_model_metrics(model_path: str, data_yaml: str) -> dict:
    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    # Ultralytics metrics object exposes overall stats; use robust access with fallbacks.
    metrics = {
        "Model": os.path.basename(model_path),
        "Precision": None,
        "Recall": None,
        "mAP50": None,
        "mAP50-95": None,
    }

    # Newer Ultralytics versions provide metrics.box; fall back if needed.
    box = getattr(results, "box", None)
    if box is not None:
        metrics["Precision"] = float(getattr(box, "p", np.nan))
        metrics["Recall"] = float(getattr(box, "r", np.nan))
        metrics["mAP50"] = float(getattr(box, "map50", np.nan))
        metrics["mAP50-95"] = float(getattr(box, "map", np.nan))
    else:
        # Fallback for older versions
        metrics["Precision"] = float(getattr(results, "p", np.nan))
        metrics["Recall"] = float(getattr(results, "r", np.nan))
        metrics["mAP50"] = float(getattr(results, "map50", np.nan))
        metrics["mAP50-95"] = float(getattr(results, "map", np.nan))

    return metrics


def run_detection_eval(
    model_path: str = "best.pt",
    data_yaml: str = "data.yaml",
    out_csv: str = "detection_metrics.csv",
) -> pd.DataFrame:
    """Run YOLO val() for a single model and save metrics to CSV."""
    resolved_model = resolve_model_path(model_path)
    metrics = _eval_model_metrics(resolved_model, data_yaml)
    df = pd.DataFrame([metrics])
    df.to_csv(out_csv, index=False)

    print("Detection metrics:")
    print(df.to_string(index=False))
    print(f"Saved: {out_csv}")
    return df


def run_detection_eval_multi(
    models_dir: str = "models",
    data_yaml: str = "data.yaml",
    out_csv: str = "detection_metrics.csv",
) -> pd.DataFrame:
    """Run YOLO val() for each .pt in models/ and save combined metrics to CSV."""
    mdir = Path(models_dir)
    if not mdir.exists() or not mdir.is_dir():
        print(f"Models directory not found: {models_dir}")
        return run_detection_eval(model_path="best.pt", data_yaml=data_yaml, out_csv=out_csv)

    model_files = sorted(mdir.glob("*.pt"))
    if not model_files:
        print(f"No .pt files found in {models_dir}")
        return run_detection_eval(model_path="best.pt", data_yaml=data_yaml, out_csv=out_csv)

    rows = []
    for mp in model_files:
        print(f"Evaluating model: {mp}")
        rows.append(_eval_model_metrics(str(mp), data_yaml))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Detection metrics (per model):")
    print(df.to_string(index=False))
    print(f"Saved: {out_csv}")
    return df


def _read_mot_gt(gt_file_path: str) -> pd.DataFrame:
    """Read MOTChallenge GT file into a DataFrame."""
    cols = [
        "FrameId",
        "Id",
        "X",
        "Y",
        "Width",
        "Height",
        "Confidence",
        "ClassId",
        "Visibility",
    ]
    df = pd.read_csv(gt_file_path, header=None, names=cols)
    # Keep only valid gt rows (confidence==1 in MOT format)
    if "Confidence" in df.columns:
        df = df[df["Confidence"] == 1]
    return df


def _collect_tracking_outputs(
    model: YOLO,
    video_path: str,
    tracker_yaml: str,
) -> pd.DataFrame:
    """Run YOLO+BoT-SORT on a video and return tracker outputs in MOT format."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_yaml,
            verbose=False,
        )
        r0 = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(r0, "boxes", None)
        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", None)
            ids = getattr(boxes, "id", None)
            if xyxy is not None:
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
                id_arr = None
                if ids is not None:
                    try:
                        id_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
                    except Exception:
                        id_arr = None

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    track_id = int(id_arr[i]) if id_arr is not None else -1
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    records.append(
                        {
                            "FrameId": frame_idx + 1,  # MOT format is 1-based
                            "Id": track_id,
                            "X": float(x1),
                            "Y": float(y1),
                            "Width": w,
                            "Height": h,
                        }
                    )

        frame_idx += 1

    cap.release()
    return pd.DataFrame.from_records(records)


def evaluate_tracking(
    video_path: str,
    gt_file_path: str,
    model_path: str = "best.pt",
    tracker_yaml: str = "trackers/botsort.yaml",
) -> Optional[pd.DataFrame]:
    """Evaluate tracking metrics using MOTChallenge GT and motmetrics."""
    if not os.path.exists(gt_file_path):
        print(f"GT file not found. Skipping tracking eval: {gt_file_path}")
        return None

    resolved_model = resolve_model_path(model_path)
    model = YOLO(resolved_model)

    print("Running tracking inference...")
    pred_df = _collect_tracking_outputs(model, video_path, tracker_yaml)
    if pred_df.empty:
        print("No tracker outputs collected. Skipping metrics.")
        return None

    print("Loading ground truth...")
    gt_df = _read_mot_gt(gt_file_path)

    acc = mm.utils.compare_to_groundtruth(
        gt_df,
        pred_df,
        distfunc=mm.distances.iou_matrix,
        distth=0.5,
    )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "mostly_tracked", "mostly_lost"],
        name="tracking",
    )

    print("Tracking metrics:")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            "mota": "MOTA",
            "motp": "MOTP",
            "idf1": "IDF1",
            "mostly_tracked": "MT",
            "mostly_lost": "ML",
        },
    ))

    return summary


def main() -> None:
    # Step 1: Detection evaluation (optional)
    run_detection = False
    if run_detection:
        run_detection_eval_multi(
            models_dir="models",
            data_yaml="data.yaml",
            out_csv="detection_metrics.csv",
        )

    # Step 2: Tracking evaluation (skip if GT missing)
    # Update these paths to your test video and MOTChallenge GT file.
    footage_dir = Path("footage")
    video_path = ""
    if footage_dir.exists() and footage_dir.is_dir():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"):
            matches = sorted(footage_dir.glob(ext))
            if matches:
                video_path = str(matches[0])
                break

    if not video_path:
        video_path = "test_video.mp4"

    gt_file_path = "gt/gt.txt"
    if os.path.exists(video_path):
        evaluate_tracking(
            video_path=video_path,
            gt_file_path=gt_file_path,
            model_path="models/rider.pt",
            tracker_yaml="trackers/botsort.yaml",
        )
    else:
        print(f"Video not found. Skipping tracking eval: {video_path}")


def resolve_model_path(model_path: str) -> str:
    """Resolve model path from repo root or models/ folder when best.pt isn't present."""
    if os.path.exists(model_path):
        return model_path

    candidates = []
    models_dir = Path("models")
    if models_dir.exists() and models_dir.is_dir():
        # Preferred names first
        for name in ["best.pt", "rider.pt"]:
            cand = models_dir / name
            if cand.exists():
                candidates.append(str(cand))

        # Any .pt file as fallback
        if not candidates:
            for cand in sorted(models_dir.glob("*.pt")):
                candidates.append(str(cand))

    if candidates:
        chosen = candidates[0]
        print(f"Model not found at '{model_path}'. Using: {chosen}")
        return chosen

    raise FileNotFoundError(f"Model not found: {model_path}. Place it in repo root or models/.")


if __name__ == "__main__":
    main()

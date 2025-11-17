#!/usr/bin/env python3
"""
train.py — Calibration & baseline script (NO model training)

Replaces prior training script. This file *does not* train a neural network.
It runs a short webcam calibration using MediaPipe FaceMesh to estimate per-user
baseline eye aspect ratio (EAR), blink rate, and recommended thresholds used
by the live pipeline for fatigue computation.

Outputs:
 - config.json (in project root) with EAR baseline, blink threshold, blink_rate
 - optionally a short CSV with per-frame EAR samples

Dependencies:
 - mediapipe
 - opencv-python
 - numpy

Run:
  python train.py --duration 15 --out config.json

"""

import time
import json
import argparse
from pathlib import Path
import math

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("mediapipe import failed. Install with `pip install mediapipe`.") from e

# FaceMesh landmark indices for eyes (MediaPipe's indexing)
# These are commonly used sets for EAR calculation
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_idx, image_w, image_h):
    """Compute EAR for one eye using 6 landmarks.
    landmarks: list of mediapipe landmarks
    eye_idx: list of 6 ints
    returns EAR (float) and pixel coords used (for optional drawing)
    """
    def _pt(i):
        lm = landmarks[i]
        return np.array([lm.x * image_w, lm.y * image_h])

    p1 = _pt(eye_idx[0])
    p2 = _pt(eye_idx[1])
    p3 = _pt(eye_idx[2])
    p4 = _pt(eye_idx[3])
    p5 = _pt(eye_idx[4])
    p6 = _pt(eye_idx[5])

    # vertical distances
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    # horizontal distance
    C = np.linalg.norm(p1 - p4)
    if C < 1e-6:
        return 0.0, (p1, p2, p3, p4, p5, p6)
    ear = (A + B) / (2.0 * C)
    return float(ear), (p1, p2, p3, p4, p5, p6)


def collect_baseline(duration=15, cam=0, show=True, save_samples=None):
    """Capture webcam frames for `duration` seconds and compute per-frame EAR.
    Returns dict with statistics and optional samples list.
    """
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 refine_landmarks=True,
                                 max_num_faces=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {cam}")

    t0 = time.time()
    ears = []
    timestamps = []
    blink_count = 0
    closed = False
    # We will detect a blink when EAR drops below a dynamic threshold and then rises

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            now = time.time()
            elapsed = now - t0
            H, W = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            ear_val = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left_ear, _ = eye_aspect_ratio(lm, LEFT_EYE_IDX, W, H)
                right_ear, _ = eye_aspect_ratio(lm, RIGHT_EYE_IDX, W, H)
                ear_val = float((left_ear + right_ear) / 2.0)

                # naive blink detection using instantaneous threshold; we will refine later
                # mark closed when ear below 0.18 (rough), count transitions
                if ear_val < 0.18 and not closed:
                    closed = True
                    blink_count += 1
                if ear_val >= 0.20 and closed:
                    closed = False

                # draw for user feedback
                if show:
                    cv2.putText(frame, f"EAR: {ear_val:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                if show:
                    cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            if ear_val is not None:
                ears.append(ear_val)
                timestamps.append(now)

            if show:
                cv2.imshow("Calibration — look at the camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if elapsed >= duration:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

    # compute stats
    ears_np = np.array(ears) if len(ears) > 0 else np.array([0.0])
    mean_ear = float(np.mean(ears_np))
    median_ear = float(np.median(ears_np))
    std_ear = float(np.std(ears_np))
    duration_sec = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else float(duration)
    blinks_per_min = (blink_count / max(1e-6, duration_sec)) * 60.0

    samples = None
    if save_samples:
        import csv
        samples = save_samples
        with open(save_samples, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["ts", "ear"])
            for ts,val in zip(timestamps, ears):
                w.writerow([ts, val])

    stats = {
        "mean_ear": mean_ear,
        "median_ear": median_ear,
        "std_ear": std_ear,
        "duration_sec": duration_sec,
        "blink_count": blink_count,
        "blinks_per_min": blinks_per_min,
        "samples_saved": samples
    }
    return stats


def suggest_thresholds(stats):
    """Given EAR statistics, suggest thresholds to use in the live pipeline.
    Returns a dict with EAR_threshold and closure_time_estimate and smoothing window.
    """
    mean = stats["mean_ear"]
    std = stats["std_ear"]
    # A common heuristic: blink threshold at 70% of median EAR
    ear_thresh = max(0.10, stats["median_ear"] * 0.7)
    # closure_time default (seconds) — used if you detect long eye closure
    closure_time = 0.25  # default
    # smoothing window (seconds)
    smooth = 5
    return {
        "EAR_threshold": float(ear_thresh),
        "closure_time": float(closure_time),
        "smoothing_window": int(smooth)
    }


def main():
    ap = argparse.ArgumentParser("Calibration for live fatigue pipeline (MediaPipe)")
    ap.add_argument("--duration", type=int, default=15, help="seconds to record calibration")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--out", type=str, default="config.json", help="output config file path")
    ap.add_argument("--save_samples", type=str, default="", help="optional CSV path to save per-frame EAR samples")
    args = ap.parse_args()

    print("Running calibration — keep your face visible to the webcam and behave normally.")
    stats = collect_baseline(duration=args.duration, cam=args.cam, show=True, save_samples=args.save_samples or None)
    print("Calibration stats:")
    for k,v in stats.items():
        print(f"  {k}: {v}")

    thresholds = suggest_thresholds(stats)
    print("Suggested thresholds:")
    for k,v in thresholds.items():
        print(f"  {k}: {v}")

    cfg = {
        "calibration_time": int(time.time()),
        "stats": stats,
        "thresholds": thresholds,
        "notes": "This file is auto-generated. Use these thresholds in your live_coach.py and infer.py. Do NOT train a vision model in this project."
    }

    outp = Path(args.out)
    with open(outp, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved config to {outp.resolve()}")

if __name__ == '__main__':
    main()

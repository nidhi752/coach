#!/usr/bin/env python3
# infer.py — Realtime/video-file + audio-file inference for Cognitive Health Coach
# - Video: webcam index or path to a video file
# - Audio: path to an audio file (wav/mp3); transcribe + prosody once, reused while video runs
# - Overlays fatigue score + recommendation on frames
# - Logs to data/processed/infer_logs.csv

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # kill MSMF priority, prefer other backends

import time
import argparse
import pathlib
import warnings
import csv

warnings.filterwarnings("ignore")

import cv2
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms as T
from pathlib import Path

try:
    import whisper
except Exception as e:
    raise RuntimeError("Whisper import failed. Install via `pip install -U openai-whisper` and ensure ffmpeg on PATH.") from e

try:
    import mediapipe as mp
except Exception:
    mp = None  # we'll raise later if needed

# ---- Paths ----
ROOT   = pathlib.Path(__file__).resolve().parent
DATA   = ROOT / "data"
PROC   = DATA / "processed"
MODELS = ROOT / "models"
CFG    = ROOT / "config.json"
PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ---------- Face model loader (ResNet18 4-class, GPU-aware) ----------
def load_face_model(ckpt=str(MODELS / "face_emotion_resnet18.pt")):
    class FaceWrap:
        EMOS = ["angry", "happy", "neutral", "sad"]  # must match ImageFolder order during training

        def __init__(self, ck, device):
            self.device = device
            self.model = tv.models.resnet18()
            self.model.fc = nn.Linear(self.model.fc.in_features, 4)
            state = torch.load(ck, map_location=device)
            self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()
            self.tfm = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])

        def infer_bgr(self, bgr):
            from PIL import Image
            # BGR -> RGB
            img = Image.fromarray(bgr[:, :, ::-1])
            x = self.tfm(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.model(x).softmax(-1)[0].cpu()
            idx = int(p.argmax())
            return self.EMOS[idx], float(p[idx])

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing model: {ckpt}. Train it first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Face model device: {device}")
    return FaceWrap(ckpt, device)


# ---------- Audio (Whisper + prosody) ----------
def analyze_audio(audio_path, whisper_size="medium"):
    import numpy as np
    import librosa

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load model once
    model = whisper.load_model(whisper_size)

    # Use whisper's loader to ensure expected sampling and dtype
    audio = whisper.load_audio(str(audio_path))  # float32, 16000 Hz mono
    audio = whisper.pad_or_trim(audio)

    # Normalize peak to 0.99 to avoid extremely low-amplitude inputs
    peak = float(max(1e-9, abs(audio).max()))
    audio = audio / peak * 0.99

    # Run transcription (model.transcribe accepts numpy array or path)
    # Use fp16 on GPU when available for speed
    r = model.transcribe(audio, fp16=(torch.cuda.is_available()))
    txt = r.get("text", "").strip()

    # Compute prosody with librosa using the original file to preserve length
    y, sr = librosa.load(audio_path, sr=16000)
    dur = len(y) / sr if sr > 0 else 0.0
    rms = float(librosa.feature.rms(y=y).mean()) if len(y) > 0 else 0.0
    # f0 can be nan if pitch detection fails; guard it
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch = float(np.nanmean(f0))
    except Exception:
        pitch = 0.0
    wps = (len(txt.split()) / dur) if dur > 0 else 0.0

    return {
        "text": txt,
        "rms": rms,
        "pitch": pitch,
        "wps": wps,
        "duration": dur
    }


# ---------- Face detection / crop using MediaPipe (faster + stable) ----------
# If MediaPipe is unavailable, fall back to Haar but warn the user.
mp_face_detector = None
if mp is not None:
    mp_face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def mediapipe_largest_face_crop(bgr, pad_ratio=0.15):
    """Return cropped BGR region for the largest detected face using MediaPipe Face Detection.
    If no face is detected return None.
    """
    global mp_face_detector
    if mp_face_detector is None:
        return None
    h, w = bgr.shape[:2]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_detector.process(img_rgb)
    if not results.detections:
        return None

    # pick detection with largest bounding box area
    best = None
    best_area = 0
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    if best is None:
        return None
    x, y, bw, bh = best
    pad = int(max(bw, bh) * pad_ratio)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1].copy()


# ---------- Fallback Haar crop (kept for compatibility) ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def haar_largest_face_crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    H, W = bgr.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return bgr[y0:y1, x0:x1].copy()


def largest_face_crop(bgr):
    # Prefer MediaPipe if available
    if mp_face_detector is not None:
        crop = mediapipe_largest_face_crop(bgr)
        if crop is not None:
            return crop
    # fallback
    return haar_largest_face_crop(bgr)


# ---------- Fusion & rules (unchanged) ----------
def compute_fatigue(emo_label, sent_score, rms, wps):
    emo_w = {"angry": 0.8, "sad": 0.7, "neutral": 0.3, "happy": 0.2}.get(emo_label, 0.3)
    text_w = (1 - (sent_score + 1) / 2)  # map [-1,1] -> [1,0]
    rms_w = min(1.0, rms * 10.0)         # rough norm
    wps_w = 1.0 - min(1.0, wps / 3.0)    # slower speech -> higher fatigue
    raw = 0.4 * emo_w + 0.3 * text_w + 0.2 * rms_w + 0.1 * wps_w
    return max(0.0, min(1.0, raw))


def recommend(f01, neg_sent_ratio=0.0, screen_time_min=60):
    f = f01 * 100.0
    if f >= 75:
        return "Guided breathing (3 min) and hydrate now."
    if f >= 55:
        return "2-min stretch + 5-min walk."
    if f <= 25 and screen_time_min > 50:
        return "20–20–20 microbreak."
    if neg_sent_ratio > 0.5:
        return "3-line brain dump. Then resume."
    return "You’re fine. Recheck in 15 min."


# ---------- Lightweight text sentiment (no transformers) ----------
def sentiment_from_text(text: str) -> float:
    if not text:
        return 0.0
    text_l = text.lower()

    pos_words = {
        "good", "great", "happy", "excited", "calm", "relaxed",
        "awesome", "fantastic", "nice", "satisfied", "motivated"
    }
    neg_words = {
        "tired", "exhausted", "stressed", "anxious", "sad",
        "angry", "upset", "overwhelmed", "burnt", "burned", "frustrated"
    }

    pos = sum(w in text_l for w in pos_words)
    neg = sum(w in text_l for w in neg_words)

    if pos == 0 and neg == 0:
        return 0.0

    score = (pos - neg) / float(pos + neg)
    return max(-1.0, min(1.0, score))


# ---------- Overlay UI (unchanged) ----------
def draw_overlay(frame, fatigue, rec, emo_label, emo_conf, audio_feats, fps=None):
    h, w = frame.shape[:2]
    # Fatigue bar
    bar_w = int(w * 0.35)
    bar_h = 18
    x0, y0 = 20, 20
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), 1)
    fill = int(bar_w * fatigue)
    color = (0, 200, 0) if fatigue < 0.55 else (0, 180, 255) if fatigue < 0.75 else (0, 0, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.putText(frame, f"Fatigue: {int(round(fatigue * 100))}/100", (x0, y0 + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Text block
    y = y0 + bar_h + 40
    cv2.putText(frame, f"Emotion: {emo_label} ({emo_conf:.2f})", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(frame,
                f"Audio RMS: {audio_feats['rms']:.3f}  Pitch: {audio_feats['pitch']:.1f}  WPS: {audio_feats['wps']:.2f}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24

    # Recommendation (wrap if long)
    rec_lines = []
    line = ""
    for word in rec.split():
        if len(line) + len(word) + 1 > 40:
            rec_lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        rec_lines.append(line)
    for rl in rec_lines:
        cv2.putText(frame, rl, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return frame


# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser("Realtime/Video inference + Audio fusion")
    ap.add_argument("--video", default="0", help="webcam index (e.g., 0) or path to video file")
    ap.add_argument("--audio", default=None, help="path to audio file (wav/mp3) for whisper+prosody")
    ap.add_argument("--text", default="", help="optional text to include in sentiment")
    ap.add_argument("--skip", type=int, default=4, help="process every Nth frame (perf)")
    ap.add_argument("--resize", type=int, default=640, help="resize longer edge to this before face crop")
    ap.add_argument("--save_csv", default=str(PROC / "infer_logs.csv"))
    ap.add_argument("--whisper_model", default="medium", help="whisper model size to use (tiny, small, medium, large)")
    ap.add_argument("--config", default=None, help="path to calibration config.json (optional)")

    args = ap.parse_args()

    # Load models/resources
    face = load_face_model()
    sent_score = sentiment_from_text(args.text) if args.text else 0.0
    print(f"[INFO] Text sentiment score = {sent_score:.3f}")
    audio_feats = {"rms": 0.0, "pitch": 150.0, "wps": 0.0, "duration": 0.0}
    neg_ratio = 1.0 if sent_score < 0 else 0.0
    processed_count = 0

    # Load calibration if present (from --config or default CFG)
    cfg = None
    cfg_path = Path(args.config) if args.config else Path(CFG)
    if cfg_path.exists():
        try:
            import json
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            print(f"[INFO] Loaded calibration from {cfg_path}")
        except Exception:
            cfg = None

    if args.audio:
        print(f"[AUDIO] analyzing {args.audio} …")
        audio_feats = analyze_audio(args.audio, whisper_size=args.whisper_model)
        print(
            f"[AUDIO] text='{audio_feats['text'][:80]}...'  "
            f"rms={audio_feats['rms']:.3f}  pitch={audio_feats['pitch']:.1f}  "
            f"wps={audio_feats['wps']:.2f}"
        )
    # Open video source
    src = 0
    is_cam = False
    if args.video.isdigit():
        src = int(args.video)
        is_cam = True
    elif os.path.exists(args.video):
        src = args.video
        is_cam = False
    else:
        raise FileNotFoundError(f"Video source not found: {args.video}")

    # Force DirectShow for webcam on Windows to avoid MSMF issues
    if is_cam:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.video}")

    # Logging
    csv_path = pathlib.Path(args.save_csv)
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(csv_path, "a", newline="")
    writer = csv.writer(logf)
    if new_file:
        writer.writerow(["ts", "emo", "emo_conf", "rms", "pitch", "wps", "fatigue", "rec"])
        processed_count += 1
        if processed_count % 10 == 0:
            logf.flush()

    # EMA smoothing
    ema = 0.3
    t_last = time.time()
    frame_id = 0
    fps = 0.0
    processed = 0

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Failed to grab frame, stopping.")
            break
        frame_id += 1

        # Basic resize for speed
        H, W = frame.shape[:2]
        scale = args.resize / max(H, W)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(W * scale), int(H * scale)))

        emo_label, emo_conf = "neutral", 1.0

        if frame_id % args.skip == 0:
            # Crop largest face if detected
            face_roi = largest_face_crop(frame)
            roi = face_roi if face_roi is not None else frame

            # Run face emotion
            emo_label, emo_conf = face.infer_bgr(roi)

            # Compute fatigue & recommendation
            fatigue_raw = compute_fatigue(emo_label, sent_score, audio_feats["rms"], audio_feats["wps"])
            ema = 0.8 * ema + 0.2 * fatigue_raw
            fatigue = float(max(0.0, min(1.0, ema)))
            rec = recommend(fatigue, neg_sent_ratio=neg_ratio, screen_time_min=60)

            # FPS calc
            now = time.time()
            dt = now - t_last
            if dt > 0:
                fps = 1.0 / dt
            t_last = now

            # Overlay + display
            frame = draw_overlay(frame, fatigue, rec, emo_label, emo_conf, audio_feats, fps=fps)

            # Log row
            writer.writerow([
                time.time(),
                emo_label,
                f"{emo_conf:.4f}",
                f"{audio_feats['rms']:.6f}",
                f"{audio_feats['pitch']:.2f}",
                f"{audio_feats['wps']:.4f}",
                f"{fatigue:.4f}",
                rec
            ])
            processed += 1

        cv2.imshow("Cognitive Health Coach — Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logf.close()
    print(f"[DONE] Processed {processed} frames. Logs saved to {csv_path}")


if __name__ == "__main__":
    main()

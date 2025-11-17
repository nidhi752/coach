#!/usr/bin/env python3
# live_coach.py — Live webcam + mic + Whisper + MediaPipe FaceMesh + calibrated EAR

import os
import time
import csv
import tempfile
import argparse
import warnings
import pathlib
import math

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms as T
from pathlib import Path


import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import whisper

try:
    import mediapipe as mp
except Exception:
    mp = None

# ---------- Paths ----------
ROOT   = pathlib.Path(__file__).resolve().parent
DATA   = ROOT / "data"
PROC   = DATA / "processed"
MODELS = ROOT / "models"
CFG    = ROOT / "config.json"

for p in [DATA, PROC, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "face_emotion_resnet18.pt"
LOG_CSV    = PROC / "live_logs.csv"


# ---------- Face model (GPU-aware) ----------
class FaceModel:
    EMOS = ["angry", "happy", "neutral", "sad"]  # must match your training order

    def __init__(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing face model at {ckpt_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Face model device: {self.device}")

        self.model = tv.models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.tfm = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def infer_bgr(self, bgr: np.ndarray):
        from PIL import Image
        img = Image.fromarray(bgr[:, :, ::-1])  # BGR -> RGB
        x = self.tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = self.model(x).softmax(-1)[0].cpu()
        idx = int(p.argmax())
        return self.EMOS[idx], float(p[idx])


# ---------- MediaPipe FaceMesh EAR utilities ----------
# indices for eye landmarks (MediaPipe FaceMesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

mp_face_mesh = None
if mp is not None:
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                  max_num_faces=1,
                                                  refine_landmarks=True,
                                                  min_detection_confidence=0.4,
                                                  min_tracking_confidence=0.4)


def compute_ear_from_landmarks(landmarks, W, H):
    def _pt(i):
        lm = landmarks[i]
        return np.array([lm.x * W, lm.y * H])
    try:
        p1 = _pt(LEFT_EYE_IDX[0]); p2 = _pt(LEFT_EYE_IDX[1]); p3 = _pt(LEFT_EYE_IDX[2])
        p4 = _pt(LEFT_EYE_IDX[3]); p5 = _pt(LEFT_EYE_IDX[4]); p6 = _pt(LEFT_EYE_IDX[5])
        A = np.linalg.norm(p2 - p6); B = np.linalg.norm(p3 - p5); C = np.linalg.norm(p1 - p4)
        left_ear = (A + B) / (2.0 * C) if C > 1e-6 else 0.0
        p1 = _pt(RIGHT_EYE_IDX[0]); p2 = _pt(RIGHT_EYE_IDX[1]); p3 = _pt(RIGHT_EYE_IDX[2])
        p4 = _pt(RIGHT_EYE_IDX[3]); p5 = _pt(RIGHT_EYE_IDX[4]); p6 = _pt(RIGHT_EYE_IDX[5])
        A = np.linalg.norm(p2 - p6); B = np.linalg.norm(p3 - p5); C = np.linalg.norm(p1 - p4)
        right_ear = (A + B) / (2.0 * C) if C > 1e-6 else 0.0
        return float((left_ear + right_ear) / 2.0)
    except Exception:
        return 0.0


# ---------- Audio recording (improved) ----------
def record_audio_to_wav(path: str, seconds: int = 6, samplerate: int = 16000,
                        normalize_peak: bool = True, trim_silence: bool = True):
    """Record audio with sounddevice, normalize peak, trim silence, and write wav.
    Returns path to file.
    """
    print(f"[AUDIO] Recording {seconds}s... Speak now.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    y = audio.flatten()

    # Peak normalization
    if normalize_peak:
        peak = max(1e-9, np.max(np.abs(y)))
        y = y / peak * 0.99

    # Trim silence
    if trim_silence:
        try:
            y, _ = librosa.effects.trim(y, top_db=30)
        except Exception:
            pass

    # Ensure at least 0.5s of audio to avoid empty trim results
    if len(y) < 0.5 * samplerate:
        pad_len = int(0.5 * samplerate) - len(y)
        if pad_len > 0:
            y = np.pad(y, (0, pad_len))

    # Write int16 WAV
    y_int16 = (y * 32767).astype(np.int16)
    wavfile.write(path, samplerate, y_int16)
    print("[AUDIO] Recording saved.")
    return path


# ---------- Whisper + prosody (use whisper model passed in) ----------
def analyze_audio_whisper(path: str, whisper_model):
    print(f"[AUDIO] Transcribing with Whisper: {path}")
    # Use whisper loader to get same preprocessing
    audio = whisper.load_audio(str(path))
    audio = whisper.pad_or_trim(audio)

    # peak normalization of array
    peak = float(max(1e-9, abs(audio).max()))
    audio = audio / peak * 0.99

    # transcribe using model instance passed
    result = whisper_model.transcribe(audio, fp16=(torch.cuda.is_available()))
    text = result.get("text", "").strip()

    # prosody using librosa for RMS/pitch
    y, sr = librosa.load(path, sr=16000)
    dur = len(y) / sr if sr > 0 else 0.0
    rms = float(librosa.feature.rms(y=y).mean()) if len(y) > 0 else 0.0
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch = float(np.nanmean(f0))
    except Exception:
        pitch = 0.0
    wps = (len(text.split()) / dur) if dur > 0 else 0.0

    print(f"[AUDIO] text='{text[:120]}...' rms={rms:.4f} pitch={pitch:.1f} wps={wps:.2f}")
    return {"text": text, "rms": rms, "pitch": pitch, "wps": wps, "duration": dur}


# ---------- Replacement for largest_face_crop using MediaPipe detection ----------
mp_face_detector = None
if mp is not None:
    mp_face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)


def mediapipe_face_crop(bgr, pad_ratio=0.15):
    if mp_face_detector is None:
        return None
    h, w = bgr.shape[:2]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_detector.process(img_rgb)
    if not results.detections:
        return None
    best = None
    best_area = 0
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w); y = int(bbox.ymin * h)
        bw = int(bbox.width * w); bh = int(bbox.height * h)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    pad = int(max(bw, bh) * pad_ratio)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1].copy()


# Fallback Haar kept for compatibility
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def haar_face_crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    H, W = bgr.shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return bgr[y0:y1, x0:x1].copy()


def largest_face_crop(bgr):
    crop = None
    if mp_face_detector is not None:
        crop = mediapipe_face_crop(bgr)
        if crop is not None:
            return crop
    return haar_face_crop(bgr)


# ---------- Simple sentiment (unchanged) ----------
def sentiment_from_text(text: str) -> float:
    if not text:
        return 0.0
    text_l = text.lower()
    pos_words = {"good", "great", "happy", "excited", "calm", "relaxed",
                 "awesome", "fantastic", "nice", "satisfied", "motivated"}
    neg_words = {"tired", "exhausted", "stressed", "anxious", "sad",
                 "angry", "upset", "overwhelmed", "burnt", "burned", "frustrated", "depressed"}
    pos = sum(w in text_l for w in pos_words)
    neg = sum(w in text_l for w in neg_words)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / float(pos + neg)
    return max(-1.0, min(1.0, score))


# ---------- Fusion & recommendation (unchanged) ----------
def compute_fatigue(emo_label, sent_score, rms, wps):
    emo_w = {"angry": 0.8, "sad": 0.7, "neutral": 0.3, "happy": 0.2}.get(emo_label, 0.3)
    text_w = (1 - (sent_score + 1) / 2)  # [-1,1] -> [1,0]
    rms_w = min(1.0, rms * 10.0)
    wps_w = 1.0 - min(1.0, wps / 3.0)
    raw = 0.55 * emo_w + 0.25 * text_w + 0.15 * rms_w + 0.05 * wps_w
    return max(0.0, min(1.0, raw))


def recommend(f01: float, neg_sent_ratio=0.0, screen_time_min=60):
    f = f01 * 100.0

    # Highest fatigue first
    if f >= 75:
        return "Guided breathing (5 min) and hydrate now."
    if f >= 60:
        return "Guided breathing (3 min) and hydrate now."
    if f >= 55:
        return "2-min stretch + 5-min walk."
    if f >= 40:
        return "2-min stretch + 5-min walk."

    # Low fatigue but long screen-time
    if f <= 25 and screen_time_min > 50:
        return "20–20–20 microbreak."

    # Strong negativity override
    if neg_sent_ratio > 0.5:
        return "3-line brain dump. Then resume."

    return "You’re fine. Recheck in 15 min."


# ---------- Overlay (unchanged) ----------
def draw_overlay(frame, fatigue, rec, emo_label, emo_conf, audio_feats, fps=None, status_msg=None):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.35)
    bar_h = 18
    x0, y0 = 20, 20
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), 1)
    fill = int(bar_w * fatigue)
    color = (0, 180, 255) if fatigue < 0.55 else (0, 200, 0) if fatigue < 0.75 else (0, 0, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.putText(frame, f"Fatigue: {int(round(fatigue * 100))}/100", (x0, y0 + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    y = y0 + bar_h + 40
    cv2.putText(frame, f"Emotion: {emo_label} ({emo_conf:.2f})", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(frame, f"Audio RMS: {audio_feats['rms']:.3f}  Pitch: {audio_feats['pitch']:.1f}  WPS: {audio_feats['wps']:.2f}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24
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
        cv2.putText(frame, rl, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    if status_msg:
        cv2.putText(frame, status_msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser("Live Cognitive Health Coach")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (default=0)")
    ap.add_argument("--video_interval", type=float, default=5.0, help="seconds between emotion snapshots")
    ap.add_argument("--audio_duration", type=float, default=6.0, help="seconds of mic recording")
    ap.add_argument("--logcsv", type=str, default=str(LOG_CSV))
    ap.add_argument("--whisper_model", type=str, default="medium", help="whisper model name")
    ap.add_argument("--ear_window", type=float, default=1.0, help="seconds smoothing for EAR")
    ap.add_argument("--config", default=None, help="path to calibration config.json (optional)")

    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train it first.")

    face = FaceModel(str(MODEL_PATH))
    print(f"[INFO] Loading Whisper ({args.whisper_model})...")
    whisper_model = whisper.load_model(args.whisper_model)

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

    # CSV log
    csv_path = pathlib.Path(args.logcsv)
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(csv_path, "a", newline="", encoding="utf-8")

    writer = csv.writer(logf)
    if new_file:
        writer.writerow(["ts", "emo", "emo_conf", "ear", "blink_rate", "text", "sent_score",
                         "rms", "pitch", "wps", "fatigue", "rec"])

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {args.cam}")

    state = "video"
    last_state_t = time.time()

    emo_label, emo_conf = "neutral", 1.0
    audio_feats = {"rms": 0.0, "pitch": 150.0, "wps": 0.0}
    sent_score = 0.0
    neg_ratio = 0.0
    fatigue = 0.3
    rec = "Waiting for first cycle..."
    fps = 0.0
    t_last_frame = time.time()

    # EAR buffer for smoothing and blink counting
    ear_buffer = []
    ear_timestamps = []
    blink_count = 0
    closed = False

    # Thresholds
    EAR_threshold = 0.18
    if cfg and 'thresholds' in cfg and 'EAR_threshold' in cfg['thresholds']:
        EAR_threshold = float(cfg['thresholds']['EAR_threshold'])
        print(f"[INFO] Using calibrated EAR_threshold={EAR_threshold:.3f}")

    print("[INFO] Live coach running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Failed to read frame from camera.")
            break

        now = time.time()
        dt_frame = now - t_last_frame
        if dt_frame > 0:
            fps = 1.0 / dt_frame
        t_last_frame = now

        status_msg = f"State: {state}"

        # compute EAR from face mesh every frame (if possible)
        ear_val = 0.0
        if mp is not None and mp_face_mesh is not None:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_face_mesh.process(img_rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    H, W = frame.shape[:2]
                    ear_val = compute_ear_from_landmarks(lm, W, H)
            except Exception:
                ear_val = 0.0

        # update buffer
        ear_buffer.append(ear_val)
        ear_timestamps.append(now)
        # trim buffer to window
        while len(ear_timestamps) > 0 and (now - ear_timestamps[0] > args.ear_window):
            ear_buffer.pop(0); ear_timestamps.pop(0)

        # simple blink detection using EAR_threshold
        if ear_val > 0.0:
            if ear_val < EAR_threshold and not closed:
                closed = True
                blink_count += 1
            if ear_val >= EAR_threshold and closed:
                closed = False

        blink_rate = (blink_count / max(1e-6, max(1.0, (now - (ear_timestamps[0] if ear_timestamps else now))))) * 60.0 if ear_timestamps else 0.0

        # ---- State machine ----
        if state == "video":
            if now - last_state_t >= args.video_interval:
                face_roi = largest_face_crop(frame)
                roi = face_roi if face_roi is not None else frame
                emo_label, emo_conf = face.infer_bgr(roi)
                print(f"[VIDEO] Emotion={emo_label} ({emo_conf:.2f}) EAR={ear_val:.3f} blinks={blink_count}")

                # switch to audio stage
                state = "audio"
                last_state_t = now

        elif state == "audio":
            status_msg = "State: audio (recording...)"
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                record_audio_to_wav(tmp_path, seconds=int(args.audio_duration))
                af = analyze_audio_whisper(tmp_path, whisper_model)
                audio_feats = af
                sent_score = sentiment_from_text(af['text'])
                neg_ratio = 1.0 if sent_score < 0 else 0.0

                # compute fatigue using emo + text + prosody + blink rate
                fatigue_raw = compute_fatigue(emo_label, sent_score, af['rms'], af['wps'])
                # incorporate blink_rate deviations: higher blink_rate -> more fatigue
                # normalize blink_rate: assume baseline 15-20 blinks/min
                blink_w = min(1.0, max(0.0, (blink_rate - 12.0) / 30.0))
                fatigue_raw = min(1.0, fatigue_raw + 0.20 * blink_w)

                fatigue = 0.3 * fatigue + 0.7 * fatigue_raw
                fatigue = float(max(0.0, min(1.0, fatigue)))
                rec = recommend(fatigue, neg_sent_ratio=neg_ratio, screen_time_min=60)

                # Log
                writer.writerow([
                    time.time(),
                    emo_label,
                    f"{emo_conf:.4f}",
                    f"{ear_val:.3f}",
                    f"{blink_rate:.2f}",
                    af['text'],
                    f"{sent_score:.4f}",
                    f"{af['rms']:.6f}",
                    f"{af['pitch']:.2f}",
                    f"{af['wps']:.4f}",
                    f"{fatigue:.4f}",
                    rec
                ])
                logf.flush()

                print(f"[FUSION] emo={emo_label} ({emo_conf:.2f}) sent={sent_score:.2f} blink_rate={blink_rate:.2f} fatigue={fatigue:.2f}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            # back to video cycle
            state = "video"
            last_state_t = time.time()
            status_msg = "State: video (next cycle...)"

        # ---- Overlay & show ----
        avg_ear = float(np.mean(ear_buffer)) if len(ear_buffer) > 0 else 0.0
        rec = recommend(fatigue, neg_sent_ratio=neg_ratio, screen_time_min=60)
        frame = draw_overlay(frame, fatigue, rec, emo_label, emo_conf, audio_feats, fps=fps, status_msg=status_msg)

        cv2.imshow("Live Cognitive Health Coach", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logf.close()
    print(f"[DONE] Logs saved to {csv_path}")


if __name__ == "__main__":
    main()

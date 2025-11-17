# Live Cognitive Health Coach

## Overview

The **Live Cognitive Health Coach** is a real-time system designed to monitor cognitive and emotional fatigue in users by analyzing webcam video, microphone audio, and text inputs. It fuses multiple modalities including facial emotion recognition, eye-blink rate, speech prosody, and sentiment analysis to provide actionable recommendations for reducing fatigue and improving mental well-being.

The system leverages:

* **Face emotion detection** with ResNet18 trained on emotional datasets.
* **Eye Aspect Ratio (EAR)** to detect fatigue via eye blinks.
* **Audio analysis** using OpenAI's Whisper for transcription and prosody extraction.
* **Sentiment scoring** to detect negative emotional content in speech.
* **Real-time recommendations** based on fused metrics.

---

## Features

* Real-time video and audio processing.
* Facial emotion recognition: `angry`, `happy`, `neutral`, `sad`.
* Eye-blink detection and EAR computation using MediaPipe FaceMesh.
* Audio recording, normalization, and trimming.
* Speech transcription using Whisper model.
* Sentiment analysis of spoken text.
* Fatigue computation by fusing emotion, blink rate, speech metrics, and sentiment.
* Actionable recommendations: stretches, microbreaks, guided breathing.
* Visual overlay displaying metrics in real-time.
* CSV logging of all metrics for offline analysis.

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd live_coach
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure you have a trained face emotion model at `models/face_emotion_resnet18.pt`. You can train one or download a pre-trained model.

---

## Usage

### Command-line options:

```bash
python live_coach.py [--cam 0] [--video_interval 5] [--audio_duration 6] [--logcsv path/to/log.csv] [--whisper_model medium] [--ear_window 1.0] [--config path/to/config.json]
```

* `--cam`: Webcam index.
* `--video_interval`: Seconds between emotion snapshots.
* `--audio_duration`: Seconds of microphone recording.
* `--logcsv`: Path to CSV log file.
* `--whisper_model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`).
* `--ear_window`: Smoothing window for EAR computation.
* `--config`: Path to optional calibration config.

### Example:

```bash
python live_coach.py --cam 0 --video_interval 5 --audio_duration 6 --logcsv data/processed/live_logs.csv --whisper_model medium
```

---

## How it Works

### Video Processing

1. Capture webcam frames.
2. Detect face using **MediaPipe Face Detection**; fallback to Haar cascades.
3. Crop largest face and pass through **ResNet18** emotion classifier.
4. Compute **EAR** for blink detection and fatigue estimation.

### Audio Processing

1. Record audio for specified duration.
2. Normalize peak and trim silence using **librosa**.
3. Transcribe speech using **Whisper**.
4. Compute prosody metrics: RMS, pitch, words per second.
5. Compute sentiment score from transcribed text.

### Fusion & Fatigue Computation

* Combine emotion confidence, sentiment score, RMS, WPS, and blink rate.
* Normalize blink rate (baseline 12-20 blinks/min).
* Compute weighted fatigue score:

  * Emotion: 55%
  * Sentiment: 25%
  * RMS: 15%
  * WPS: 5%
* Recommendations generated based on fatigue thresholds and sentiment.

### Recommendations Logic

* `f >= 75`: Guided breathing (5 min) + hydrate
* `f >= 60`: Guided breathing (3 min) + hydrate
* `f >= 55`: 2-min stretch + 5-min walk
* `f >= 40`: 2-min stretch + 5-min walk
* `f <= 25` and long screen-time: 20-20-20 microbreak
* Strong negative sentiment: 3-line brain dump
* Otherwise: Recheck in 15 min

---

## Output

* Real-time video overlay displaying:

  * Fatigue bar
  * Emotion label + confidence
  * Audio metrics (RMS, pitch, WPS)
  * Recommendations
  * FPS and status

* CSV log with columns:

  ```csv
  ts, emo, emo_conf, ear, blink_rate, text, sent_score, rms, pitch, wps, fatigue, rec
  ```

---

## File Structure

```
live_coach/
│
├─ live_coach.py        # Main script
├─ requirements.txt     # Dependencies
├─ models/             # Trained models
│   └─ face_emotion_resnet18.pt
├─ data/
│   └─ processed/      # CSV logs
├─ config.json         # Optional thresholds/config
└─ README.md
```

---

## Dependencies

* Python 3.8+
* OpenCV
* Torch & Torchvision
* Mediapipe
* Sounddevice
* Librosa
* Whisper
* Scipy
* Numpy
* PIL
* Scikit-learn

---

## Notes

* **GPU Recommended** for real-time performance with Whisper and ResNet.
* EAR thresholds can be calibrated using `config.json`.
* Logs can be used for offline analysis or training fatigue models.

---

## License

MIT License

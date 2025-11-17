# Live Cognitive Health Coach — NLP-Driven Multimodal Fatigue Estimation

## Overview

The **Live Cognitive Health Coach** is fundamentally an **NLP-centered, multimodal cognitive‑state estimation system**.  
While it integrates computer vision (facial emotion recognition, eye‑blink analysis), its **primary intelligence comes from NLP** — speech‑to‑text processing, linguistic sentiment interpretation, paralinguistic prosody analysis, and natural‑language recommendation generation.

NLP shapes the model’s understanding of user intent, emotional nuance, and cognitive load, making it the backbone of the entire system.

---

## 🔥 Why This Is an NLP Project

Although multimodal, the system’s *decisive* features depend on NLP:

### **1. Automatic Speech Recognition (ASR) — Whisper**
Audio → token sequences → text  
This provides the linguistic substrate for all downstream inference.

### **2. Sentiment & Semantic Polarity Analysis**
The transcript is analyzed for lexical affect (positive/negative/stress indicators).  
This semantic polarity directly modifies the fatigue score.

### **3. Speech Prosody as Para‑Linguistic NLP**
Prosodic features extracted from speech:
- RMS (energy stability)
- Pitch (intonation)
- WPS (speech rate)

These correlate with:
- mental fatigue  
- cognitive load  
- emotional dulling  

Prosody is a recognized **NLP task under paralinguistics**.

### **4. Natural Language Generation (NLG)**
The system generates:
- stress‑reduction instructions  
- microbreak suggestions  
- fatigue interventions  

This is rule‑based NLG tuned for real‑time guidance.

### **5. Multimodal Fusion with Linguistic Dominance**
The linguistic (text + prosody) signal has the *strongest* influence in the final fatigue estimation.

---

## Features

### 🔹 **NLP Features**
- Whisper‑based ASR  
- Sentiment polarity scoring  
- Prosody extraction (RMS, Pitch, WPS)  
- NLP‑driven fatigue inference  
- NLG‑based recommendation engine  

### 🔹 **Vision Features (Secondary Modalities)**
- Facial emotion recognition via ResNet18  
- MediaPipe Eye Aspect Ratio (EAR)  
- Blink‑rate analysis  

### 🔹 **Fusion Engine**
Combines NLP, audio paralinguistics, and computer vision into a single fatigue score.

---

## Pipeline Breakdown (NLP‑First Perspective)

### **A. Speech Pipeline**
1. Record audio  
2. Whisper transforms speech → text  
3. Sentiment analysis on transcript  
4. Compute prosody (RMS, pitch, WPS)  
5. NLP features weighed into fatigue score  

### **B. Vision Pipeline**
1. Face detection  
2. Emotion classification  
3. EAR + blink rate  
4. Supplementary emotional cues  

### **C. Fusion + NLG**
The system combines linguistic + paralinguistic + visual signals to:
- compute fatigue  
- generate contextual natural‑language recommendations  

---

## Architecture Diagram (Conceptual)

```
Audio → Whisper ASR → Text → Sentiment → NLP Fatigue Features
                     ↘ Prosody (RMS/Pitch/WPS) ↗

Video → Face Emotion → Visual Features
         EAR/Blinks  → Physiological Features

All → Fusion Engine → Fatigue Score → NLG Advice
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Live NLP + Vision Cognitive Coach
```bash
python live_coach.py --cam 0 --whisper_model medium
```

### Video/Audio Inference
```bash
python infer.py --video 0 --audio input.wav
```

---

## Outputs

Logged CSV includes:

```
timestamp,
emotion_label,
emotion_confidence,
EAR,
blink_rate,
transcript,
sentiment_score,
RMS,
pitch,
WPS,
fatigue_score,
recommendation
```

This supports NLP downstream tasks like:
- stress modeling  
- semantic drift tracking  
- prosody‑fatigue correlation  

---

## Conclusion

This project is best understood as a **multimodal NLP system augmented with vision**, not the other way around.  
The **linguistic layer** — text, sentiment, prosody, NLG — drives the cognitive inference, while computer vision provides complementary signals.

The result is a robust, real‑time cognitive health coach capable of interpreting both *what* users say and *how* they say it.


# 📄 `README.md`

# Automated Analysis of Football Match Commentary from Audio Signals

## 📜 Overview

This project develops an **end-to-end audio processing pipeline** for football (soccer) match commentary.  
Starting from raw audio, the system:

- Transcribes speech using OpenAI's Whisper model
- Extracts prosodic features (pitch and energy) using librosa
- Scores and detects emphasized segments based on prosodic analysis
- Saves enriched structured outputs for further analysis

The final goal is to **extract key events** (goals, fouls, etc.) and **generate match summaries** automatically.

---

## ✅ Current Features

- **Transcription**:  
  High-quality transcription of football audio using Whisper.

- **Prosody Extraction**:  
  Extraction of pitch (fundamental frequency) and energy (loudness) for each segment.

- **Emphasis Detection**:  
  Segments are automatically scored and marked as emphasized based on normalized prosody.

- **Structured Output**:  
  All results are saved in `.json` and `.txt` formats for further analysis.

---

## 📈 What’s Next

| Phase                  | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| **Event Extraction**   | Identify goals, fouls, substitutions from transcripts    |
| **Summarization**      | Summarize match commentary into short readable summaries |
| **Visualization**      | Graphs and UI to visualize emphasis, match moments       |
| **Audio Highlighting** | Export only the emphasized segments as clips             |

---

## 📁 Project Structure

```bash
football_audio_analysis/
├── data/
│   └── raw/                   # Raw input audio files (.mp3, .wav)
├── outputs/
│   └── transcripts/           # Transcription and emphasis outputs
├── src/
│   ├── asr/
│   │   └── whisper_transcriber.py  # Transcription + prosody extraction
│   ├── emphasis/
│   │   └── scorer.py               # Emphasis scoring module
│   └── main.py                     # Main pipeline entry point
├── requirements.txt
└── README.md
```

````

---

## 🚀 How to Run

1. **Install Requirements**

```bash
pip install -r requirements.txt
```

2. **Run the Main Script**

```bash
python src/main.py
```

3. **Outputs**

- `outputs/transcripts/sample.json` → full transcript + prosody
- `outputs/transcripts/sample.txt` → readable text
- `outputs/transcripts/sample_emphasis.json` → segments with emphasis scores

---

## 🛠 Requirements

- Python 3.11+
- [Whisper](https://github.com/openai/whisper) (`pip install git+https://github.com/openai/whisper.git`)
- librosa
- numpy
- torch

---

## 📌 Notes

- Currently optimized for **CPU** usage (no GPU-specific code yet).
- Works well with 2-5 minute audio clips.
- Emphasis detection is prosody-only; semantic context detection will be added next.

---

## 🤝 Future Improvements

- Use domain-tuned NLP models for football-specific event detection
- Fine-tune summarization models on match reports
- Build a lightweight visualization dashboard with Streamlit
````

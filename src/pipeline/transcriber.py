# src/pipeline/transcriber.py

import whisper
import time
import os
import json
import librosa
import numpy as np
from src.utils.io import save_json, save_plaintext
from src.utils.io import format_timestamp


def transcribe_audio(file_path: str, model_size: str = "base", save_to: str = None) -> dict:
    print(f"⏳ Loading Whisper model '{model_size}' (this may take a moment)...")
    model = whisper.load_model(model_size)

    print(f"🎧 Transcribing: {file_path}")
    start = time.time()
    result = model.transcribe(file_path)
    end = time.time()
    print(f"✅ Transcription complete in {end - start:.2f} seconds.")

    y, sr = librosa.load(file_path, sr=16000)
    enriched_segments = []

    for seg in result["segments"]:
        start_sec, end_sec = seg["start"], seg["end"]
        start_sample, end_sample = int(start_sec * sr), int(end_sec * sr)
        y_seg = y[start_sample:end_sample]

        # Pitch
        pitches, mags = librosa.piptrack(y=y_seg, sr=sr)
        pitch_vals = pitches[mags > np.median(mags)]
        pitch = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0.0

        # Energy
        energy = np.mean(librosa.feature.rms(y=y_seg))

        enriched_segments.append({
            "start": start_sec,
            "end": end_sec,
            "text": seg["text"],
            "pitch": round(float(pitch), 2),
            "energy": round(float(energy), 6)
        })

    output = {
        "text": result["text"],
        "segments": enriched_segments,
        "language": result["language"]
    }

    if save_to:
        save_json(output, save_to + ".json")
        save_plaintext(enriched_segments, save_to + ".txt")

    return output

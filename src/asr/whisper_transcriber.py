# src/asr/whisper_transcriber.py

import whisper
import time
import os
import json
import librosa
import numpy as np

def transcribe_audio(file_path: str, model_size: str = "base", save_to: str = None) -> dict:
    """
    Transcribes the given audio file using Whisper and enriches each segment with prosody features (pitch, energy).

    Args:
        file_path (str): Path to the audio file.
        model_size (str): One of "tiny", "base", "small", etc.
        save_to (str): Output path prefix (no extension). Saves .json and .txt.

    Returns:
        dict: Transcript containing text, enriched segments, and language.
    """
    print(f"⏳ Loading Whisper model '{model_size}' (this may take a moment)...")
    model = whisper.load_model(model_size)

    print(f"🎧 Transcribing: {file_path}")
    start = time.time()
    result = model.transcribe(file_path)
    end = time.time()
    print(f"✅ Transcription complete in {end - start:.2f} seconds.")

    # Load audio for prosody extraction
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

    # Save outputs
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

        with open(save_to + ".json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        with open(save_to + ".txt", "w", encoding="utf-8") as f:
            f.write("FULL TRANSCRIPT\n\n")
            for seg in enriched_segments:
                ts_start = format_timestamp(seg["start"])
                ts_end = format_timestamp(seg["end"])
                f.write(f"[{ts_start} - {ts_end}] {seg['text']}\n")

        print(f"💾 Saved to {save_to}.json and {save_to}.txt")

    return output


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}"

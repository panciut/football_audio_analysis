# src/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.asr.whisper_transcriber import transcribe_audio

if __name__ == "__main__":
    AUDIO_PATH = "data/raw/sample2.mp3"
    OUTPUT_PREFIX = "outputs/transcripts/sample"

    result = transcribe_audio(
        file_path=AUDIO_PATH,
        model_size="base",     # Change to "small", "medium", etc. as needed
        save_to=OUTPUT_PREFIX
    )

    print("\n📝 Full Transcript:\n")
    print(result["text"])

    print("\n📍 Segments:\n")
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

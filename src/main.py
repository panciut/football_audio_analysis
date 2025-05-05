# src/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AUDIO_PATH, OUTPUT_PREFIX, WHISPER_MODEL
from src.pipeline.transcriber import transcribe_audio
from src.pipeline.emphasis import score_emphasis
from src.pipeline.segmenter import segment_sentences
from src.utils.io import save_json

if __name__ == "__main__":
    # 1. Transcribe
    result = transcribe_audio(file_path=AUDIO_PATH, model_size=WHISPER_MODEL, save_to=OUTPUT_PREFIX)

    print("\n📝 Transcript Preview:\n")
    print(result["text"][:500], "...")

    # 2. Sentence Segmentation
    sentences = segment_sentences(result["text"])
    result["sentences"] = sentences

    print("\n🪛 Sentence Segmentation:\n")
    for s in sentences[:5]:
        print(f"→ {s['text']}")

    # 3. Emphasis Scoring
    result["segments"] = score_emphasis(result["segments"])

    print("\n📍 Emphasis Detection:\n")
    for seg in result["segments"]:
        flag = "🔥" if seg.get("emphasized", False) else "  "
        print(f"{flag} [{seg['start']:.2f}-{seg['end']:.2f}] pitch={seg['pitch']}Hz, energy={seg['energy']}, "
              f"score={seg['emphasis_score']} → {seg['text']}")

    # 4. Save output
    save_json(result, OUTPUT_PREFIX + "_emphasis.json")

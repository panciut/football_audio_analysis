# src/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.asr.whisper_transcriber import transcribe_audio
from src.emphasis.scorer import score_emphasis

if __name__ == "__main__":
    AUDIO_PATH = "data/raw/sample.mp3"
    OUTPUT_PREFIX = "outputs/transcripts/sample"

    result = transcribe_audio(
        file_path=AUDIO_PATH,
        model_size="small", # Options: "tiny", "base", "small", "medium", "large" 
        save_to=OUTPUT_PREFIX
    )

    print("\n📝 Transcript Preview:\n")
    print(result["text"][:500], "...")

    # Apply emphasis scoring
    segments = score_emphasis(result["segments"])

    print("\n📍 Emphasis Detection:\n")
    for seg in segments:
        flag = "🔥" if seg["emphasized"] else "  "
        print(f"{flag} [{seg['start']:.2f}-{seg['end']:.2f}] pitch={seg['pitch']}Hz, energy={seg['energy']}, score={seg['emphasis_score']} → {seg['text']}")

    # (Optional) Save updated output
    import json
    with open(OUTPUT_PREFIX + "_emphasis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Emphasis results saved to {OUTPUT_PREFIX}_emphasis.json")

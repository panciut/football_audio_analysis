# src/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    AUDIO_PATH, OUTPUT_PREFIX, SHOULD_TRANSCRIBE, WHISPER_MODEL, ENABLE_CANONICAL_SUBSTITUTION,
    LOG_TRANSCRIPT_PREVIEW, LOG_SENTENCES, LOG_NER, LOG_EMPHASIS
)
from src.pipeline.transcriber import transcribe_audio
from src.pipeline.emphasis import score_emphasis
from src.pipeline.segmenter import segment_sentences
from src.pipeline.ner import extract_named_entities
from src.utils.text import apply_canonical_substitutions
from src.utils.io import save_json

if __name__ == "__main__":
   # 1. Transcription (optional)
    if SHOULD_TRANSCRIBE:
        result = transcribe_audio(file_path=AUDIO_PATH, model_size=WHISPER_MODEL, save_to=OUTPUT_PREFIX)
    else:
        from src.utils.io import load_json
        result = load_json(OUTPUT_PREFIX + ".json")
        print(f"📂 Loaded existing transcript from {OUTPUT_PREFIX}.json")

    if LOG_TRANSCRIPT_PREVIEW:
        print("\n📝 Transcript Preview:\n")
        print(result["text"][:500], "...")

    # 2. Sentence Segmentation
    sentences = segment_sentences(result["text"])
    result["sentences"] = sentences
    save_json(result, OUTPUT_PREFIX + "_step2_sentences.json")

    if LOG_SENTENCES:
        print("\n🪛 Sentence Segmentation:\n")
        for s in sentences[:5]:
            print(f"→ {s['text']}")

    # 3. Named Entity Recognition
    result["sentences"] = extract_named_entities(result["sentences"])
    save_json(result, OUTPUT_PREFIX + "_step3_entities.json")

    if LOG_NER:
        print("\n🔍 Named Entity Recognition (NER):\n")
        for s in result["sentences"]:
            ents = ", ".join(f"{e['text']} ({e['label']})" for e in s["entities"])
            print(f"→ {s['text']}\n   ↳ Entities: {ents}")

    # 4. Canonical Substitution (optional)
    if ENABLE_CANONICAL_SUBSTITUTION:
        result["sentences"], result["segments"] = apply_canonical_substitutions(result["sentences"], result["segments"])
        save_json(result, OUTPUT_PREFIX + "_step4_substituted.json")

    # 5. Emphasis Scoring
    result["segments"] = score_emphasis(result["segments"])
    save_json(result, OUTPUT_PREFIX + "_step5_emphasis.json")

    if LOG_EMPHASIS:
        print("\n📍 Emphasis Detection:\n")
        for seg in result["segments"][:5]:
            flag = "🔥" if seg.get("emphasized", False) else "  "
            print(f"{flag} [{seg['start']:.2f}-{seg['end']:.2f}] pitch={seg['pitch']}Hz, energy={seg['energy']}, "
                  f"score={seg['emphasis_score']} → {seg['text']}")

    # 6. Final Output
    save_json(result, OUTPUT_PREFIX + "_final.json")

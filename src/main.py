# src/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    AUDIO_PATH, LOG_EVENTS, OUTPUT_PREFIX, SHOULD_TRANSCRIBE, WHISPER_MODEL, ENABLE_CANONICAL_SUBSTITUTION,
    LOG_TRANSCRIPT_PREVIEW, LOG_SENTENCES, LOG_NER, LOG_EMPHASIS
)

from src.pipeline.transcriber import transcribe_audio
from src.pipeline.emphasis import score_emphasis
from src.pipeline.segmenter import segment_sentences
from src.pipeline.ner import extract_named_entities
from src.pipeline.event_zero_shot import detect_events_zero_shot  # Updated import
from src.utils.text import apply_canonical_substitutions
from src.utils.io import save_json, load_json

if __name__ == "__main__":
    # 1. Transcription
    if SHOULD_TRANSCRIBE:
        result = transcribe_audio(file_path=AUDIO_PATH, model_size=WHISPER_MODEL, save_to=OUTPUT_PREFIX)
    else:
        result = load_json(OUTPUT_PREFIX + ".json")
        print(f"📂 Loaded existing transcript from {OUTPUT_PREFIX}.json")

    if LOG_TRANSCRIPT_PREVIEW:
        print("\n📝 Transcript Preview:\n")
        print(result["text"][:500], "...")

    # 2. Sentence Segmentation
    result["sentences"] = segment_sentences(result["text"], result["segments"])
    save_json(result["sentences"], OUTPUT_PREFIX + "_step2_sentences.json")

    if LOG_SENTENCES:
        print("\n🪛 Sentence Segmentation:\n")
        for s in result["sentences"][:5]:
            print(f"→ {s['text']}")

    # 3. Named Entity Recognition
    result["sentences"] = extract_named_entities(result["sentences"])
    entities_only = [{"text": s["text"], "entities": s["entities"]} for s in result["sentences"]]
    save_json(entities_only, OUTPUT_PREFIX + "_step3_entities.json")

    if LOG_NER:
        print("\n🔍 Named Entity Recognition (NER):\n")
        for s in result["sentences"]:
            ents = ", ".join(f"{e['text']} ({e['label']})" for e in s["entities"])
            print(f"→ {s['text']}\n   ↳ Entities: {ents}")

    # 4. Canonical Substitution (optional)
    if ENABLE_CANONICAL_SUBSTITUTION:
        result["sentences"], result["segments"] = apply_canonical_substitutions(result["sentences"], result["segments"])
        substituted = {
            "sentences": result["sentences"],
            "segments": result["segments"]
        }
        save_json(substituted, OUTPUT_PREFIX + "_step4_substituted.json")

    # 5. Emphasis Scoring
    result["segments"] = score_emphasis(result["segments"])
    save_json(result["segments"], OUTPUT_PREFIX + "_step5_emphasis.json")

    if LOG_EMPHASIS:
        print("\n📍 Emphasis Detection:\n")
        for seg in result["segments"][:5]:
            flag = "🔥" if seg.get("emphasized", False) else "  "
            print(f"{flag} [{seg['start']:.2f}-{seg['end']:.2f}] pitch={seg['pitch']}Hz, energy={seg['energy']}, "
                  f"score={seg['emphasis_score']} → {seg['text']}")

    # 6. Event Extraction (Zero-Shot)
    print("\n🚦 Running event detection on context windows...")
    events = detect_events_zero_shot(result["sentences"])
    result["events"] = events  # assign only after detection is done
    print(f"✅ Detected {len(events)} events.")

#   Save after all processing
    save_json(events, OUTPUT_PREFIX + "_step6_events.json")


    if LOG_EVENTS:
        print("\n📅 Event Extraction:\n")
        for e in result["events"]:
            print(f"→ [{e['type'].upper()}] {e['text']}\n   ↳ Confidence: {e['confidence']}")

    # Final output with everything
    save_json(result, OUTPUT_PREFIX + "_final.json")

# src/pipeline/events.py

import yaml
import re
from pathlib import Path


def load_event_schema(schema_path="data/event_schema.yaml"):
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]


def detect_events(sentences, schema_path="data/event_schema.yaml"):
    schema = load_event_schema(schema_path)
    detected_events = []

    for s in sentences:
        sentence_text = s["text"].lower()
        entities_by_type = {ent["label"]: ent for ent in s.get("entities", [])}

        for event in schema:
            matched_trigger = any(re.search(rf"\b{re.escape(word)}\b", sentence_text)
                                  for word in event["trigger_words"])
            if not matched_trigger:
                continue

            has_required = all(t in entities_by_type for t in event.get("required_entities", []))
            if has_required:
                event_obj = {
                    "type": event["type"],
                    "text": s["text"],
                    "trigger": next(w for w in event["trigger_words"] if w in sentence_text),
                    "entities": [e for e in s["entities"]
                                 if e["label"] in event.get("required_entities", []) + event.get("optional_entities", [])]
                }
                detected_events.append(event_obj)

    return detected_events

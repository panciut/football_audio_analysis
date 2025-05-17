# src/pipeline/event_zero_shot.py

from transformers import pipeline

# Load once globally (can be moved to config if needed)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CANDIDATE_LABELS = [
    "Goal", "Foul", "Red Card", "Yellow Card", "Substitution", "Offside", "None"
]

def detect_events_zero_shot(sentences, labels=CANDIDATE_LABELS, threshold=0.7):
    """
    Classifies each sentence into an event type using zero-shot classification.

    Args:
        sentences (list): list of dicts with 'text' and optionally 'start'.
        labels (list): candidate event types.
        threshold (float): minimum confidence to accept prediction.

    Returns:
        List of detected events with metadata.
    """
    results = []

    for s in sentences:
        text = s.get("text", "").strip()
        if not text:
            continue

        out = zero_shot_classifier(text, candidate_labels=labels, multi_label=False)
        pred, score = out["labels"][0], out["scores"][0]

        if pred != "None" and score >= threshold:
            results.append({
                "type": pred,
                "text": text,
                "confidence": round(score, 3),
                "start": s.get("start", None),
                "entities": s.get("entities", []),
                "source": "zero-shot"
            })

    return results

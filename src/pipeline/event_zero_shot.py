# src/pipeline/event_zero_shot.py

from transformers import pipeline
import torch

# Select device (MPS on Mac, fallback to CPU)
device = 0 if torch.backends.mps.is_available() else -1

zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

CANDIDATE_LABELS = [
    "Goal", "Disallowed Goal", "Foul", "Red Card", "Yellow Card",
    "Substitution", "Offside", "Penalty", "Injury", "Corner Kick",
    "Match Start", "Halftime", "Second Half Start", "Match End",
     "None"
]


def detect_events_zero_shot(sentences, labels=CANDIDATE_LABELS, threshold=0.7, window_size=15.0):
    """
    Groups sentence segments into ~20s blocks, then applies zero-shot classification.
    Detects events like goals, fouls, substitutions, and period transitions.

    Args:
        sentences (list): list of dicts with 'text' and 'start' timestamp.
        labels (list): candidate event types.
        threshold (float): minimum confidence to accept prediction.
        window_size (float): seconds per context window.

    Returns:
        List of detected event dicts.
    """
    # Filter only segments with valid start times
    segments = sorted(
        [s for s in sentences if s.get("start") is not None],
        key=lambda x: x["start"]
    )

    results = []
    i = 0

    while i < len(segments):
        window_text = []
        window_start = segments[i]["start"]
        window_end = window_start + window_size

        j = i
        while j < len(segments) and segments[j]["start"] <= window_end:
            window_text.append(segments[j]["text"])
            j += 1

        text_block = " ".join(window_text).strip()
        if not text_block:
            i = j
            continue

        output = zero_shot_classifier(text_block, candidate_labels=labels, multi_label=False)
        pred, score = output["labels"][0], output["scores"][0]

        if pred != "None" and score >= threshold:
            results.append({
                "type": pred,
                "text": text_block,
                "confidence": round(score, 3),
                "start": round(window_start, 2),
                "end": round(segments[j - 1]["start"], 2) if j > i else round(window_start + window_size, 2),
                "source": "zero-shot"
            })

        i = j

    return results

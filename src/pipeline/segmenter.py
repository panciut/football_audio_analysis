# src/pipeline/segmenter.py

import spacy

# Load the spaCy English model once at import time
nlp = spacy.load("en_core_web_sm")

def segment_sentences(text: str) -> list[dict]:
    """
    Splits the transcript text into sentence-level segments using spaCy.

    Returns:
        List of dicts with sentence span and text.
    """
    doc = nlp(text)
    return [
        {"start_char": sent.start_char, "end_char": sent.end_char, "text": sent.text.strip()}
        for sent in doc.sents
    ]

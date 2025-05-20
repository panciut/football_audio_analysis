# src/pipeline/segmenter.py

import spacy
nlp = spacy.load("en_core_web_sm")


def segment_sentences(text: str, segments: list = None) -> list[dict]:
    """
    Splits transcript text into sentences, optionally aligns each sentence
    with its source Whisper segment to inherit start timestamps.
    """
    doc = nlp(text)
    split_sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_start = None

        # Try to inherit timing from nearest Whisper segment
        if segments:
            for seg in segments:
                if sent_text in seg["text"]:
                    sent_start = seg["start"]
                    break

        split_sentences.append({
            "text": sent_text,
            "start": sent_start
        })

    return split_sentences

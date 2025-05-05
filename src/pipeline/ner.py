# src/pipeline/ner.py

import spacy

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(sentences):
    """
    Runs NER on each sentence and appends entity information.

    Args:
        sentences (list of dict): Sentences with 'text' field.

    Returns:
        list of dict: Each sentence enriched with 'entities' list.
    """
    for s in sentences:
        doc = nlp(s["text"])
        s["entities"] = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "DATE", "TIME", "GPE", "EVENT"}
        ]
    return sentences

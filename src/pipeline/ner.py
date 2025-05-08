# src/pipeline/ner.py

import spacy
import json
from rapidfuzz import fuzz, process
from src.config import GAZETTEER_PATH

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Load structured gazetteer
with open(GAZETTEER_PATH, "r", encoding="utf-8") as f:
    GAZETTEER = json.load(f)["entities"]

# Build alias lookup
ALIAS_LOOKUP = {}
for ent in GAZETTEER:
    all_names = [ent["name"]] + ent.get("aliases", [])
    for alias in all_names:
        ALIAS_LOOKUP[alias.lower()] = {
            "canonical": ent["name"],
            "type": ent["type"]
        }

ALL_ALIASES = list(ALIAS_LOOKUP.keys())


def exact_alias_match(text: str):
    """
    Returns all exact alias matches in the sentence.
    """
    matches = []
    lowered = text.lower()
    for alias, entity in ALIAS_LOOKUP.items():
        if alias in lowered:
            matches.append({
                "text": entity["canonical"],
                "label": entity["type"],
                "source": "gazetteer",
                "canonical": entity["canonical"]
            })
    return matches


def fuzzy_match_tokens(text: str, threshold=87):
    """
    Searches token spans (2–3 grams) for fuzzy alias matches.
    Only matches non-stopword, non-short phrases.
    """
    matches = []
    tokens = text.split()

    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):  # n-grams up to 3
            phrase = " ".join(tokens[i:j]).strip()
            if len(phrase) < 4:
                continue  # Skip short/inconclusive matches

            match, score, _ = process.extractOne(phrase.lower(), ALL_ALIASES, scorer=fuzz.partial_ratio)
            if score >= threshold:
                entity_info = ALIAS_LOOKUP[match]
                matches.append({
                    "text": entity_info["canonical"],
                    "label": entity_info["type"],
                    "source": "fuzzy",
                    "canonical": entity_info["canonical"],
                    "matched_span": phrase
                })
    return matches


def extract_named_entities(sentences):
    for s in sentences:
        doc = nlp(s["text"])

        # First: run gazetteer and fuzzy matches
        gazetteer_entities = exact_alias_match(s["text"])
        fuzzy_entities = fuzzy_match_tokens(s["text"])

        # Keep a lowercase set of already found entity texts
        domain_texts = {e["text"].lower() for e in gazetteer_entities + fuzzy_entities}

        # spaCy entities, override PERSON → PLAYER
        spacy_entities = []
        for ent in doc.ents:
            if ent.label_ in { "ORG", "DATE", "TIME", "GPE", "EVENT"}:
                label = "PLAYER" if ent.label_ == "PERSON" else ent.label_
                if ent.text.lower() not in domain_texts:
                    spacy_entities.append({
                        "text": ent.text,
                        "label": label,
                        "source": "spacy"
                    })

        # Merge all sources
        combined = spacy_entities + gazetteer_entities + fuzzy_entities

        # Deduplicate final list
        seen = set()
        deduped = []
        for e in combined:
            key = (e["text"].lower(), e["label"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        s["entities"] = deduped

    return sentences

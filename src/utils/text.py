# src/utils/text.py

def safe_replace(text, span, replacement):
    import re
    pattern = r'\b' + re.escape(span) + r'\b'
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def apply_canonical_substitutions(sentences, segments=None):
    for s in sentences:
        text = s["text"]
        replaced_spans = set()
        for ent in s.get("entities", []):
            if ent.get("source") in {"gazetteer", "fuzzy"} and ent.get("canonical"):
                span = ent.get("matched_span", ent["text"])
                canonical = ent["canonical"]
                if span.lower() != canonical.lower() and canonical.lower() not in replaced_spans:
                    new_text = safe_replace(text, span, canonical)
                    if new_text != text:
                        text = new_text
                        replaced_spans.add(canonical.lower())
        s["text"] = text

    if segments is not None:
        for seg in segments:
            seg_text = seg["text"]
            for s in sentences:
                for ent in s.get("entities", []):
                    if ent.get("source") in {"gazetteer", "fuzzy"} and ent.get("canonical"):
                        span = ent.get("matched_span", ent["text"])
                        canonical = ent["canonical"]
                        if span.lower() != canonical.lower() and canonical.lower() not in seg_text.lower():
                            new_seg_text = safe_replace(seg_text, span, canonical)
                            if new_seg_text != seg_text:
                                seg_text = new_seg_text
            seg["text"] = seg_text
    return sentences, segments

def get_entities_by_type(sentences, label):
    """
    Returns a flat list of (entity, sentence_text) for entities of a given type.
    """
    return [
        (e["text"], s["text"])
        for s in sentences
        for e in s.get("entities", [])
        if e["label"] == label
    ]

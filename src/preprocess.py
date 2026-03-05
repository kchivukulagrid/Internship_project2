"""Preprocessing helpers for converting CoNLL tags into JSON-entity prompts."""

import json
import re


# ---------------------------------------------------------------------
# Synonym table used by optional train augmentation.
# ---------------------------------------------------------------------
SYNONYM_MAP = {
    "PER": ["Alex Morgan", "Jordan Lee", "Taylor Kim", "Chris Patel"],
    "ORG": ["Acme Corp", "Globex", "Nova Labs", "Orion Group"],
    "LOC": ["New York", "San Francisco", "Berlin", "Tokyo"],
    "MISC": ["World Cup", "Nobel Prize", "iPhone", "Python"],
}


def _append_entity_if_active(entities: list[dict], current_tokens: list[str], current_label: str | None) -> None:
    """Append active BIO span if present."""
    if current_tokens:
        entities.append({"text": " ".join(current_tokens), "label": current_label})


def extract_entities(tokens: list[str], tags: list[int], label_names: list[str]) -> list[dict]:
    """Convert BIO tags into structured entity spans."""
    entities = []
    current_tokens = []
    current_label = None

    for token, tag_id in zip(tokens, tags):
        label = label_names[tag_id]

        if label.startswith("B-"):
            _append_entity_if_active(entities, current_tokens, current_label)
            current_tokens = [token]
            current_label = label[2:]
        elif label.startswith("I-") and current_label:
            current_tokens.append(token)
        else:
            _append_entity_if_active(entities, current_tokens, current_label)
            current_tokens = []
            current_label = None

    _append_entity_if_active(entities, current_tokens, current_label)

    return entities


def build_prompt(text: str, prompt_style: str = "with_defs") -> str:
    """Build instruction-style prompt with selectable definition block."""
    if prompt_style == "no_defs":
        return f"""Extract named entities as JSON.

Text:
{text}
"""

    return f"""Extract named entities as JSON.

Definitions:
PER = person
ORG = organization
LOC = location
MISC = miscellaneous

Text:
{text}
"""


def _pick_synonym(label: str, source_text: str) -> str:
    """Deterministically pick a synonym replacement for a label."""
    options = SYNONYM_MAP.get(label, [])
    if not options:
        return source_text
    idx = abs(hash(source_text)) % len(options)
    return options[idx]


def augment_text_and_entities(text: str, entities: list[dict]) -> tuple[str, list[dict]]:
    """Create a synonym-augmented variant by replacing entity mentions."""
    updated_entities = []
    updated_text = text

    for entity in entities:
        ent_text = entity["text"]
        ent_label = entity["label"]
        replacement = _pick_synonym(ent_label, ent_text)
        updated_entities.append({"text": replacement, "label": ent_label})
        pattern = re.escape(ent_text)
        updated_text = re.sub(pattern, replacement, updated_text, count=1)

    return updated_text, updated_entities


def convert_example(example: dict, label_names: list[str], prompt_style: str = "with_defs") -> dict:
    """Convert one CoNLL example to prompt/output training format."""
    tokens = example["tokens"]
    tags = example["ner_tags"]

    text = " ".join(tokens)
    entities = extract_entities(tokens, tags, label_names)

    prompt = build_prompt(text, prompt_style=prompt_style)
    output = json.dumps({"entities": entities})

    return {"prompt": prompt, "output": output}

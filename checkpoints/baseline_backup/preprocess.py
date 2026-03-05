import json


def extract_entities(tokens, tags, label_names):
    """
    Convert BIO tags into structured entity spans.
    """
    entities = []
    current_tokens = []
    current_label = None

    for token, tag_id in zip(tokens, tags):
        label = label_names[tag_id]

        if label.startswith("B-"):
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label
                })
            current_tokens = [token]
            current_label = label[2:]

        elif label.startswith("I-") and current_label:
            current_tokens.append(token)

        else:
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label
                })
            current_tokens = []
            current_label = None

    if current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "label": current_label
        })

    return entities


def build_prompt(text):
    """
    Build instruction-style prompt.
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


def convert_example(example, label_names):
    """
    Convert one CoNLL example to training format.
    """
    tokens = example["tokens"]
    tags = example["ner_tags"]

    text = " ".join(tokens)
    entities = extract_entities(tokens, tags, label_names)

    prompt = build_prompt(text)
    output = json.dumps({"entities": entities})

    return {
        "prompt": prompt,
        "output": output
    }

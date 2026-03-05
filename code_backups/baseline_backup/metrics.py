import json
import re

VALID_LABELS = {"PER", "ORG", "LOC", "MISC"}


def _normalize_entities(entities):
    normalized = []
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        text = entity.get("text")
        label = entity.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            continue
        text = text.strip()
        label = label.strip().upper()
        if not text or label not in VALID_LABELS:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"text": text, "label": label})
    return normalized


def _close_unbalanced_json(candidate):
    opens = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            opens.append(ch)
        elif ch in "}]":
            if not opens:
                continue
            top = opens[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                opens.pop()
    if in_string:
        candidate += '"'
    while opens:
        top = opens.pop()
        candidate += "}" if top == "{" else "]"
    return candidate


def _find_balanced_json(text):
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts:
        stack = []
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        return text[start : idx + 1]
                else:
                    break
    return None


def _json_to_schema(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("entities"), list):
            return {"entities": _normalize_entities(obj["entities"])}
        if isinstance(obj.get("entity"), dict):
            ent = obj["entity"]
            candidate = {"text": ent.get("label") or ent.get("value"), "label": ent.get("type")}
            return {"entities": _normalize_entities([candidate])}
    if isinstance(obj, list):
        return {"entities": _normalize_entities(obj)}
    return None


def _regex_recover_entities(text):
    entities = []
    p1 = re.finditer(
        r'"text"\s*:\s*"(?P<text>[^"]+)"\s*,\s*"label"\s*:\s*"(?P<label>PER|ORG|LOC|MISC)"',
        text,
    )
    p2 = re.finditer(
        r'"label"\s*:\s*"(?P<label>PER|ORG|LOC|MISC)"\s*,\s*"text"\s*:\s*"(?P<text>[^"]+)"',
        text,
    )
    p3 = re.finditer(
        r'"type"\s*:\s*"(?P<label>PER|ORG|LOC|MISC)"\s*,\s*"(?:label|value|text)"\s*:\s*"(?P<text>[^"]+)"',
        text,
    )
    p4 = re.finditer(
        r'"(?:label|value|text)"\s*:\s*"(?P<text>[^"]+)"\s*,\s*"type"\s*:\s*"(?P<label>PER|ORG|LOC|MISC)"',
        text,
    )
    for match in [*p1, *p2, *p3, *p4]:
        entities.append({"text": match.group("text"), "label": match.group("label")})
    entities = _normalize_entities(entities)
    if entities:
        return {"entities": entities}
    return None


def extract_json(text):
    """
    Extract and normalize model output to {"entities":[{"text","label"}]}.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    candidates = [text]
    balanced = _find_balanced_json(text)
    if balanced and balanced != text:
        candidates.append(balanced)
    candidates += [_close_unbalanced_json(c) for c in list(candidates)]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            normalized = _json_to_schema(parsed)
            if normalized is not None:
                return normalized
        except json.JSONDecodeError:
            continue

    return _regex_recover_entities(text)


def to_set(entities):
    return set(
        (e["text"], e["label"])
        for e in entities
        if isinstance(e, dict) and "text" in e and "label" in e
    )


def compute_metrics(file_path):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    valid_json = 0
    repaired_json = 0
    total = 0

    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)

            gt = json.loads(item["ground_truth"])
            pred_json = extract_json(item["prediction"])

            if pred_json is not None:
                valid_json += 1
                original_valid = False
                try:
                    original = json.loads(item["prediction"])
                    original_valid = _json_to_schema(original) is not None
                except json.JSONDecodeError:
                    original_valid = False
                if not original_valid:
                    repaired_json += 1
                pred_entities = pred_json.get("entities", [])
            else:
                pred_entities = []

            gt_entities = gt.get("entities", [])

            gt_set = to_set(gt_entities)
            pred_set = to_set(pred_entities)

            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    validity = valid_json / total if total > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "validity": validity,
        "total_examples": total,
        "valid_json_count": valid_json,
        "repaired_json_count": repaired_json,
    }

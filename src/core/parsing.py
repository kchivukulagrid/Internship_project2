"""Parsing and normalization for Task 1 schema."""

from __future__ import annotations

import json
import re
from typing import Any

from src.core.schema import VALID_TYPES

# ---------------------------------------------------------------------
# JSON Repair Helpers
# ---------------------------------------------------------------------


def _close_unbalanced_json(candidate: str) -> str:
    # Repair common truncation by balancing braces/brackets.
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


def _find_balanced_json(text: str) -> str | None:
    # Find the first balanced JSON object/array within a string.
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


# ---------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------
def _normalize_entity(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    # Allow legacy keys for smoother migration.
    ent_type = raw.get("type") or raw.get("label")
    value = raw.get("value") or raw.get("text")
    start = raw.get("start")
    end = raw.get("end")

    if not isinstance(ent_type, str) or ent_type.strip().upper() not in VALID_TYPES:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        return None

    return {
        "type": ent_type.strip().upper(),
        "value": value.strip(),
        "start": start,
        "end": end,
    }


def _normalize_payload(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    entities = obj.get("entities")
    confidence = obj.get("confidence")
    if not isinstance(entities, list):
        return None
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        return None
    # De-duplicate identical spans.
    normalized = []
    seen = set()
    for ent in entities:
        norm = _normalize_entity(ent)
        if norm is None:
            return None
        key = (norm["type"], norm["value"], norm["start"], norm["end"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(norm)
    return {"entities": normalized, "confidence": float(confidence)}


# ---------------------------------------------------------------------
# Public Parsing API
# ---------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any] | None:
    """Extract and normalize model output to Task 1 schema."""
    if not isinstance(text, str):
        return None

    text = text.strip()
    # Strip common fenced code block wrappers.
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
            normalized = _normalize_payload(parsed)
            if normalized is not None:
                return normalized
        except json.JSONDecodeError:
            continue

    return None

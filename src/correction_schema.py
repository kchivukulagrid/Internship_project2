"""Schema helpers for correction payloads."""

from __future__ import annotations

from typing import Any

VALID_LABELS = {"PER", "ORG", "LOC", "MISC"}


def normalize_entities(value: Any) -> list[dict[str, str]]:
    """Normalize and validate entity list into canonical shape."""
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
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


def normalize_payload(payload: Any) -> dict[str, list[dict[str, str]]]:
    """Normalize any payload to {'entities': [...]} shape."""
    if isinstance(payload, dict) and "entities" in payload:
        return {"entities": normalize_entities(payload.get("entities"))}
    return {"entities": []}


def is_valid_payload(payload: Any) -> bool:
    """Return True only when payload is canonical and valid."""
    if not isinstance(payload, dict):
        return False
    entities = payload.get("entities")
    return normalize_payload(payload)["entities"] == normalize_entities(entities)

"""Dataset utilities for Task 1 schema with character offsets."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------
# Text + Offset Helpers
# ---------------------------------------------------------------------


def build_text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens with spaces and compute character offsets for each token."""
    offsets: list[tuple[int, int]] = []
    pieces: list[str] = []
    cursor = 0
    for idx, token in enumerate(tokens):
        # Ensure offsets line up with space-joined text.
        if idx > 0:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        end = cursor
        offsets.append((start, end))
    return "".join(pieces), offsets


# ---------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------
def extract_entities_with_offsets(
    tokens: list[str],
    tags: list[int],
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Convert BIO tags into entities with char offsets."""
    text, offsets = build_text_and_offsets(tokens)
    entities: list[dict[str, Any]] = []
    current_label: str | None = None
    start_idx: int | None = None

    def flush(end_idx: int | None) -> None:
        nonlocal current_label, start_idx
        # Emit a span when we close a BIO run.
        if current_label is None or start_idx is None or end_idx is None:
            current_label = None
            start_idx = None
            return
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]
        value = text[start_char:end_char]
        entities.append(
            {
                "type": current_label,
                "value": value,
                "start": start_char,
                "end": end_char,
            }
        )
        current_label = None
        start_idx = None

    for idx, tag_id in enumerate(tags):
        label = label_names[tag_id]
        # Start a new entity.
        if label.startswith("B-"):
            flush(idx - 1 if start_idx is not None else None)
            current_label = label[2:]
            start_idx = idx
        # Continue the active entity span.
        elif label.startswith("I-") and current_label:
            continue
        else:
            # Outside of any entity span.
            flush(idx - 1 if start_idx is not None else None)

    flush(len(tags) - 1 if start_idx is not None else None)
    return entities


# ---------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------
def build_output(entities: list[dict[str, Any]], confidence: float = 1.0) -> dict[str, Any]:
    """Build output payload with the canonical schema."""
    return {"entities": entities, "confidence": confidence}

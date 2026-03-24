"""Schema definitions for Task 1 constrained JSON output."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------
# Schema Constants
# ---------------------------------------------------------------------

VALID_TYPES = {"PER", "ORG", "LOC", "MISC"}


# ---------------------------------------------------------------------
# Public Schema API
# ---------------------------------------------------------------------
def ner_schema() -> dict[str, Any]:
    """Return JSON schema for constrained decoding."""
    # Keep schema minimal and strict so constrained decoders can enforce it.
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "value": {"type": "string"},
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 0},
                    },
                    "required": ["type", "value", "start", "end"],
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["entities", "confidence"],
    }


# ---------------------------------------------------------------------
# Convenience Helpers
# ---------------------------------------------------------------------
def empty_output() -> dict[str, Any]:
    """Return an empty, valid output shape."""
    # Confidence defaults to 0.0 to avoid implying correctness.
    return {"entities": [], "confidence": 0.0}


def is_valid_type(value: Any) -> bool:
    """Return True if value is a supported entity type."""
    return isinstance(value, str) and value.strip().upper() in VALID_TYPES

"""Prompt templates for Task 1 schema."""

from __future__ import annotations

# ---------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------


def build_prompt(text: str, prompt_style: str = "with_defs") -> str:
    """Build instruction prompt that requests strict JSON schema output."""
    # Keep schema inline to encourage strict formatting from the model.
    schema_hint = (
        'Return ONLY valid JSON with schema: {"entities":[{"type":"PER|ORG|LOC|MISC",'
        '"value":"...", "start":0, "end":0}], "confidence":0.0}\n'
        "Use character offsets into the provided Text. "
        "The end offset is exclusive.\n"
        "Output:\n"
    )

    if prompt_style == "no_defs":
        return f"""Extract named entities as JSON.

Text:
{text}
{schema_hint}"""

    return f"""Extract named entities as JSON.

Definitions:
PER = person
ORG = organization
LOC = location
MISC = miscellaneous

Text:
{text}
{schema_hint}"""

from __future__ import annotations

from typing import Any

from src.core.schema import ner_schema

import json as _json


def generate_constrained_json(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
) -> str:
    """Generate JSON using outlines if available, otherwise raise."""
    try:
        from outlines import generate, models
    except Exception:
        # Newer Outlines uses `generator` instead of `generate`.
        try:
            from outlines import generator as generate  # type: ignore
            from outlines import models
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Outlines is required for constrained decoding. "
                "Install it with: pip install outlines"
            ) from exc

    # Build Outlines model wrapper for transformers backend.
    if hasattr(models, "Transformers"):
        outlines_model = models.Transformers(model, tokenizer=tokenizer)
    elif hasattr(models, "transformers") and hasattr(models.transformers, "Transformers"):
        outlines_model = models.transformers.Transformers(model, tokenizer=tokenizer)
    else:
        raise RuntimeError("Unsupported Outlines version: cannot build transformers model.")

    # Use json_schema logits processor + HF generation for maximum compatibility.
    try:
        from outlines.generator import get_json_schema_logits_processor
    except Exception as exc:
        raise RuntimeError("Unsupported Outlines version: JSON schema processor not found.") from exc

    schema_json = _json.dumps(ner_schema())
    try:
        processor = get_json_schema_logits_processor(
            backend_name="transformers",
            model=outlines_model,
            json_schema=schema_json,
        )
    except ValueError:
        # Fallback to the default backend exposed by this Outlines version.
        processor = get_json_schema_logits_processor(
            backend_name=None,
            model=outlines_model,
            json_schema=schema_json,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.05,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=[processor],
    )
    gen = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def generate_unconstrained(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate text using standard Hugging Face decoding."""
    # Deterministic decoding when temperature is 0.
    do_sample = temperature > 0.0
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=1.05,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature if do_sample else 0.0,
    )
    gen = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)

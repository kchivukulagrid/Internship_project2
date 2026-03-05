"""Model loading helpers for LoRA fine-tuning and inference."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


# ---------------------------
# Device Selection
# ---------------------------
def get_device() -> str:
    """Return the preferred available accelerator device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------
# Model + LoRA Loader
# ---------------------------
def load_model(model_name: str):
    """Load base model and apply project LoRA configuration."""
    device = get_device()
    print(f"\nUsing device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.to(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, device

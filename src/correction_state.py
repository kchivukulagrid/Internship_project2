"""State container used by the correction interface."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CorrectionState:
    model_name: str
    adapter_path: str
    prediction_export: str
    correction_export: str
    active_learning_export: str
    processed_count: int = 0
    saved_count: int = 0
    last_message: str = "Ready"
    extra: dict = field(default_factory=dict)

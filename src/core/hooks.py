"""Hooks for logit-lens and activation capture."""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------
# Forward Hook Utilities
# ---------------------------------------------------------------------


def register_forward_hooks(
    modules: list[tuple[str, Any]],
    hook_fn: Callable[[str, Any, Any], None],
) -> list[Any]:
    """Register hooks on modules and return handles."""
    handles = []
    for name, module in modules:
        def _hook(mod, inp, out, layer_name=name):
            hook_fn(layer_name, inp, out)
        handles.append(module.register_forward_hook(_hook))
    return handles


def remove_hooks(handles: list[Any]) -> None:
    """Remove all registered hooks."""
    for handle in handles:
        handle.remove()

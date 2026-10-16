"""Small growth invariants of GaME's density control, free of renderer imports (checks run without CUDA)."""
from __future__ import annotations


def clear_densification_rows(model, rows):
    """Discard a protected row's previous density statistics, including restored ones."""
    for name in ("xyz_gradient_accum", "denom", "max_radii2D"):
        values = getattr(model, name)
        if not len(values):
            continue
        if len(values) != len(rows):
            raise ValueError(f"{name}: {len(values)} rows, mask has {len(rows)}")
        values[rows] = 0


def restored_session_settings(state, default_support_tol):
    """Recover saved context, falling back only to recorded session boundaries.

    Older checkpoints record ``session_starts`` but omit the active boundary. An
    empty list provides no boundary; callers must establish one before refinement.
    The configured sensor tolerance is the legacy fallback for the unsaved band.
    """
    if "session_start" in state:
        start = state["session_start"]
    else:
        starts = state.get("session_starts", [])
        start = starts[-1] if starts else None
    return (None if start is None else int(start),
            float(state.get("support_tol", default_support_tol)))

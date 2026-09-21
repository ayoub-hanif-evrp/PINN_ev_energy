"""Honest failure for later paper phases that are not implemented yet."""

from __future__ import annotations


class PhaseNotImplementedError(NotImplementedError):
    """Raised by Phase 5–13 stubs so `make paper` cannot emit fake metrics."""


def not_implemented(phase: int, name: str) -> None:
    raise PhaseNotImplementedError(
        f"Phase {phase} ({name}) is not implemented yet. "
        "Run `python scripts/audit_data.py` first. "
        "This stub exists so the paper pipeline cannot fabricate results."
    )

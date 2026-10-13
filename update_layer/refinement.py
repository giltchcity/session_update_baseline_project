"""Completion accounting for optional end-of-run refinement; no backend dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class RefinementProgress:
    """Count successful optimizer steps, separately from the requested iteration budget."""

    requested_iterations: int
    completed_iterations: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if (type(self.requested_iterations) is not int or type(self.completed_iterations) is not int
                or not 0 <= self.completed_iterations <= self.requested_iterations):
            raise ValueError("refinement iterations must be integers with 0 <= completed <= requested")
        if self.stop_reason is not None and (not isinstance(self.stop_reason, str) or not self.stop_reason):
            raise ValueError("a refinement stop reason must be a nonempty string")

    def advance(self) -> None:
        """Call only after an optimizer step has completed successfully."""
        if self.stop_reason is not None or self.completed_iterations >= self.requested_iterations:
            raise ValueError("cannot advance a stopped or completed refinement")
        self.completed_iterations += 1

    def stop(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("a refinement stop reason must be a nonempty string")
        self.stop_reason = reason

    def as_dict(self) -> dict:
        complete = self.completed_iterations == self.requested_iterations and self.stop_reason is None
        status = ("disabled" if self.requested_iterations == 0 else "ok") if complete else "partial"
        reason = self.stop_reason
        if not complete and reason is None:
            reason = "iteration_budget_not_completed"
        return {"status": status, "requested_iterations": self.requested_iterations,
                "completed_iterations": self.completed_iterations, "stop_reason": reason}


def normalize_refinement_report(report: Mapping | None) -> dict:
    """Keep legacy backends' None return; verify iteration counts in reporting backends."""
    if report is None:
        return {"status": "ok"}
    if not isinstance(report, Mapping):
        raise TypeError("finish_session must return a refinement report or None")
    result = dict(report)
    progress = RefinementProgress(result["requested_iterations"], result["completed_iterations"],
                                  result.get("stop_reason"))
    verified = progress.as_dict()
    # A backend reporting a failure must never be upgraded to success by its iteration count.
    if result.get("status") in ("partial", "failed"):
        verified["status"] = result["status"]
        verified["stop_reason"] = verified["stop_reason"] or "backend_reported_incomplete"
    result.update(verified)
    return result


def completion_exit_code(report: Mapping) -> int:
    """Use after writing partial artifacts and their report, so a stopped run is recoverable."""
    return 0 if report.get("status") in ("ok", "disabled") else 3

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StabilizerConfig:
    min_confidence: float = 0.6
    k_consecutive: int = 3
    lockout_seconds: float = 1.0
    unknown_label: str = "UNKNOWN"


class StateStabilizer:
    """Simple hysteresis-based state stabilizer for streaming predictions."""

    def __init__(self, *, config: StabilizerConfig):
        if config.k_consecutive <= 0:
            raise ValueError("k_consecutive must be > 0")
        if config.min_confidence < 0.0 or config.min_confidence > 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        self.config = config

        self.current_state: str = config.unknown_label
        self._candidate_state: str | None = None
        self._candidate_count: int = 0
        self._last_switch_time: float | None = None

    def update(self, predicted_state: str, confidence: float, *, t: float | None) -> str:
        # Low-confidence predictions become UNKNOWN.
        state = predicted_state
        if confidence < self.config.min_confidence:
            state = self.config.unknown_label

        # If unknown: hold the last stable state (do not switch).
        if state == self.config.unknown_label:
            self._candidate_state = None
            self._candidate_count = 0
            return self.current_state

        # No change.
        if state == self.current_state:
            self._candidate_state = None
            self._candidate_count = 0
            return self.current_state

        # Lockout period after a switch.
        if t is not None and self._last_switch_time is not None:
            if (t - self._last_switch_time) < self.config.lockout_seconds:
                return self.current_state

        # Candidate accumulation.
        if self._candidate_state == state:
            self._candidate_count += 1
        else:
            self._candidate_state = state
            self._candidate_count = 1

        if self._candidate_count >= self.config.k_consecutive:
            self.current_state = state
            self._last_switch_time = t
            self._candidate_state = None
            self._candidate_count = 0

        return self.current_state


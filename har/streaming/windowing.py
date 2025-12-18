from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from har.streaming.protocol import ImuSample


@dataclass(frozen=True)
class ImuWindow:
    x: np.ndarray  # (window_size, 6) float32
    t_start: float | None
    t_end: float | None
    labels: list[str] | None


class SlidingImuWindow:
    def __init__(self, *, window_size: int, hop_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        if hop_size > window_size:
            raise ValueError("hop_size must be <= window_size")

        self.window_size = int(window_size)
        self.hop_size = int(hop_size)

        self._vectors: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._times: deque[float | None] = deque(maxlen=self.window_size)
        self._labels: deque[str | None] = deque(maxlen=self.window_size)

        self._total_samples = 0
        self._last_emit_total = 0

    def append(self, sample: ImuSample) -> list[ImuWindow]:
        vec = np.array([sample.ax, sample.ay, sample.az, sample.gx, sample.gy, sample.gz], dtype=np.float32)
        self._vectors.append(vec)
        self._times.append(sample.t)
        self._labels.append(sample.label)
        self._total_samples += 1

        windows: list[ImuWindow] = []
        # Emit at hop intervals once we have enough history for a full window.
        if len(self._vectors) < self.window_size:
            return windows
        if (self._total_samples - self._last_emit_total) < self.hop_size:
            return windows

        x = np.stack(list(self._vectors), axis=0)
        t_start = self._times[0]
        t_end = self._times[-1]
        labels = [l for l in self._labels if l is not None] or None
        windows.append(ImuWindow(x=x, t_start=t_start, t_end=t_end, labels=labels))
        self._last_emit_total = self._total_samples
        return windows


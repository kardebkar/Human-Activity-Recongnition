from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticMotionDataset:
    x: np.ndarray
    y: np.ndarray
    class_names: tuple[str, ...]


def create_synthetic_motion_dataset(
    *,
    n_samples: int = 2000,
    seq_len: int = 128,
    seed: int = 42,
) -> SyntheticMotionDataset:
    """Create a small synthetic motion dataset with simple, realistic-ish patterns.

    This is a local-only fallback/demo dataset inspired by the notebook:
    acceleration + gyroscope signals -> summary statistics features.
    """

    rng = np.random.default_rng(seed)

    class_names = ("WALKING", "RUNNING", "SITTING", "JUMPING", "STANDING", "CYCLING")

    x_rows: list[list[float]] = []
    y_rows: list[str] = []

    samples_per_class = max(1, n_samples // len(class_names))

    for activity in class_names:
        for _ in range(samples_per_class):
            t = np.linspace(0, 5, seq_len)

            if activity == "WALKING":
                accel_x = np.sin(2 * np.pi * 2 * t) + 0.2 * rng.standard_normal(seq_len)
                accel_y = 0.5 * np.sin(2 * np.pi * 2 * t + np.pi / 4) + 0.1 * rng.standard_normal(seq_len)
                accel_z = 9.8 + 0.3 * np.sin(2 * np.pi * 4 * t) + 0.1 * rng.standard_normal(seq_len)
                gyro_x = 0.5 * np.cos(2 * np.pi * 2 * t) + 0.05 * rng.standard_normal(seq_len)

            elif activity == "RUNNING":
                accel_x = 2 * np.sin(2 * np.pi * 4 * t) + 0.3 * rng.standard_normal(seq_len)
                accel_y = np.sin(2 * np.pi * 4 * t + np.pi / 3) + 0.2 * rng.standard_normal(seq_len)
                accel_z = 9.8 + np.abs(np.sin(2 * np.pi * 8 * t)) + 0.2 * rng.standard_normal(seq_len)
                gyro_x = np.cos(2 * np.pi * 4 * t) + 0.1 * rng.standard_normal(seq_len)

            elif activity == "SITTING":
                accel_x = 0.05 * rng.standard_normal(seq_len)
                accel_y = 0.05 * rng.standard_normal(seq_len)
                accel_z = 9.8 + 0.02 * rng.standard_normal(seq_len)
                gyro_x = 0.01 * rng.standard_normal(seq_len)

            elif activity == "JUMPING":
                jumps = np.zeros(seq_len)
                jump_times = (20, 50, 80, 110)
                for jt in jump_times:
                    if 5 <= jt < seq_len - 5:
                        window = np.arange(10) - 5
                        jumps[jt - 5 : jt + 5] = 3 * np.exp(-(window**2) / 5)
                accel_x = jumps + 0.2 * rng.standard_normal(seq_len)
                accel_y = 0.5 * jumps + 0.1 * rng.standard_normal(seq_len)
                accel_z = 9.8 + 2 * np.abs(jumps) + 0.2 * rng.standard_normal(seq_len)
                gyro_x = np.gradient(jumps) + 0.05 * rng.standard_normal(seq_len)

            elif activity == "STANDING":
                accel_x = 0.02 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(seq_len)
                accel_y = 0.02 * np.sin(2 * np.pi * 0.2 * t + np.pi / 2) + 0.02 * rng.standard_normal(
                    seq_len
                )
                accel_z = 9.8 + 0.01 * rng.standard_normal(seq_len)
                gyro_x = 0.01 * rng.standard_normal(seq_len)

            else:  # CYCLING
                accel_x = np.sin(2 * np.pi * 1.5 * t) + 0.15 * rng.standard_normal(seq_len)
                accel_y = np.cos(2 * np.pi * 1.5 * t) + 0.15 * rng.standard_normal(seq_len)
                accel_z = 9.8 + 0.2 * np.sin(2 * np.pi * 3 * t) + 0.1 * rng.standard_normal(seq_len)
                gyro_x = 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.05 * rng.standard_normal(seq_len)

            gyro_y = 0.3 * gyro_x + 0.03 * rng.standard_normal(seq_len)
            gyro_z = 0.1 * gyro_x + 0.03 * rng.standard_normal(seq_len)

            features = np.column_stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])

            # Summary statistics per axis.
            row: list[float] = []
            for col in range(features.shape[1]):
                series = features[:, col]
                row.extend(
                    [
                        float(series.mean()),
                        float(series.std()),
                        float(series.max()),
                        float(series.min()),
                        float(np.percentile(series, 25)),
                        float(np.percentile(series, 75)),
                    ]
                )

            x_rows.append(row)
            y_rows.append(activity)

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    return SyntheticMotionDataset(x=x, y=y, class_names=class_names)


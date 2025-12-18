from __future__ import annotations

import numpy as np

_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
_STATS = ("mean", "std", "min", "max", "energy")


def feature_names() -> list[str]:
    names: list[str] = []
    for ch in _CHANNELS:
        for s in _STATS:
            names.append(f"{ch}_{s}")
    for mag in ("acc_mag", "gyro_mag"):
        for s in _STATS:
            names.append(f"{mag}_{s}")
    # Correlations
    names += [
        "acc_corr_xy",
        "acc_corr_xz",
        "acc_corr_yz",
        "gyro_corr_xy",
        "gyro_corr_xz",
        "gyro_corr_yz",
        "accgyro_corr_mag",
    ]
    return names


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(c) or np.isinf(c):
        return 0.0
    return c


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract tabular features from a (T, 6) IMU window: [ax, ay, az, gx, gy, gz]."""

    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"Expected window shape (T, 6), got {window.shape}")

    feats: list[float] = []
    # Per-channel stats
    for i in range(6):
        v = window[:, i].astype(np.float64, copy=False)
        feats.extend(
            [
                float(v.mean()),
                float(v.std(ddof=0)),
                float(v.min()),
                float(v.max()),
                float(np.mean(v * v)),
            ]
        )

    accel = window[:, 0:3].astype(np.float64, copy=False)
    gyro = window[:, 3:6].astype(np.float64, copy=False)
    acc_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    for v in (acc_mag, gyro_mag):
        feats.extend(
            [
                float(v.mean()),
                float(v.std(ddof=0)),
                float(v.min()),
                float(v.max()),
                float(np.mean(v * v)),
            ]
        )

    # Correlations within accel and gyro + between magnitudes
    feats.append(_safe_corr(accel[:, 0], accel[:, 1]))
    feats.append(_safe_corr(accel[:, 0], accel[:, 2]))
    feats.append(_safe_corr(accel[:, 1], accel[:, 2]))

    feats.append(_safe_corr(gyro[:, 0], gyro[:, 1]))
    feats.append(_safe_corr(gyro[:, 0], gyro[:, 2]))
    feats.append(_safe_corr(gyro[:, 1], gyro[:, 2]))

    feats.append(_safe_corr(acc_mag, gyro_mag))

    return np.array(feats, dtype=np.float32)


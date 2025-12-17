from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularWearableDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    groups: np.ndarray
    devices: np.ndarray | None


def load_aw_fb_tabular_csv(
    csv_path: str | Path,
    *,
    device: str | None = None,
    include_demographics: bool = False,
) -> TabularWearableDataset:
    """Load the Apple Watch / Fitbit tabular activity dataset from CSV.

    Supports:
    - `data_for_weka_aw.csv` (Apple Watch only)
    - `data_for_weka_fb.csv` (Fitbit only)
    - `aw_fb_data.csv` (combined; includes `device` column)

    Parameters
    ----------
    device:
        Optional filter for combined CSVs. Accepted values:
        - "apple watch"
        - "fitbit"
    include_demographics:
        Include age/gender/height/weight features. Defaults to False.

    Returns
    -------
    TabularWearableDataset
        Features (`x`), labels (`y`), feature names, class names, `groups`
        (participant proxy), and optional `devices`.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    label_col = "activity_trimmed" if "activity_trimmed" in df.columns else "activity" if "activity" in df.columns else None
    if label_col is None:
        raise ValueError(f"Could not find activity label column in {csv_path}. Expected 'activity' or 'activity_trimmed'.")

    if "device" in df.columns:
        if device is not None:
            df = df[df["device"].str.lower() == device.lower()].copy()
        devices = df["device"].astype(str).to_numpy()
    else:
        devices = None

    # Participant proxy (no explicit subject column). This is useful for group splits.
    demo_cols = [c for c in ("age", "gender", "height", "weight") if c in df.columns]
    if len(demo_cols) != 4:
        raise ValueError(
            f"Expected demographic columns {('age','gender','height','weight')} in {csv_path}, found {demo_cols}."
        )

    # Make a stable per-row group id.
    demo = df[demo_cols].copy()
    if "weight" in demo.columns:
        demo["weight"] = demo["weight"].astype(float).round(3)
    groups = demo.astype(str).agg("|".join, axis=1).to_numpy()

    # Prepare features
    drop_cols = {label_col, "Unnamed: 0", "X1"}
    if not include_demographics:
        drop_cols |= set(demo_cols)
    if "device" in df.columns:
        drop_cols.add("device")

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if feature_df.empty:
        raise ValueError("No feature columns remain after filtering/dropping columns.")

    # Convert all features to numeric; fail loudly if unexpected strings are present.
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="raise")

    x = feature_df.to_numpy(dtype=np.float32)
    y = df[label_col].astype(str).to_numpy()

    # Canonical class order for metrics/plots
    class_names = sorted(pd.unique(y).tolist())

    return TabularWearableDataset(
        x=x,
        y=y,
        feature_names=list(feature_df.columns),
        class_names=class_names,
        groups=groups,
        devices=devices,
    )


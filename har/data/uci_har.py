from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

UCI_HAR_URL: Final[str] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)


@dataclass(frozen=True)
class UciHarDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]


def _default_activity_map() -> dict[int, str]:
    return {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }


def download_uci_har(data_dir: str | Path = "data", force: bool = False) -> Path:
    """Download + extract the UCI HAR dataset into `data_dir`.

    Returns the extracted dataset root directory:
      `<data_dir>/UCI HAR Dataset/`
    """

    data_dir = Path(data_dir)
    dataset_root = data_dir / "UCI HAR Dataset"
    if dataset_root.exists() and not force:
        return dataset_root

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "uci_har.zip"

    if force and dataset_root.exists():
        # Keep it simple: remove only on extract failure; otherwise overwrite via zip extract.
        pass

    if force or not zip_path.exists():
        urlretrieve(UCI_HAR_URL, zip_path)  # noqa: S310 - expected dataset download

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    if not dataset_root.exists():
        raise RuntimeError(f"Expected dataset at {dataset_root}, but it was not found after extraction.")

    return dataset_root


def load_uci_har(
    data_dir: str | Path = "data",
    *,
    download: bool = True,
    activity_map: dict[int, str] | None = None,
) -> UciHarDataset:
    """Load UCI HAR dataset features/labels.

    Notes:
    - Uses the precomputed 561-feature vectors provided by the dataset.
    - Labels are returned as strings (e.g. 'WALKING').
    """

    data_dir = Path(data_dir)
    dataset_root = data_dir / "UCI HAR Dataset"

    if not dataset_root.exists():
        if not download:
            raise FileNotFoundError(
                f"UCI HAR dataset not found at {dataset_root}. "
                "Set download=True to download it automatically."
            )
        dataset_root = download_uci_har(data_dir)

    activity_map = activity_map or _default_activity_map()

    x_train = pd.read_csv(dataset_root / "train" / "X_train.txt", sep=r"\s+", header=None).to_numpy()
    y_train_raw = pd.read_csv(dataset_root / "train" / "y_train.txt", header=None)[0].to_numpy()

    x_test = pd.read_csv(dataset_root / "test" / "X_test.txt", sep=r"\s+", header=None).to_numpy()
    y_test_raw = pd.read_csv(dataset_root / "test" / "y_test.txt", header=None)[0].to_numpy()

    y_train = np.vectorize(activity_map.get)(y_train_raw)
    y_test = np.vectorize(activity_map.get)(y_test_raw)

    class_names = tuple(activity_map[i] for i in sorted(activity_map))
    return UciHarDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
    )


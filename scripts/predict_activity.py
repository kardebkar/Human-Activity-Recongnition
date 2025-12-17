#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from har.data.synthetic_motion import create_synthetic_motion_dataset
from har.data.uci_har import load_uci_har
from har.models.realdata_transformer import RealDataMotionTransformer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained transformer HAR model.")
    parser.add_argument("--model-path", required=True, help="Path to a .pt artifact saved by train_activity_model.py")
    parser.add_argument("--dataset", choices=["uci_har", "synthetic"], default="uci_har")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow dataset download (uci_har only).",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of samples to predict.")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cpu, cuda). Default: auto.")
    return parser.parse_args()


def _scale_features(x: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0, 1.0, scale)
    return (x - mean) / safe_scale


def main() -> int:
    args = _parse_args()
    model_path = Path(args.model_path)

    import torch

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PyTorch 2.6+ defaults to `weights_only=True`, so keep artifacts pickle-free.
    try:
        artifact = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location=device)

    class_names = artifact["class_names"]
    input_dim = int(artifact["input_dim"])
    hidden_dim = int(artifact["hidden_dim"])

    model = RealDataMotionTransformer(input_dim=input_dim, num_classes=len(class_names), hidden_dim=hidden_dim).to(device)
    model.load_state_dict(artifact["state_dict"])

    if args.dataset == "uci_har":
        ds = load_uci_har(args.data_dir, download=args.download)
        x = ds.x_test
        y = ds.y_test
    else:
        ds = create_synthetic_motion_dataset(n_samples=500, seed=42)
        x = ds.x
        y = ds.y

    n = min(args.n, len(x))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=n, replace=False)

    if "scaler_mean" not in artifact or "scaler_scale" not in artifact:
        raise SystemExit(
            "This model artifact is missing scaler stats. Re-train with the latest "
            "`scripts/train_activity_model.py` to produce a safe, portable checkpoint."
        )

    mean = np.asarray(artifact["scaler_mean"], dtype=np.float32)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float32)
    x_scaled = _scale_features(x[idx].astype(np.float32), mean=mean, scale=scale)
    x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    preds = model.predict_with_description(x_t, class_names)

    for i, pred in enumerate(preds):
        actual = str(y[idx[i]])
        status = "OK" if pred.activity == actual else "WRONG"
        print(f"[{status}] actual={actual}  predicted={pred.activity}  conf={pred.confidence:.2f}")
        print(f"       {pred.description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

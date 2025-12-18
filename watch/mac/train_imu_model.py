#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from har.streaming.features import extract_features, feature_names
from har.train.evaluation import evaluate_classification
from har.train.plots import plot_confusion_matrix
from har.train.utils import ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular classifier on raw watch IMU CSV (ax..gz + label).")
    parser.add_argument("--raw-csv", required=True, help="CSV with columns ax,ay,az,gx,gy,gz,label (timestamp optional).")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-col", default=None, help="Optional column for subject/session grouping.")

    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--min-label-purity", type=float, default=0.8, help="Skip windows with mixed labels below this fraction.")

    parser.add_argument("--model", choices=["rf", "extra_trees"], default="rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def _majority_label(labels: np.ndarray) -> tuple[str, float]:
    uniq, counts = np.unique(labels, return_counts=True)
    idx = int(np.argmax(counts))
    label = str(uniq[idx])
    purity = float(counts[idx]) / float(labels.size) if labels.size else 0.0
    return label, purity


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)

    outputs_dir = ensure_dir(args.outputs_dir)
    models_dir = ensure_dir(args.models_dir)

    df = pd.read_csv(args.raw_csv)
    required = ["ax", "ay", "az", "gx", "gy", "gz", args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {args.raw_csv}: {missing}")

    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(str)

    for c in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    x_raw = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
    y_raw = df[args.label_col].to_numpy(dtype=object)
    groups = df[args.group_col].astype(str).to_numpy(dtype=object) if args.group_col and args.group_col in df.columns else None

    window_size = int(round(args.window_seconds * args.sample_rate_hz))
    hop_size = int(round(args.hop_seconds * args.sample_rate_hz))
    if window_size <= 0 or hop_size <= 0 or hop_size > window_size:
        raise SystemExit("Invalid window/hop settings.")

    feats: list[np.ndarray] = []
    labels: list[str] = []
    window_groups: list[str] = []

    for start in range(0, x_raw.shape[0] - window_size + 1, hop_size):
        end = start + window_size
        window_x = x_raw[start:end]
        window_y = y_raw[start:end]
        label, purity = _majority_label(window_y)
        if purity < args.min_label_purity:
            continue

        feats.append(extract_features(window_x))
        labels.append(label)
        if groups is not None:
            window_groups.append(str(groups[start]))

    if not feats:
        raise SystemExit("No windows were created. Check labels, window size, and min-label-purity.")

    x = np.stack(feats, axis=0)
    y = np.array(labels, dtype=object)

    class_names = sorted(set(y.tolist()))
    print(f"Raw samples: {len(df):,}")
    print(f"Windows: {len(y):,} | Window: {args.window_seconds:.2f}s | Hop: {args.hop_seconds:.2f}s | Features: {x.shape[1]}")
    print("Classes:", ", ".join(class_names))
    print()

    if groups is not None:
        g = np.array(window_groups, dtype=object)
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(x, y, groups=g))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(x, y))

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=600, random_state=args.seed, n_jobs=-1)
    else:
        model = ExtraTreesClassifier(n_estimators=800, random_state=args.seed, n_jobs=-1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = evaluate_classification(y_test, y_pred, class_names=class_names)
    print(f"Accuracy: {metrics.accuracy * 100:.2f}%")

    out_cm = outputs_dir / "watch_imu_confusion_matrix.png"
    plot_confusion_matrix(metrics.confusion_normalized, class_names, title="Confusion Matrix (watch_imu)", out_path=out_cm)
    (outputs_dir / "watch_imu_classification_report.json").write_text(json.dumps(metrics.report, indent=2), encoding="utf-8")
    print(f"Saved: {out_cm}")

    model_path = models_dir / f"watch_imu_{args.model}.joblib"
    try:
        import joblib

        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names(),
                "class_names": class_names,
                "sample_rate_hz": float(args.sample_rate_hz),
                "window_seconds": float(args.window_seconds),
                "hop_seconds": float(args.hop_seconds),
            },
            model_path,
        )
        print(f"Saved model: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not save model via joblib: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

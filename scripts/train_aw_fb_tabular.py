#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import GroupShuffleSplit

from har.data.aw_fb_tabular import load_aw_fb_tabular_csv
from har.train.evaluation import top_confusions
from har.train.mlp import train_mlp_classifier
from har.train.plots import plot_confusion_matrix
from har.train.utils import ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an activity classifier on Apple Watch / Fitbit tabular CSVs.")
    parser.add_argument("--csv-path", required=True, help="Path to aw_fb_data.csv or data_for_weka_*.csv")
    parser.add_argument("--device", default=None, choices=[None, "apple watch", "fitbit"], help="Filter device (combined CSV only).")
    parser.add_argument(
        "--include-demographics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include age/gender/height/weight as features (default: False).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)

    outputs_dir = ensure_dir(args.outputs_dir)
    models_dir = ensure_dir(args.models_dir)

    ds = load_aw_fb_tabular_csv(
        args.csv_path,
        device=args.device,
        include_demographics=args.include_demographics,
    )

    # Group split by participant proxy to reduce leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(ds.x, ds.y, groups=ds.groups))

    x_train, y_train = ds.x[train_idx], ds.y[train_idx]
    x_test, y_test = ds.x[test_idx], ds.y[test_idx]

    print(f"CSV: {args.csv_path}")
    if args.device:
        print(f"Device filter: {args.device}")
    print(f"Samples: {len(ds.x):,} | Features: {ds.x.shape[1]}")
    print(f"Train: {len(x_train):,} | Test: {len(x_test):,}")
    print("Labels:", ", ".join(ds.class_names))
    print()

    result = train_mlp_classifier(x_train, y_train, x_test, y_test, random_state=args.seed)
    class_order = [str(c) for c in result.label_encoder.classes_]

    print(f"Accuracy: {result.metrics.accuracy * 100:.2f}%")
    pairs = top_confusions(result.metrics.confusion, class_order, k=8)
    if pairs:
        print("\nTop confusions:")
        for p in pairs:
            print(f"- {p['actual']} -> {p['predicted']}: {p['count']} ({p['pct']:.1f}%)")

    tag = Path(args.csv_path).stem
    if args.device:
        tag = f"{tag}_{args.device.replace(' ', '_')}"
    if args.include_demographics:
        tag = f"{tag}_with_demo"

    out_cm = outputs_dir / f"{tag}_mlp_confusion_matrix.png"
    plot_confusion_matrix(
        result.metrics.confusion_normalized,
        class_order,
        title=f"Confusion Matrix ({tag})",
        out_path=out_cm,
    )

    (outputs_dir / f"{tag}_mlp_classification_report.json").write_text(
        json.dumps(result.metrics.report, indent=2),
        encoding="utf-8",
    )

    # Save model
    try:
        import joblib

        joblib.dump(
            {
                "model": result.model,
                "scaler": result.scaler,
                "label_encoder": result.label_encoder,
                "feature_names": ds.feature_names,
                "class_names": class_order,
            },
            models_dir / f"{tag}_mlp.joblib",
        )
        print(f"\nSaved model: {models_dir / f'{tag}_mlp.joblib'}")
    except Exception as e:
        print(f"\n⚠️ Could not save model via joblib: {e}")

    print(f"Saved outputs: {out_cm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


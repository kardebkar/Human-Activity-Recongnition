#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from har.data.uci_har import load_uci_har
from har.train.evaluation import top_confusions
from har.train.mlp import train_mlp_classifier
from har.train.plots import plot_classification_overview
from har.train.utils import ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze UCI HAR dataset with a classic MLP baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow dataset download from UCI.",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)

    outputs_dir = ensure_dir(args.outputs_dir)

    ds = load_uci_har(args.data_dir, download=args.download)
    result = train_mlp_classifier(
        ds.x_train,
        ds.y_train,
        ds.x_test,
        ds.y_test,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    class_order = [str(c) for c in result.label_encoder.classes_]

    print(f"UCI HAR baseline accuracy: {result.metrics.accuracy * 100:.2f}%")

    pairs = top_confusions(result.metrics.confusion, class_order, k=5)
    if pairs:
        print("\nTop confusion pairs:")
        for p in pairs:
            print(f"- {p['actual']} -> {p['predicted']}: {p['count']} ({p['pct']:.1f}%)")

    out_png = outputs_dir / "uci_har_analysis.png"
    plot_classification_overview(
        confusion_normalized=result.metrics.confusion_normalized,
        report=result.metrics.report,
        class_names=class_order,
        accuracy=result.metrics.accuracy,
        title="UCI HAR Motion Recognition Analysis",
        out_path=out_png,
    )

    (outputs_dir / "uci_har_classification_report.json").write_text(
        json.dumps(result.metrics.report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

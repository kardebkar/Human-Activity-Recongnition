#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from har.data.synthetic_motion import create_synthetic_motion_dataset
from har.data.uci_har import load_uci_har
from har.train.evaluation import top_confusions
from har.train.mlp import train_mlp_classifier
from har.train.plots import plot_confusion_matrix, plot_training_curves
from har.train.utils import ensure_dir, set_seed


def _save_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HAR activity classifier.")

    parser.add_argument(
        "--dataset",
        choices=["uci_har", "synthetic"],
        default="uci_har",
        help="Dataset to use (uci_har downloads from UCI by default).",
    )
    parser.add_argument("--data-dir", default="data", help="Where datasets are stored/downloaded.")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow dataset download (uci_har only).",
    )

    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp", help="Model type.")
    parser.add_argument("--seed", type=int, default=42)

    # Torch model options
    parser.add_argument("--epochs", type=int, default=10, help="Transformer epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Transformer batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Transformer learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer hidden dim.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. 'cpu', 'cuda'). Default: auto.",
    )

    parser.add_argument("--outputs-dir", default="outputs", help="Where plots/reports are saved.")
    parser.add_argument("--models-dir", default="models", help="Where trained models are saved.")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)

    outputs_dir = ensure_dir(args.outputs_dir)
    models_dir = ensure_dir(args.models_dir)

    if args.dataset == "uci_har":
        ds = load_uci_har(args.data_dir, download=args.download)
        x_train, y_train, x_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        class_names = list(ds.class_names)
    else:
        ds = create_synthetic_motion_dataset(n_samples=2000, seed=args.seed)
        x = ds.x
        y = ds.y

        # Simple split (avoid pulling in extra sklearn dependency here beyond what we already use in training).
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(x))
        split = int(0.8 * len(x))
        train_idx, test_idx = indices[:split], indices[split:]
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        class_names = list(ds.class_names)

    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(x_train):,} samples | Test: {len(x_test):,} samples | Features: {x_train.shape[1]}")
    print(f"Classes: {class_names}")
    print()

    model_tag = f"{args.dataset}_{args.model}"

    if args.model == "mlp":
        result = train_mlp_classifier(x_train, y_train, x_test, y_test, random_state=args.seed)
        class_order = [str(c) for c in result.label_encoder.classes_]

        print(f"Accuracy: {result.metrics.accuracy * 100:.2f}%")
        pairs = top_confusions(result.metrics.confusion, class_order, k=5)
        if pairs:
            print("\nTop confusions:")
            for p in pairs:
                print(f"- {p['actual']} -> {p['predicted']}: {p['count']} ({p['pct']:.1f}%)")

        cm_path = outputs_dir / f"{model_tag}_confusion_matrix.png"
        plot_confusion_matrix(
            result.metrics.confusion_normalized,
            class_order,
            title=f"Confusion Matrix ({model_tag})",
            out_path=cm_path,
        )
        _save_report(result.metrics.report, outputs_dir / f"{model_tag}_classification_report.json")

        # Save model (pickle via joblib)
        try:
            import joblib

            joblib.dump(
                {
                    "model": result.model,
                    "scaler": result.scaler,
                    "label_encoder": result.label_encoder,
                    "class_names": class_order,
                },
                models_dir / f"{model_tag}.joblib",
            )
            print(f"\nSaved model: {models_dir / f'{model_tag}.joblib'}")
        except Exception as e:  # pragma: no cover - depends on joblib availability
            print(f"\n⚠️ Could not save model via joblib: {e}")

        print(f"Saved outputs: {cm_path}")
        return 0

    # Transformer
    from har.train.torch_trainer import train_transformer_classifier

    result = train_transformer_classifier(
        x_train,
        y_train,
        x_test,
        y_test,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )

    class_order = [str(c) for c in result.label_encoder.classes_]

    print(f"Accuracy: {result.metrics.accuracy * 100:.2f}%")
    pairs = top_confusions(result.metrics.confusion, class_order, k=5)
    if pairs:
        print("\nTop confusions:")
        for p in pairs:
            print(f"- {p['actual']} -> {p['predicted']}: {p['count']} ({p['pct']:.1f}%)")

    curves_path = outputs_dir / f"{model_tag}_training_curves.png"
    plot_training_curves(
        result.history["train_loss"],
        result.history["train_acc"],
        result.history["val_acc"],
        title=f"Training Curves ({model_tag})",
        out_path=curves_path,
    )

    cm_path = outputs_dir / f"{model_tag}_confusion_matrix.png"
    plot_confusion_matrix(
        result.metrics.confusion_normalized,
        class_order,
        title=f"Confusion Matrix ({model_tag})",
        out_path=cm_path,
    )

    _save_report(result.metrics.report, outputs_dir / f"{model_tag}_classification_report.json")

    # Save torch artifact
    try:
        import torch

        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                # IMPORTANT: class order matches the model outputs.
                "class_names": class_order,
                # Avoid pickling sklearn objects so torch.load(weights_only=True) works.
                "scaler_mean": result.scaler.mean_.tolist(),
                "scaler_scale": result.scaler.scale_.tolist(),
            },
            models_dir / f"{model_tag}.pt",
        )
        print(f"\nSaved model: {models_dir / f'{model_tag}.pt'}")
    except Exception as e:  # pragma: no cover - torch not installed
        print(f"\n⚠️ Could not save torch model: {e}")

    print(f"Saved outputs: {cm_path} | {curves_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

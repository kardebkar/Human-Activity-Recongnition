#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from har.data.aw_fb_tabular import TabularWearableDataset, load_aw_fb_tabular_csv
from har.train.evaluation import ClassificationMetrics, evaluate_classification
from har.train.plots import plot_classification_overview, plot_confusion_matrix
from har.train.utils import ensure_dir, set_seed


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    label_set: str
    split: str
    model: str
    metrics: ClassificationMetrics


def _collapse_labels_4(y: np.ndarray) -> np.ndarray:
    mapping = {
        "Lying": "Lying",
        "Sitting": "Sitting",
        "Self Pace walk": "Walking",
        "Running 3 METs": "Running",
        "Running 5 METs": "Running",
        "Running 7 METs": "Running",
    }
    return np.array([mapping.get(str(v), str(v)) for v in y], dtype=object)


def _split_indices(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split: str,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "random":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y))
        return train_idx, test_idx
    if split == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        return train_idx, test_idx
    raise ValueError(f"Unknown split type: {split}. Expected 'random' or 'group'.")


def _fit_predict(
    *,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    if model_name == "dummy":
        model = DummyClassifier(strategy="most_frequent")
    elif model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, n_jobs=None),
        )
    elif model_name == "svm_rbf":
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale"),
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=700,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(x_train, y_train)
    return model.predict(x_test)


def _metrics_to_summary(metrics: ClassificationMetrics) -> dict[str, Any]:
    macro_f1 = float(metrics.report.get("macro avg", {}).get("f1-score", 0.0))
    weighted_f1 = float(metrics.report.get("weighted avg", {}).get("f1-score", 0.0))
    macro_recall = float(metrics.report.get("macro avg", {}).get("recall", 0.0))
    macro_precision = float(metrics.report.get("macro avg", {}).get("precision", 0.0))
    return {
        "accuracy": float(metrics.accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
    }


def _dataset_overview(ds: TabularWearableDataset) -> dict[str, Any]:
    y = ds.y.astype(str)
    labels, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)[::-1]
    label_counts = {str(labels[i]): int(counts[i]) for i in order}

    group_sizes = {}
    unique_groups, group_counts = np.unique(ds.groups.astype(str), return_counts=True)
    group_sizes["n_groups"] = int(len(unique_groups))
    group_sizes["min"] = int(group_counts.min()) if len(group_counts) else 0
    group_sizes["max"] = int(group_counts.max()) if len(group_counts) else 0
    group_sizes["median"] = float(np.median(group_counts)) if len(group_counts) else 0.0

    nan_count = int(np.isnan(ds.x).sum()) if np.issubdtype(ds.x.dtype, np.floating) else 0
    zero_var = [name for name, std in zip(ds.feature_names, ds.x.std(axis=0), strict=True) if float(std) == 0.0]

    return {
        "n_samples": int(ds.x.shape[0]),
        "n_features": int(ds.x.shape[1]),
        "label_counts": label_counts,
        "group_stats": group_sizes,
        "nan_values": nan_count,
        "zero_variance_features": zero_var,
    }


def _device_shift_stats(ds: TabularWearableDataset) -> dict[str, Any] | None:
    if ds.devices is None:
        return None
    devices = ds.devices.astype(str)
    uniq = sorted(set(devices.tolist()))
    if len(uniq) < 2:
        return None

    stats: dict[str, Any] = {"devices": uniq, "feature_mean_std": {}}
    for device in uniq:
        mask = devices == device
        x = ds.x[mask]
        stats["feature_mean_std"][device] = {
            "mean": {n: float(m) for n, m in zip(ds.feature_names, x.mean(axis=0), strict=True)},
            "std": {n: float(s) for n, s in zip(ds.feature_names, x.std(axis=0), strict=True)},
        }

    # Simple “shift score”: abs(mean_a - mean_b) / pooled_std for each feature (two devices only).
    if len(uniq) == 2:
        a, b = uniq
        mean_a = np.array([stats["feature_mean_std"][a]["mean"][n] for n in ds.feature_names], dtype=np.float64)
        mean_b = np.array([stats["feature_mean_std"][b]["mean"][n] for n in ds.feature_names], dtype=np.float64)
        std_a = np.array([stats["feature_mean_std"][a]["std"][n] for n in ds.feature_names], dtype=np.float64)
        std_b = np.array([stats["feature_mean_std"][b]["std"][n] for n in ds.feature_names], dtype=np.float64)
        pooled = np.sqrt((std_a**2 + std_b**2) / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.abs(mean_a - mean_b) / pooled
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        top = np.argsort(score)[::-1][:8]
        stats["top_shift_features"] = [
            {"feature": ds.feature_names[i], "score": float(score[i]), "mean_a": float(mean_a[i]), "mean_b": float(mean_b[i])}
            for i in top
        ]
    return stats


def _run_dataset_experiments(
    *,
    ds: TabularWearableDataset,
    dataset_name: str,
    outputs_dir: Path,
    seed: int,
    test_size: float,
) -> tuple[list[ExperimentResult], dict[str, Any]]:
    results: list[ExperimentResult] = []

    overview = _dataset_overview(ds)
    device_stats = _device_shift_stats(ds)
    if device_stats is not None:
        overview["device_shift"] = device_stats

    label_sets: list[tuple[str, np.ndarray]] = [
        ("6_class", ds.y.astype(str)),
        ("4_class_collapsed", _collapse_labels_4(ds.y.astype(str))),
    ]
    splits = ["random", "group"]
    models = ["dummy", "logreg", "svm_rbf", "rf", "extra_trees"]

    for label_set_name, y_all in label_sets:
        class_names = sorted(set(y_all.tolist()))
        for split in splits:
            train_idx, test_idx = _split_indices(
                x=ds.x,
                y=y_all,
                groups=ds.groups,
                split=split,
                test_size=test_size,
                seed=seed,
            )
            x_train, y_train = ds.x[train_idx], y_all[train_idx]
            x_test, y_test = ds.x[test_idx], y_all[test_idx]

            for model_name in models:
                y_pred = _fit_predict(model_name=model_name, x_train=x_train, y_train=y_train, x_test=x_test, seed=seed)
                metrics = evaluate_classification(y_test, y_pred, class_names=class_names)
                results.append(
                    ExperimentResult(
                        dataset=dataset_name,
                        label_set=label_set_name,
                        split=split,
                        model=model_name,
                        metrics=metrics,
                    )
                )

            # Save one overview plot per split/label-set using the strongest baseline (extra_trees tends to be strong).
            best = max(
                (r for r in results if r.dataset == dataset_name and r.label_set == label_set_name and r.split == split),
                key=lambda r: r.metrics.accuracy,
            )
            tag = f"{dataset_name}_{label_set_name}_{split}_{best.model}"
            out_overview = outputs_dir / f"{tag}_overview.png"
            plot_classification_overview(
                confusion_normalized=best.metrics.confusion_normalized,
                report=best.metrics.report,
                class_names=class_names,
                accuracy=best.metrics.accuracy,
                title=f"{dataset_name} ({label_set_name}, {split}, {best.model})",
                out_path=out_overview,
            )
            out_cm = outputs_dir / f"{tag}_confusion_matrix.png"
            plot_confusion_matrix(
                best.metrics.confusion_normalized,
                class_names,
                title=f"Confusion Matrix ({tag})",
                out_path=out_cm,
            )

    # Cross-device generalization (only for combined dataset with device labels).
    if ds.devices is not None and set(ds.devices.astype(str).tolist()) >= {"apple watch", "fitbit"}:
        devices = ds.devices.astype(str)
        for label_set_name, y_all in label_sets:
            class_names = sorted(set(y_all.tolist()))
            for train_device, test_device in [("apple watch", "fitbit"), ("fitbit", "apple watch")]:
                train_mask = devices == train_device
                test_mask = devices == test_device
                x_train, y_train = ds.x[train_mask], y_all[train_mask]
                x_test, y_test = ds.x[test_mask], y_all[test_mask]
                for model_name in ["logreg", "rf", "extra_trees"]:
                    y_pred = _fit_predict(model_name=model_name, x_train=x_train, y_train=y_train, x_test=x_test, seed=seed)
                    metrics = evaluate_classification(y_test, y_pred, class_names=class_names)
                    results.append(
                        ExperimentResult(
                            dataset=f"{dataset_name}:{train_device}→{test_device}",
                            label_set=label_set_name,
                            split="cross_device",
                            model=model_name,
                            metrics=metrics,
                        )
                    )

    summary: dict[str, Any] = {"dataset": dataset_name, "overview": overview, "experiments": []}
    for r in results:
        summary["experiments"].append(
            {
                "dataset": r.dataset,
                "label_set": r.label_set,
                "split": r.split,
                "model": r.model,
                "metrics": _metrics_to_summary(r.metrics),
            }
        )
    return results, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the Dataverse Apple Watch / Fitbit tabular activity CSVs.")
    parser.add_argument(
        "--dataverse-dir",
        default=None,
        help="Directory containing data_for_weka_aw.csv, data_for_weka_fb.csv, and aw_fb_data.csv.",
    )
    parser.add_argument("--aw-csv", default=None, help="Path to data_for_weka_aw.csv")
    parser.add_argument("--fb-csv", default=None, help="Path to data_for_weka_fb.csv")
    parser.add_argument("--combined-csv", default=None, help="Path to aw_fb_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--outputs-dir", default="outputs")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if args.dataverse_dir:
        d = Path(args.dataverse_dir)
        paths["aw"] = d / "data_for_weka_aw.csv"
        paths["fb"] = d / "data_for_weka_fb.csv"
        paths["combined"] = d / "aw_fb_data.csv"

    if args.aw_csv:
        paths["aw"] = Path(args.aw_csv)
    if args.fb_csv:
        paths["fb"] = Path(args.fb_csv)
    if args.combined_csv:
        paths["combined"] = Path(args.combined_csv)

    existing = {k: v for k, v in paths.items() if v.exists()}
    if not existing:
        raise FileNotFoundError(
            "No CSVs found. Provide --dataverse-dir or explicit --aw-csv/--fb-csv/--combined-csv paths."
        )
    return existing


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)
    outputs_dir = ensure_dir(args.outputs_dir)

    paths = _resolve_paths(args)
    all_summaries: dict[str, Any] = {"seed": args.seed, "test_size": args.test_size, "runs": []}

    for name, path in paths.items():
        ds = load_aw_fb_tabular_csv(path, include_demographics=False)
        results, summary = _run_dataset_experiments(
            ds=ds,
            dataset_name=name,
            outputs_dir=outputs_dir,
            seed=args.seed,
            test_size=args.test_size,
        )

        # Print a compact leaderboard (accuracy).
        print(f"\n== {name} ==")
        overview = summary["overview"]
        print(f"Samples: {overview['n_samples']:,} | Features: {overview['n_features']}")
        print(f"Groups (proxy): {overview['group_stats']['n_groups']} | group size {overview['group_stats']['min']}–{overview['group_stats']['max']}")
        print("Labels:", ", ".join(overview["label_counts"].keys()))

        rows = []
        for r in results:
            if r.dataset != name:
                continue
            rows.append((r.label_set, r.split, r.model, r.metrics.accuracy))
        rows.sort(key=lambda t: float(t[3]), reverse=True)
        for label_set, split, model, acc in rows[:8]:
            print(f"- {label_set:16s} | {split:6s} | {model:11s} | {acc*100:6.2f}%")

        all_summaries["runs"].append(summary)

    out_json = outputs_dir / "aw_fb_dataset_analysis.json"
    out_json.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


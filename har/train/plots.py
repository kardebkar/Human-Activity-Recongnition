from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_MPLCONFIGDIR = Path.cwd() / ".cache" / "matplotlib"
if "MPLCONFIGDIR" not in os.environ:
    _DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_DEFAULT_MPLCONFIGDIR)

# Headless-safe backend (avoids macOS GUI backends in sandboxed environments).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    confusion_normalized: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    *,
    title: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        confusion_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
    *,
    title: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "g-", label="Train Acc")
    axes[1].plot(epochs, val_acc, "r-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classification_overview(
    *,
    confusion_normalized: np.ndarray,
    report: dict,
    class_names: list[str] | tuple[str, ...],
    accuracy: float,
    title: str,
    out_path: str | Path,
) -> None:
    """Create a 2x2 overview similar to the notebook analysis."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_names_list = list(class_names)
    per_class_acc = np.diag(confusion_normalized) * 100.0
    f1_scores = [float(report.get(name, {}).get("f1-score", 0.0)) * 100.0 for name in class_names_list]
    supports = [int(report.get(name, {}).get("support", 0)) for name in class_names_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion matrix
    ax1 = axes[0, 0]
    sns.heatmap(
        confusion_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[n[:10] for n in class_names_list],
        yticklabels=[n[:10] for n in class_names_list],
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix (Normalized)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Per-class accuracy
    ax2 = axes[0, 1]
    colors = ["green" if acc >= 95 else "orange" if acc >= 90 else "red" for acc in per_class_acc]
    bars = ax2.bar(range(len(class_names_list)), per_class_acc, color=colors)
    ax2.set_xticks(range(len(class_names_list)))
    ax2.set_xticklabels([n[:12] for n in class_names_list], rotation=45, ha="right")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Per-Class Accuracy")
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, per_class_acc):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1, f"{acc:.1f}%", ha="center")

    # F1 scores
    ax3 = axes[1, 0]
    ax3.barh(range(len(class_names_list)), f1_scores, color="coral")
    ax3.set_yticks(range(len(class_names_list)))
    ax3.set_yticklabels(class_names_list)
    ax3.set_xlabel("F1 Score (%)")
    ax3.set_title("F1 Scores by Activity")
    ax3.grid(True, alpha=0.3, axis="x")

    # Support distribution
    ax4 = axes[1, 1]
    ax4.pie(supports, labels=class_names_list, autopct="%1.1f%%")
    ax4.set_title("Test Set Distribution")

    plt.suptitle(f"{title} (Accuracy: {accuracy * 100:.1f}%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from har.models.realdata_transformer import RealDataMotionTransformer
from har.train.evaluation import ClassificationMetrics, evaluate_classification


@dataclass(frozen=True)
class TorchTrainingResult:
    model: RealDataMotionTransformer
    scaler: StandardScaler
    label_encoder: LabelEncoder
    history: dict[str, list[float]]
    metrics: ClassificationMetrics
    y_pred: np.ndarray


def train_transformer_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> TorchTrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    class_names = [str(c) for c in label_encoder.classes_]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    model = RealDataMotionTransformer(input_dim=x_train.shape[1], num_classes=len(class_names), hidden_dim=hidden_dim).to(
        device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        iterator = train_loader
        if progress:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total += int(batch_y.size(0))
            correct += int((preds == batch_y).sum().item())

            if progress:
                iterator.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.0 * correct / max(1, total):.1f}%")

        scheduler.step()

        # Validation accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_total += int(batch_y.size(0))
                val_correct += int((preds == batch_y).sum().item())

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        val_acc = 100.0 * val_correct / max(1, val_total)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

    # Final predictions
    model.eval()
    preds_list: list[int] = []
    with torch.no_grad():
        for batch_x, _batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds_list.extend(preds)

    y_pred_enc = np.asarray(preds_list, dtype=np.int64)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    metrics = evaluate_classification(y_test, y_pred, class_names=class_names)
    return TorchTrainingResult(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        history=history,
        metrics=metrics,
        y_pred=y_pred,
    )


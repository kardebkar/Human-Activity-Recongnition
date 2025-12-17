from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from har.train.evaluation import ClassificationMetrics, evaluate_classification


@dataclass(frozen=True)
class MlpTrainingResult:
    model: MLPClassifier
    scaler: StandardScaler
    label_encoder: LabelEncoder
    metrics: ClassificationMetrics
    y_pred: np.ndarray


def train_mlp_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    max_iter: int = 200,
    random_state: int = 42,
) -> MlpTrainingResult:
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    class_names = [str(c) for c in label_encoder.classes_]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )

    # NOTE: We label-encode y to avoid a scikit-learn 1.8.0 early-stopping bug
    # where `np.isnan` is called on string class labels inside `_score_with_function`.
    model.fit(x_train_scaled, y_train_enc)

    y_pred_enc = model.predict(x_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    metrics = evaluate_classification(y_test, y_pred, class_names=class_names)

    return MlpTrainingResult(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        metrics=metrics,
        y_pred=y_pred,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class SklearnWindowClassifier:
    def __init__(self, *, model: Any, class_names: list[str], expected_feature_names: list[str] | None = None):
        self.model = model
        self.class_names = list(class_names)
        self.expected_feature_names = list(expected_feature_names) if expected_feature_names is not None else None

        if not hasattr(model, "predict_proba"):
            raise TypeError("Model must support predict_proba().")

    @classmethod
    def load_joblib(cls, path: str | Path) -> "SklearnWindowClassifier":
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            class_names = [str(c) for c in obj.get("class_names", getattr(model, "classes_", []))]
            feature_names = obj.get("feature_names")
            return cls(model=model, class_names=class_names, expected_feature_names=feature_names)

        # Allow raw sklearn estimators if they have classes_.
        model = obj
        class_names = [str(c) for c in getattr(model, "classes_", [])]
        return cls(model=model, class_names=class_names)

    def predict(self, features: np.ndarray) -> Prediction:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))

        # Prefer the estimator's classes_ ordering, fall back to stored class_names.
        classes = getattr(self.model, "classes_", None)
        if classes is not None and len(classes) == len(proba):
            labels = [str(c) for c in classes]
        else:
            labels = self.class_names

        probs = {labels[i]: float(proba[i]) for i in range(len(proba))}
        return Prediction(label=labels[idx], confidence=float(proba[idx]), probabilities=probs)


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActivityPrediction:
    activity: str
    description: str
    confidence: float


MOTION_DESCRIPTIONS: dict[str, list[str]] = {
    "WALKING": [
        "Person walks forward at a steady pace.",
        "Individual strides with regular rhythm.",
        "Human moves ahead with a normal gait.",
        "Someone walks comfortably.",
        "Pedestrian advances smoothly.",
    ],
    "WALKING_UPSTAIRS": [
        "Person climbs stairs upward.",
        "Individual ascends a staircase.",
        "Human walks up the steps.",
        "Someone goes upstairs steadily.",
        "Person moves up a stairway.",
    ],
    "WALKING_DOWNSTAIRS": [
        "Person descends stairs carefully.",
        "Individual walks down steps.",
        "Human goes downstairs.",
        "Someone descends a staircase.",
        "Person moves down stairs.",
    ],
    "SITTING": [
        "Person sits still calmly.",
        "Individual remains seated.",
        "Human rests in a chair.",
        "Someone sits quietly.",
        "Person stays seated.",
    ],
    "STANDING": [
        "Person stands motionless.",
        "Individual remains upright.",
        "Human stands still.",
        "Someone stays standing.",
        "Person holds position.",
    ],
    "LAYING": [
        "Person lies down resting.",
        "Individual reclines horizontally.",
        "Human lies on a surface.",
        "Someone rests lying down.",
        "Person remains laying.",
    ],
    # Extra labels used by the synthetic dataset.
    "RUNNING": [
        "Person runs quickly forward.",
        "Athlete sprints with speed.",
        "Runner moves rapidly.",
        "Individual jogs fast.",
        "Someone dashes ahead.",
    ],
    "JUMPING": [
        "Person jumps up energetically.",
        "Individual leaps upward.",
        "Human bounces vertically.",
        "Someone hops repeatedly.",
        "Athlete jumps high.",
    ],
    "CYCLING": [
        "Person cycles on a bike.",
        "Individual pedals a bicycle.",
        "Cyclist moves forward steadily.",
        "Someone rides at a consistent pace.",
        "Person bikes along.",
    ],
}


class RealDataMotionTransformer(nn.Module):
    """A small MLP + self-attention classifier for vectorized HAR features."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)  # (B, 1, H)
        attended, _ = self.attention(features, features, features)
        attended = attended.squeeze(1)  # (B, H)
        return self.classifier(attended)

    @torch.no_grad()
    def predict_with_description(self, x: torch.Tensor, class_names: list[str] | tuple[str, ...]) -> list[ActivityPrediction]:
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)

        predictions: list[ActivityPrediction] = []
        for row in probs:
            pred_idx = int(torch.argmax(row).item())
            confidence = float(row[pred_idx].item())
            activity = str(class_names[pred_idx])
            description_candidates = MOTION_DESCRIPTIONS.get(activity, [f"{activity} detected."])
            description = str(np.random.choice(description_candidates))
            predictions.append(ActivityPrediction(activity=activity, description=description, confidence=confidence))

        return predictions


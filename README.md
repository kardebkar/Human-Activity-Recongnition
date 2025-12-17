# Human Activity Recognition (HAR)

- **UCI HAR baseline**: scikit-learn MLP on the canonical 561-feature vectors
- **PyTorch classifier**: compact “MLP + self-attention” model on the same feature vectors
- **Optional (GPU/Colab)**: Unsloth/TRL fine-tuning script to map sensor summaries → activity label

Artifacts are written to `data/`, `outputs/`, and `models/` (all `.gitignore`d).

## What’s In Here

- `har/data/uci_har.py`: UCI HAR download + loader
- `har/train/mlp.py`: scikit-learn baseline training
- `har/train/torch_trainer.py`: PyTorch training loop
- `scripts/analyze_uci_har.py`: baseline analysis + overview plot
- `scripts/train_activity_model.py`: train MLP or PyTorch model
- `scripts/predict_activity.py`: run inference with the saved PyTorch model

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (only if you want the PyTorch model):

```bash
pip install -r requirements-torch.txt
```

## Quick Start (Baseline)

Downloads UCI HAR (if missing), trains an MLP baseline, and produces an overview plot:

```bash
python scripts/analyze_uci_har.py --download
```

Expected outputs:
- `outputs/uci_har_analysis.png`
- `outputs/uci_har_classification_report.json`

## Train Models

### MLP baseline (scikit-learn)

```bash
python scripts/train_activity_model.py --dataset uci_har --model mlp --download
```

### Transformer-style classifier (PyTorch)

```bash
python scripts/train_activity_model.py --dataset uci_har --model transformer --epochs 10
```

### Synthetic dataset (no downloads)

```bash
python scripts/train_activity_model.py --dataset synthetic --model transformer --epochs 10
```

## Inference (PyTorch Model)

```bash
python scripts/predict_activity.py --model-path models/uci_har_transformer.pt --dataset uci_har --n 5
```

## Optional: LLM Fine-Tuning (Colab/GPU)

This is intended for Colab/Linux with CUDA:

```bash
pip install -r requirements-llm.txt
python scripts/finetune_activity_llm_unsloth.py
```

If Kaggle download fails, place `train.csv` and `test.csv` at `data/kaggle_har/`.

## Future Directions (Parked)

### Accessibility & Assistive Tech on Apple Watch (Adaptive, Context-Aware UI)

Goal: use wrist sensing (accel/gyro + optional altimeter/pedometer/HR) to adapt an interface for users in motion.

- **Stability first:** sliding windows + hysteresis + an explicit `uncertain` state to avoid mode thrash.
- **Personalization:** short per-user calibration to handle dominant hand/strap fit/gait variance.
- **Interaction policy layer:** map `(activity_state, uncertainty, user_prefs)` → UI mode (large targets, voice-first, haptic confirmations, reduced steps).
- **HCI metrics:** false mode-switches/hour, time-to-stable-mode, task success while moving, NASA‑TLX workload, SUS usability, perceived safety.
- **Privacy by design:** on-device inference, minimal retention, clear user control over data.

### Social Robotics / HRI Framing (ICSR-style)

Position the watch as part of a socially assistive system that improves *when* and *how* an embodied agent communicates:

- **Interruptibility-aware assistance:** time prompts to low-cost moments; choose modality (haptic vs voice vs visual) based on context.
- **Transparent adaptation:** concise “why” explanations (“I switched to large controls because you’re walking”) to improve trust/acceptance.
- **Distributed embodiment:** watch provides context; an embodied agent (phone/tablet avatar or robot) delivers social assistance.
- **Evaluation beyond accuracy:** trust, workload, perceived usefulness, and failure handling under uncertainty.

## Credits & Attribution

- **UCI HAR Dataset** (downloaded by default): “Human Activity Recognition Using Smartphones Data Set”, UCI Machine Learning Repository.  
  Citation (per the dataset `README.txt`):
  > Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.  
  > *A Public Domain Dataset for Human Activity Recognition Using Smartphones.*  
  > ESANN 2013, Bruges, Belgium, 24–26 April 2013.  
  Dataset page: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones  
  Note: the dataset README states that **commercial use is prohibited** — please review the dataset license terms before use.
- **Kaggle mirror (optional)**: `uciml/human-activity-recognition-with-smartphones` (used by the optional LLM script).  
  https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- **Notebook source**: the implementation here is derived from the user-provided notebook stored at `notebooks/Making_the_Most_of_your_Colab_Subscription.ipynb`.
- **Libraries**: scikit-learn, PyTorch, matplotlib/seaborn, Hugging Face Transformers/TRL/PEFT/Datasets, and Unsloth (for the optional LLM workflow).

## License

MIT (see `LICENSE`).

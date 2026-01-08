# Wrist2Avatar: Apple Watch IMU Streaming + Activity Recognition

Main idea: stream **raw wrist IMU** (accelerometer + gyroscope) from Apple Watch to macOS (via an iPhone relay), log labeled CSV, train a baseline model, and run **real-time classification + state stabilization** (for driving an avatar / adaptive UX later).

**Architecture**

Apple Watch (CoreMotion) → WatchConnectivity → iPhone (UDP relay) → Mac (Python receiver/classifier) → (optional Unity/avatar)

## TL;DR (Run the Stream)

1) Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Start the Mac receiver (keep running):

```bash
python watch/mac/run_stream_server.py --listen-port 5500 --log-raw-csv data/watch_raw.csv
```

3) Find your Mac’s LAN IP (use this in the iPhone app; **not** `localhost`):

```bash
ipconfig getifaddr en0
```

4) Open the Xcode project: `watch/xcode/Wrist2AvatarStreamer/Wrist2AvatarStreamer.xcodeproj`
   - Or create your own project using `watch/apple/SETUP_XCODE.md`

5) Run in Xcode (real devices recommended):
   - Run the **iOS app** on a real iPhone → enter Mac IP + port `5500` → **Start Relay** (allow Local Network permission).
   - Run the **watch app** on a real Apple Watch → **Start**.

Notes:
- Real-time WatchConnectivity (`WCSession.isReachable`) requires the counterpart app to be active (keep the iPhone app foreground/unlocked for watch→phone; keep the watch app open for phone→watch label/commands).
- Streaming artifacts are written to `data/`, `outputs/`, and `models/` (all `.gitignore`d).

Also included (secondary):
- **UCI HAR baseline**: scikit-learn MLP on canonical 561-feature vectors
- **PyTorch classifier**: compact “MLP + self-attention” model on the same feature vectors
- **Optional (GPU/Colab)**: Unsloth/TRL fine-tuning script to map sensor summaries → activity label

## What’s In Here

- `har/data/uci_har.py`: UCI HAR download + loader
- `har/train/mlp.py`: scikit-learn baseline training
- `har/train/torch_trainer.py`: PyTorch training loop
- `scripts/analyze_uci_har.py`: baseline analysis + overview plot
- `scripts/analyze_aw_fb_dataset.py`: full analysis for Dataverse Apple Watch / Fitbit CSVs (random vs group splits)
- `scripts/train_activity_model.py`: train MLP or PyTorch model
- `scripts/predict_activity.py`: run inference with the saved PyTorch model
- `watch/mac/run_stream_server.py`: receive watch IMU (UDP JSON) and output stabilized activity state
- `watch/mac/train_imu_model.py`: train a baseline on labeled raw IMU CSV for streaming inference

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

## Smoke Tests

```bash
# 1) Basic import/compile check
python3 -m py_compile $(git ls-files "*.py")

# 2) End-to-end (UCI HAR baseline)
python3 scripts/analyze_uci_har.py --download

# 3) Watch streaming path (no UDP; stdin mode)
python3 watch/mac/generate_fake_imu.py --samples 200 --sample-rate-hz 50 \
  | python3 watch/mac/run_stream_server.py --stdin --sample-rate-hz 50 --window-seconds 2.0 --hop-seconds 0.5
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

## External Wearable CSVs (Apple Watch / Fitbit)

If you have the Dataverse CSVs you mentioned (e.g. `data_for_weka_aw.csv`, `data_for_weka_fb.csv`, `aw_fb_data.csv`), you can train a baseline on them without copying the data into this repo:

```bash
python scripts/train_aw_fb_tabular.py --csv-path /path/to/aw_fb_data.csv
```

Dataset analysis (recommended before modeling):

```bash
python scripts/analyze_aw_fb_dataset.py --dataverse-dir /path/to/dataverse_files
```

Filter to a single device (combined CSV only):

```bash
python scripts/train_aw_fb_tabular.py --csv-path /path/to/aw_fb_data.csv --device "apple watch"
python scripts/train_aw_fb_tabular.py --csv-path /path/to/aw_fb_data.csv --device "fitbit"
```

Notes:
- These CSVs contain **tabular, engineered features** (steps, heart rate, calories, distance, entropy/correlation features) and 6 labels: `Lying`, `Sitting`, `Self Pace walk`, and `Running 3/5/7 METs`.
- There is **no explicit “Standing”** label in these files; for a stand/walk/stairs demo you’ll likely collect wrist IMU data for standing (or treat “stationary” as one class).
- Summary of key findings (leakage risk under random splits, cross-device shift, suggested label mapping) is in `docs/DATAVERSE_AW_FB_ANALYSIS.md`.

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

For a ready-to-write ICSR paper outline, figure plan, and results table templates, see `docs/ICSR2026_PAPER_BLUEPRINT.md`.

## Apple Watch IMU Streaming → macOS (Raw → Classifier → Stable State)

This repo includes a Mac-side UDP receiver and an offline trainer for **raw IMU streaming** (accelerometer/gyroscope) collected on Apple Watch.

Start here:
- `docs/watch/IMU_STREAMING.md`
- `watch/apple/SETUP_XCODE.md` (Apple Watch + iPhone relay scaffold)

End-to-end (Watch → iPhone → Mac):

1) Mac (receiver; keep running):

```bash
source .venv/bin/activate
python watch/mac/run_stream_server.py --listen-port 5500 --log-raw-csv data/watch_raw.csv
```

2) Find your Mac’s LAN IP (use this in the iPhone app; **not** `localhost`):

```bash
ipconfig getifaddr en0
```

3) Xcode:
- Run the **iOS app on a real iPhone**, enter Mac IP + port `5500`, tap **Start Relay** (allow Local Network permission).
- Run the **watch app on a real Watch**, tap **Start**.

Notes:
- `WCSession.isReachable` requires the counterpart app to be active (keep the iPhone app foreground/unlocked for watch→phone; keep the watch app open for phone→watch label/commands).
- A working sample Xcode project is included at `watch/xcode/Wrist2AvatarStreamer/Wrist2AvatarStreamer.xcodeproj`. The drop-in Swift scaffold also lives in `watch/apple/` if you want to integrate into your own app.

Typical workflow:

```bash
# 1) Collect raw samples (from an iPhone relay) to CSV
python watch/mac/run_stream_server.py --listen-port 5500 --log-raw-csv data/watch_raw.csv

# 2) Train a baseline model (requires a labeled CSV)
python watch/mac/train_imu_model.py --raw-csv data/watch_raw_labeled.csv --model rf

# 3) Run real-time inference + stabilization
python watch/mac/run_stream_server.py --listen-port 5500 --model-path models/watch_imu_rf.joblib
```

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
- **Dataverse Apple Watch + Fitbit dataset** (optional, not included): Daniel Fuller (2020), “Replication Data for: Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion.” Harvard Dataverse. DOI: https://doi.org/10.7910/DVN/ZS2Z2J (license: CC0 1.0).
- **Notebook source**: the implementation here is derived from the user-provided notebook stored at `notebooks/Making_the_Most_of_your_Colab_Subscription.ipynb`.
- **Libraries**: scikit-learn, PyTorch, matplotlib/seaborn, Hugging Face Transformers/TRL/PEFT/Datasets, and Unsloth (for the optional LLM workflow).

## License

MIT (see `LICENSE`).

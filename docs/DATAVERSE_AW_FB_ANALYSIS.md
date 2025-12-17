# Dataverse Apple Watch + Fitbit Dataset — Full Analysis Notes

Source dataset (CC0): **“Replication Data for: Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion.”** Harvard Dataverse. DOI: `10.7910/DVN/ZS2Z2J`.

This repo does **not** include the raw CSVs; it contains scripts to analyze/train on a local copy.

## Files & Shapes

From the Dataverse ZIP / local folder:

- `data_for_weka_aw.csv` (Apple Watch only): **3656 rows × 18 columns**
- `data_for_weka_fb.csv` (Fitbit only): **2608 rows × 18 columns**
- `aw_fb_data.csv` (combined; includes a `device` column): **6264 rows × 20 columns**

## Labels (6-class)

All three CSVs share the same activity set:

- `Lying`
- `Sitting`
- `Self Pace walk`
- `Running 3 METs`
- `Running 5 METs`
- `Running 7 METs`

Important for HCI / Unity mapping:
- There is **no explicit `Standing`** label in these CSVs.
- A practical demo mapping is: `Lying→LIE`, `Sitting→SIT`, `Self Pace walk→WALK`, `Running *→RUN` (optionally speed tiers).

## Features (Engineered Tabular)

These are **not raw IMU time-series**. They are engineered per-minute features such as:

- steps, heart rate, calories, distance
- entropy of heart/steps, correlation heart↔steps
- resting heart rate, normalized heart metrics, derived intensity

Because these are engineered features, real-time deployment requires re-computing the same features on-device (or training a model on raw IMU instead).

## Participant / Split Notes (Leakage Risk)

The CSVs do not contain an explicit subject ID. For subject-independent evaluation, this repo uses a **participant proxy**: `(age|gender|height|weight)` per row.

Observed with the provided files:
- Apple Watch: **49 unique proxies**, each with all 6 labels (group sizes 64–89)
- Fitbit: **49 unique proxies**, but many sparse groups (min group size 9)

Dataverse metadata mentions 46 participants; the proxy count mismatch is a known limitation (demographic collisions / rounding / entry variance).

## Reproducible Analysis (Recommended)

Run:

```bash
python scripts/analyze_aw_fb_dataset.py --dataverse-dir /path/to/dataverse_files
```

This writes:
- `outputs/aw_fb_dataset_analysis.json` (all metrics)
- a set of overview plots (confusion matrix + per-class accuracy + F1 + test distribution) per dataset/split

## Key Results (Seed=42, Test=20%)

**Why this matters:** results look strong under a random split, but drop substantially under a group split (proxy subject-independent), indicating leakage risk when evaluating “minutes” rather than “participants”.

### Apple Watch (`data_for_weka_aw.csv`)

- 6-class **random split** (ExtraTrees): **81.83%** accuracy
- 6-class **group split** (ExtraTrees): **52.54%** accuracy
- 4-class collapsed (`Lying/Sitting/Walking/Running`) **group split** (ExtraTrees): **63.37%** accuracy

### Fitbit (`data_for_weka_fb.csv`)

- 6-class **random split** (ExtraTrees): **91.00%** accuracy
- 6-class **group split** (ExtraTrees): **81.94%** accuracy
- 4-class collapsed **group split** (ExtraTrees): **83.05%** accuracy

### Combined (`aw_fb_data.csv`, device dropped from features)

- 6-class **random split** (ExtraTrees): **86.19%** accuracy
- 6-class **group split** (RandomForest): **65.21%** accuracy
- 4-class collapsed **group split** (ExtraTrees): **72.53%** accuracy

### Cross-Device Generalization (Combined CSV)

Training on one device and testing on the other is near chance:

- 6-class, Apple Watch → Fitbit (ExtraTrees): **18.67%** accuracy (chance ≈ 16.7%)
- 6-class, Fitbit → Apple Watch (ExtraTrees): **22.89%** accuracy

The 4-class setting can show ~47% accuracy largely by over-predicting `Running` (majority class), with low macro-F1. This is a symptom of a large device shift.

## Device Shift (Combined CSV)

Using the combined file, several features differ strongly by device (means are not comparable):

- `calories`: Apple Watch mean ≈ 5.8 vs Fitbit mean ≈ 38.7
- `steps`: Apple Watch mean ≈ 180 vs Fitbit mean ≈ 10
- `distance`: Apple Watch mean ≈ 0.08 vs Fitbit mean ≈ 33
- `corr_heart_steps`: Apple Watch mean ≈ 0.006 vs Fitbit mean ≈ 0.727

This explains poor cross-device transfer without calibration or domain adaptation.

## What This Means for the Unity / Apple Watch Demo

For a publishable interactive demo, the most robust path is:

1. **Pick one device** (Apple Watch) and build a stable, uncertainty-aware controller (see `docs/ICSR2026_PAPER_BLUEPRINT.md`).
2. Use a **4-state MVP** first (`LIE/SIT/WALK/RUN/UNKNOWN`) and add `STANDING/STAIRS` only once you have a dataset that labels them.
3. If you want `Standing`, collect a small IMU dataset with explicit labels (or fuse additional signals like Apple’s “Stand” ring / altimeter / phone pose).


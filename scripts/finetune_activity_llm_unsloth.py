#!/usr/bin/env python3
from __future__ import annotations

"""
Optional (GPU/Colab): Fine-tune a small instruct LLM to map sensor feature summaries -> activity label.

This script is adapted from the provided Colab notebook. It is not expected to run on macOS/CPU.

Typical usage (Colab):
  1) Install `requirements-llm.txt`
  2) `python scripts/finetune_activity_llm_unsloth.py`
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an activity LLM with Unsloth (Colab/GPU).")
    parser.add_argument("--data-dir", default="data/kaggle_har", help="Directory containing train.csv/test.csv")
    parser.add_argument("--max-train-rows", type=int, default=1000)
    parser.add_argument("--max-val-rows", type=int, default=200)
    parser.add_argument("--output-dir", default="models/har_llm_quick_final")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _maybe_download_kaggle_dataset(data_dir: Path) -> None:
    """Download the Kaggle HAR dataset into `data_dir` if missing."""

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
        src_train = Path(path) / "train.csv"
        src_test = Path(path) / "test.csv"
        if not src_train.exists() or not src_test.exists():
            raise FileNotFoundError("kagglehub download did not include train.csv/test.csv")

        import shutil

        shutil.copy(src_train, train_csv)
        shutil.copy(src_test, test_csv)
        print(f"✅ Downloaded Kaggle HAR dataset to {data_dir}")
    except Exception as e:
        raise RuntimeError(
            "Could not download Kaggle HAR dataset. "
            f"Place train.csv/test.csv in {data_dir} manually. Original error: {e}"
        ) from e


def _preprocess_har_rows(df: pd.DataFrame) -> list[dict]:
    """Extract a few robust, interpretable features per row."""

    activity_labels = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
        "WALKING": "WALKING",
        "WALKING_UPSTAIRS": "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS": "WALKING_DOWNSTAIRS",
        "SITTING": "SITTING",
        "STANDING": "STANDING",
        "LAYING": "LAYING",
    }

    processed: list[dict] = []
    for _idx, row in df.iterrows():
        raw = row.get("Activity")
        activity = activity_labels.get(raw, str(raw).upper().strip())

        body_acc_x = float(row.get("tBodyAcc-mean()-X", 0.0))
        body_acc_y = float(row.get("tBodyAcc-mean()-Y", 0.0))
        body_acc_z = float(row.get("tBodyAcc-mean()-Z", 0.0))
        body_acc_mag = float(np.sqrt(body_acc_x**2 + body_acc_y**2 + body_acc_z**2))

        gyro_x = float(row.get("tBodyGyro-mean()-X", 0.0))
        gyro_y = float(row.get("tBodyGyro-mean()-Y", 0.0))
        gyro_z = float(row.get("tBodyGyro-mean()-Z", 0.0))
        rotation = float(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

        energy = float(row.get("tBodyAccMag-energy()", 400.0))
        grav_z = float(row.get("tGravityAcc-mean()-Z", 0.0))
        subject = int(row.get("subject", 0))

        processed.append(
            {
                "features": {
                    "body_acc_magnitude": body_acc_mag,
                    "gravity_z_mean": grav_z,
                    "rotation_intensity": rotation,
                    "body_acc_energy": energy,
                },
                "activity": activity,
                "subject": subject,
            }
        )

    return processed


def _make_prompts(processed: list[dict], *, max_rows: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for item in processed[:max_rows]:
        f = item["features"]
        activity = item["activity"]

        prompt = "Sensor readings:\n"
        prompt += f"- Acceleration magnitude: {f['body_acc_magnitude']:.3f}\n"
        prompt += f"- Gravity Z: {f['gravity_z_mean']:.3f}\n"
        prompt += f"- Rotation intensity: {f['rotation_intensity']:.4f}\n"
        prompt += f"- Energy: {f['body_acc_energy']:.1f}\n"
        prompt += "What activity is this?"

        text = f"### Instruction:\n{prompt}\n\n### Response:\n{activity}"
        examples.append({"text": text})

    return examples


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Import GPU deps lazily so the script can be inspected on CPU machines.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for Unsloth fine-tuning. Run this on Colab (T4/A100) or a GPU machine.")

    data_dir = Path(args.data_dir)
    _maybe_download_kaggle_dataset(data_dir)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_processed = _preprocess_har_rows(train_df)
    val_processed = _preprocess_har_rows(test_df)

    train_examples = _make_prompts(train_processed, max_rows=args.max_train_rows)
    val_examples = _make_prompts(val_processed, max_rows=args.max_val_rows)

    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model_name = "unsloth/Llama-3.2-1B-Instruct"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=20,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        logging_steps=25,
        report_to="none",
        fp16=True,
        optim="adamw_8bit",
        seed=args.seed,
    )

    # TRL has changed signatures across versions; try both.
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
            args=training_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
            args=training_args,
        )

    print("🎓 Training (1 epoch quick run)...")
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✅ Saved: {output_dir}")

    # Quick smoke test
    FastLanguageModel.for_inference(model)
    prompt = (
        "### Instruction:\nAcceleration: 0.53, Energy: 195.0, Rotation intensity: 0.12. Activity?\n\n### Response:\n"
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
    print("\nSample output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

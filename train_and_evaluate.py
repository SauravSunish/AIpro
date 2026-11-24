#!/usr/bin/env python3
"""
BERT SST-2 Training and Attention Extraction Pipeline
-----------------------------------------------------
End-to-end command-line script to fine-tune BERT on SST-2 and extract
multi-head self-attention weights with static visual plots.

Usage:
    # Quick smoke-test training (500 train, 100 eval samples):
    python train_and_evaluate.py --train --fast

    # Full fine-tuning:
    python train_and_evaluate.py --train --epochs 2

    # Inference and attention extraction on a sentence:
    python train_and_evaluate.py --predict "The movie was fantastic, but the acting was terrible." --save-plots
"""

import argparse
import gc
import inspect
import json
import os
import random
import sys
import numpy as np
import torch

MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
DATASET_ID = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DEFAULT_RUN_DIR = "./runs/bert-sst2-v2"


def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    model_name: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    save_dir: str = f"{DEFAULT_RUN_DIR}/model",
    checkpoint_dir: str = f"{DEFAULT_RUN_DIR}/checkpoints",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 64,
    fast_run: bool = False,
    seed: int = 42,
):
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | FAST_RUN={fast_run}")

    # 1. Load SST-2 dataset from GLUE
    print("[*] Loading SST-2 dataset...")
    raw_datasets = load_dataset(DATASET_ID, "sst2", revision=DATASET_REVISION)
    train_data = raw_datasets["train"].shuffle(seed=seed)
    eval_data = raw_datasets["validation"]

    if fast_run:
        train_data = train_data.select(range(min(500, len(train_data))))
        eval_data = eval_data.select(range(min(100, len(eval_data))))
        epochs = min(epochs, 1)

    print(f"[*] Dataset samples -> Train: {len(train_data):,}, Eval: {len(eval_data):,}")

    # 2. Tokenize with dynamic padding
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    train_dataset = train_data.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    eval_dataset = eval_data.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model setup
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        output_attentions=False,  # Keep disabled during training
    )

    # 4. Metric function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # 5. Training Arguments
    candidate_args = {
        "output_dir": checkpoint_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_steps": 10 if fast_run else 50,
        "fp16": torch.cuda.is_available(),
        "seed": seed,
        "data_seed": seed,
        "report_to": "none",
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params and "evaluation_strategy" in candidate_args:
        del candidate_args["evaluation_strategy"]
    valid_args = {k: v for k, v in candidate_args.items() if k in ta_params}
    training_args = TrainingArguments(**valid_args)

    # 6. Trainer
    trainer_candidates = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    tr_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in tr_params and "tokenizer" in trainer_candidates:
        del trainer_candidates["tokenizer"]
    valid_tr_args = {k: v for k, v in trainer_candidates.items() if k in tr_params}
    trainer = Trainer(**valid_tr_args)

    # 7. Train & Evaluate
    print(f"[*] Starting training ({epochs} epoch(s))...")
    trainer.train()

    print("[*] Evaluating on validation set...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 8. Save
    print(f"[*] Saving fine-tuned model and tokenizer to '{save_dir}'...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()
    metadata = {
        "model_id": model_name,
        "model_revision": model_revision,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": seed,
        "fast_run": fast_run,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("[*] Training pipeline completed successfully.")


def run_inference_and_attention(
    sentence: str,
    model_dir: str = f"{DEFAULT_RUN_DIR}/model",
    save_plots: bool = False,
    plots_dir: str = f"{DEFAULT_RUN_DIR}/plots",
    max_length: int = 64,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train the model first with --train.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading fine-tuned model from '{model_dir}' (device: {device})...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)

    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
    pred_id = int(probs.argmax())
    pred_label = model.config.id2label.get(pred_id, f"Class {pred_id}")

    print("\n" + "=" * 60)
    print(f"Input: \"{sentence}\"")
    print(f"Prediction: {pred_label} (Predicted probability: {probs[pred_id]:.2%})")
    print(f"Probabilities -> Negative: {probs[0]:.2%}, Positive: {probs[1]:.2%}")
    print("=" * 60)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cpu_attentions = tuple(layer.cpu() for layer in outputs.attentions)
    print(f"[*] Extracted attention across {len(cpu_attentions)} layers.")

    if save_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(plots_dir, exist_ok=True)

        # 1. 2D Heatmap for Final Layer Head 1
        matrix = cpu_attentions[-1][0, 0].numpy()
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            annot=(len(tokens) <= 15),
            fmt=".2f",
            vmin=0.0,
            vmax=1.0,
        )
        plt.title("Attention Weights: Layer 12, Head 1", fontsize=12)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        heatmap_path = os.path.join(plots_dir, "attention_heatmap_layer12_head1.png")
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        print(f"[*] Saved attention heatmap to: {heatmap_path}")

        # 2. [CLS] Attention Bar Chart
        cls_attn = cpu_attentions[-1][0, :, 0, :].mean(dim=0).numpy()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(tokens)), cls_attn, color="steelblue", edgecolor="black", alpha=0.85)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.title("Average [CLS] Attention across Heads (Layer 12)", fontsize=12)
        plt.xlabel("Tokens")
        plt.ylabel("Attention Weight")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        cls_path = os.path.join(plots_dir, "cls_attention_layer12.png")
        plt.savefig(cls_path, dpi=200)
        plt.close()
        print(f"[*] Saved [CLS] token importance chart to: {cls_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT SST-2 Fine-Tuning and Attention Visualization Pipeline")
    parser.add_argument("--train", action="store_true", help="Execute the fine-tuning training loop")
    parser.add_argument("--fast", action="store_true", help="Run in fast smoke-test mode (500 train, 100 eval samples)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--predict", type=str, default=None, help="Sentence to classify and extract attention from")
    parser.add_argument("--save-plots", action="store_true", help="Save static attention plots as PNG files")
    parser.add_argument("--model-dir", type=str, default=f"{DEFAULT_RUN_DIR}/model", help="Path to saved model directory")
    parser.add_argument("--checkpoint-dir", type=str, default=f"{DEFAULT_RUN_DIR}/checkpoints", help="Path for training checkpoints")
    parser.add_argument("--plots-dir", type=str, default=f"{DEFAULT_RUN_DIR}/plots", help="Path for generated plots")

    args = parser.parse_args()

    set_deterministic_seed(42)

    if args.train:
        run_training(
            save_dir=args.model_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            fast_run=args.fast,
        )

    predict_text = args.predict
    if predict_text is None and not args.train:
        # Default test sentence if no flag provided
        predict_text = "The movie was absolutely fantastic, but the acting was terrible."

    if predict_text is not None:
        run_inference_and_attention(
            sentence=predict_text,
            model_dir=args.model_dir,
            save_plots=args.save_plots,
            plots_dir=args.plots_dir,
        )


if __name__ == "__main__":
    main()


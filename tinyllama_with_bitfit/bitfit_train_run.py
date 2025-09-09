#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import torch
import numpy as np
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Optional NVML single snapshot (same as your LoRA script)
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo,
    )
    NVML_OK = True
except Exception:
    NVML_OK = False

print("🚀 TinyLlama SST-2 BitFit Fine-tuning")
print("=" * 40)

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR   = "./tinyllama-sst2-bitfit"
BATCH_SIZE   = 16
LEARNING_RATE= 3e-4
EPOCHS       = 3
MAX_LENGTH   = 128

print("📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Max length: {MAX_LENGTH}")

# ---------------------------
# Dataset
# ---------------------------
print("\n📁 Loading GLUE SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
print(f"   Train samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['validation'])}")

# ---------------------------
# Tokenizer
# ---------------------------
print("\n🤖 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ---------------------------
# Model
# ---------------------------
print("\n🧠 Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto",
)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

try:
    total_params = model.num_parameters()
except Exception:
    total_params = sum(p.numel() for p in model.parameters())
print("   ✓ Base model loaded")
print(f"   📊 Total parameters: {total_params:,}")

# ---------------------------
# BitFit: freeze all; unfreeze biases + norm + classifier head
# ---------------------------
print("\n🪛 Applying BitFit (bias + norm + classifier head trainable)...")

for p in model.parameters():
    p.requires_grad = False

# Unfreeze classifier head (usually named 'score.*')
for name, p in model.named_parameters():
    if name.startswith("score.") or ".score." in name:
        p.requires_grad = True

def is_bitfit_param(n: str):
    n = n.lower()
    return (
        n.endswith(".bias")
        or ".bias" in n
        or "norm.weight" in n
        or "rmsnorm.weight" in n
        or ".ln_" in n  # some models use ln_ names
    )

for name, p in model.named_parameters():
    if is_bitfit_param(name):
        p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_percentage = 100.0 * trainable_params / total_params
print("   ✓ BitFit set")
print(f"   📊 Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")

# ---------------------------
# Tokenize (KEEP labels!)
# ---------------------------
print("\n🔤 Tokenizing dataset...")

def tokenize_function(examples):
    out = tokenizer(
        examples["sentence"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    # >>> IMPORTANT: keep labels so Trainer can compute loss
    out["labels"] = examples["label"]
    return out

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"],  # keep label/labels
    desc="Tokenizing",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# Metrics
# ---------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=preds, references=labels)

print("   ✓ Metrics and data collator ready")

# ---------------------------
# TrainingArguments / Trainer
# ---------------------------
print("\n⚙️  Building training args...")

bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="steps",      # use evaluation_strategy on older HF
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    bf16=bf16_ok,
    gradient_checkpointing=False,
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

print("   ✓ Training arguments configured")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,               # <- not processing_class
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("   ✓ Trainer created")

# ---------------------------
# Pre-training eval
# ---------------------------
print("\n📊 Pre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
zs_acc = pre_eval.get("eval_accuracy", 0.0)
print(f"   Zero-shot accuracy: {zs_acc:.4f} ({zs_acc*100:.2f}%)")

# ---------------------------
# Train
# ---------------------------
print("\n🚀 Starting BitFit fine-tuning...")
start_time = time.time()
train_result = trainer.train()
training_time = time.time() - start_time
final_train_loss = getattr(train_result, "training_loss", None)
print("   ✅ Fine-tuning completed!")
print(f"   ⏱️  Training time: {training_time/60:.1f} minutes")
if final_train_loss is not None:
    print(f"   📈 Final training loss: {final_train_loss:.4f}")

# ---------------------------
# Post-training eval
# ---------------------------
print("\n📊 Post-training evaluation...")
post_eval = trainer.evaluate()
ft_acc = post_eval.get("eval_accuracy", 0.0)
improvement = ft_acc - zs_acc
print(f"   Fine-tuned accuracy: {ft_acc:.4f} ({ft_acc*100:.2f}%)")
print(f"   📈 Improvement: +{improvement:.4f} ({improvement*100:.2f} pp)")

# ---------------------------
# Save model + tokenizer
# ---------------------------
print("\n💾 Saving fine-tuned model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# ---------------------------
# Power & VRAM snapshot (NVML)
# ---------------------------
gpu_power_w = None
gpu_vram_used_mb = None
gpu_vram_total_mb = None

if NVML_OK:
    try:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        power = nvmlDeviceGetPowerUsage(h) / 1000.0  # Watts
        mem = nvmlDeviceGetMemoryInfo(h)
        gpu_power_w = float(power)
        gpu_vram_used_mb = float(mem.used) / (1024 ** 2)
        gpu_vram_total_mb = float(mem.total) / (1024 ** 2)
    except Exception:
        pass

# ---------------------------
# JSON dump
# ---------------------------
results = {
    "method": "BitFit",
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "zero_shot_accuracy": zs_acc,
    "fine_tuned_accuracy": ft_acc,
    "improvement": improvement,
    "training_time_minutes": training_time / 60.0,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "trainable_percentage": trainable_percentage,
    "gpu_power_watts": gpu_power_w,
    "gpu_vram_used_mb": gpu_vram_used_mb,
    "gpu_vram_total_mb": gpu_vram_total_mb,
}

with open(f"{OUTPUT_DIR}/benchmark_results_bitfit.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ Benchmark results saved to {OUTPUT_DIR}/benchmark_results_bitfit.json")

# ---------------------------
# Quick sample predictions
# ---------------------------
print("\n🎯 Testing individual predictions...")

def test_prediction(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    conf = probs[0][pred].item()
    return label, conf

test_sentences = [
    "This movie is amazing!",
    "I hate this boring film.",
    "The acting was okay, nothing special.",
    "Absolutely terrible movie, waste of time.",
    "Outstanding performance and great story!",
    "Boring and predictable plot.",
]

print("\n--- Fine-tuned Model Predictions ---")
for s in test_sentences:
    label, confidence = test_prediction(s, model, tokenizer)
    print(f"'{s}'")
    print(f"→ {label} (confidence: {confidence:.3f})\n")

print("=" * 50)
print("🏁 BITFIT FINE-TUNING SUMMARY")
print("=" * 50)
print("✅ Training completed successfully!")
print("📊 Results:")
print(f"   • Zero-shot accuracy: {zs_acc*100:.2f}%")
print(f"   • Fine-tuned accuracy: {ft_acc*100:.2f}%")
print(f"   • Improvement: +{improvement*100:.2f} percentage points")
print(f"   • Training time: {training_time/60:.1f} minutes")
print(f"   • Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
print(f"💾 Model saved to: {OUTPUT_DIR}")
print("\n🎉 BitFit fine-tuning complete!")

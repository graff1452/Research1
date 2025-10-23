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
    TrainerCallback,
)

# ---- Optional NVML (NVIDIA Management Library) import ----
NVML_OK = False
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo,
    )
    NVML_OK = True
except Exception:
    try:
        from nvidia_ml_py import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetPowerUsage,
            nvmlDeviceGetMemoryInfo,
        )
        NVML_OK = True
    except Exception:
        NVML_OK = False

print("🚀 TinyLlama CoLA BitFit Fine-tuning (with checkpoint-averaged power/VRAM)")
print("=" * 60)

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR    = "./tinyllama-cola-bitfit"   # <-- changed
BATCH_SIZE    = 32
LEARNING_RATE = 1e-5
EPOCHS        = 3
MAX_LENGTH    = 128
GPU_INDEX     = 0

print("📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Max length: {MAX_LENGTH}")
print(f"   NVML available: {NVML_OK}")

# ---------------------------
# Dataset (GLUE: CoLA)
# ---------------------------
print("\n📁 Loading GLUE CoLA dataset...")
dataset = load_dataset("glue", "cola")      # <-- changed
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
    id2label={0: "UNACCEPTABLE", 1: "ACCEPTABLE"},   # <-- CoLA labels
    label2id={"UNACCEPTABLE": 0, "ACCEPTABLE": 1},
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

# Unfreeze classifier head
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
        or ".ln_" in n
    )

for name, p in model.named_parameters():
    if is_bitfit_param(name):
        p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_percentage = 100.0 * trainable_params / total_params
print("   ✓ BitFit set")
print(f"   📊 Trainable parameters: {trainable_params:,} ({trainable_percentage:.4f}%)")

# ---------------------------
# Tokenize (keep labels)
# ---------------------------
print("\n🔤 Tokenizing dataset...")

def tokenize_function(examples):
    out = tokenizer(
        examples["sentence"],      # CoLA uses "sentence"
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    out["labels"] = examples["label"]
    return out

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"],  # CoLA columns
    desc="Tokenizing",
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# Metrics (CoLA uses MCC)
# ---------------------------
mcc_metric = evaluate.load("matthews_correlation")   # <-- changed

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    out = mcc_metric.compute(predictions=preds, references=labels)
    # HF Trainer expects a flat dict; key becomes "eval_matthews_correlation"
    return out

print("   ✓ Metrics and data collator ready")

# ---------------------------
# NVML checkpoint sampler callback (unchanged)
# ---------------------------
class CheckpointNVMLCallback(TrainerCallback):
    def __init__(self, track_torch_peaks: bool = True, gpu_index: int = 0):
        self.track_torch_peaks = track_torch_peaks
        self.gpu_index = gpu_index
        self.nvml_ok = False
        self.samples_power_w = []
        self.samples_vram_used_mb = []
        self.peak_allocated_mb_between_ckpts = []
        self.timeseries = []
        self.summary = {}

    def _nvml_read(self):
        h = nvmlDeviceGetHandleByIndex(self.gpu_index)
        power_w = nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(h)
        used_mb = float(mem.used) / (1024 ** 2)
        total_mb = float(mem.total) / (1024 ** 2)
        return power_w, used_mb, total_mb

    def on_train_begin(self, args, state, control, **kwargs):
        self.nvml_ok = False
        if NVML_OK:
            try:
                nvmlInit()
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False
        if self.track_torch_peaks and torch.cuda.is_available():
            try: torch.cuda.reset_peak_memory_stats()
            except Exception: pass

    def on_save(self, args, state, control, **kwargs):
        if self.nvml_ok:
            try:
                power_w, used_mb, total_mb = self._nvml_read()
                self.samples_power_w.append(float(power_w))
                self.samples_vram_used_mb.append(float(used_mb))
                self.timeseries.append({
                    "global_step": int(state.global_step),
                    "power_watts": float(power_w),
                    "vram_used_mb": float(used_mb),
                    "vram_total_mb": float(total_mb),
                })
            except Exception:
                pass
        if self.track_torch_peaks and torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                peak_bytes = torch.cuda.max_memory_allocated(dev)
                self.peak_allocated_mb_between_ckpts.append(peak_bytes / (1024 ** 2))
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        import os, statistics
        self.summary = {
            "avg_power_watts_over_checkpoints": (
                float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None
            ),
            "avg_vram_used_mb_over_checkpoints": (
                float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None
            ),
            "num_checkpoints_sampled": len(self.samples_power_w),
        }
        if self.peak_allocated_mb_between_ckpts:
            self.summary["avg_peak_allocator_mb_between_checkpoints"] = float(
                statistics.mean(self.peak_allocated_mb_between_ckpts)
            )
            self.summary["max_peak_allocator_mb_between_checkpoints"] = float(
                max(self.peak_allocated_mb_between_ckpts)
            )
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/power_vram_timeseries.json", "w") as f:
                json.dump({"samples": self.timeseries, "summary": self.summary}, f, indent=2)
        except Exception:
            pass

# ---------------------------
# TrainingArguments / Trainer
# ---------------------------
print("\n⚙️  Building training args...")
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

from packaging import version
import transformers

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_matthews_correlation",   # <-- changed
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    bf16=bf16_ok,
    gradient_checkpointing=False,
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)
# if version.parse(transformers.__version__) >= version.parse("4.18.0"):
#     args_kwargs["evaluation_strategy"] = "steps"
# else:
args_kwargs["eval_strategy"] = "steps"

training_args = TrainingArguments(**args_kwargs)
print("   ✓ Training arguments configured")

nvml_cb = CheckpointNVMLCallback(track_torch_peaks=True, gpu_index=GPU_INDEX)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[nvml_cb],
)

print("   ✓ Trainer created")

# ---------------------------
# Pre-training eval
# ---------------------------
print("\n📊 Pre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
zs_mcc = pre_eval.get("eval_matthews_correlation", 0.0)   # <-- changed
print(f"   Zero-shot MCC: {zs_mcc:.4f}")

# ---------------------------
# Train
# ---------------------------
print("\n🚀 Starting BitFit fine-tuning (CoLA)...")
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
ft_mcc = post_eval.get("eval_matthews_correlation", 0.0)  # <-- changed
improvement = ft_mcc - zs_mcc
print(f"   Fine-tuned MCC: {ft_mcc:.4f}")
print(f"   📈 Improvement: +{improvement:.4f}")

# ---------------------------
# Aggregate NVML stats
# ---------------------------
avg_power_w = None
avg_vram_used_mb = None
avg_peak_alloc_mb = None
num_ckpt_samples = 0

if hasattr(nvml_cb, "summary"):
    s = nvml_cb.summary or {}
    avg_power_w = s.get("avg_power_watts_over_checkpoints")
    avg_vram_used_mb = s.get("avg_vram_used_mb_over_checkpoints")
    avg_peak_alloc_mb = s.get("avg_peak_allocator_mb_between_checkpoints")
    num_ckpt_samples = s.get("num_checkpoints_sampled", 0)

energy_Wh = None
if avg_power_w is not None:
    energy_Wh = avg_power_w * (training_time / 3600.0)

# ---------------------------
# Save model + tokenizer
# ---------------------------
print("\n💾 Saving fine-tuned model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# ---------------------------
# JSON dump
# ---------------------------
results = {
    "method": "BitFit",
    "task": "GLUE-CoLA",
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "zero_shot_mcc": zs_mcc,             # <-- changed field names
    "fine_tuned_mcc": ft_mcc,
    "improvement_mcc": improvement,
    "training_time_minutes": training_time / 60.0,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "trainable_percentage": trainable_percentage,
    "avg_gpu_power_watts_over_checkpoints": avg_power_w,
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_used_mb,
    "avg_peak_allocator_mb_between_checkpoints": avg_peak_alloc_mb,
    "num_checkpoint_samples": num_ckpt_samples,
    "estimated_energy_Wh": energy_Wh,
}

with open(f"{OUTPUT_DIR}/benchmark_results_bitfit.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ Benchmark results saved to {OUTPUT_DIR}/benchmark_results_bitfit.json")
print(f"   ✓ Per-checkpoint timeseries saved to {OUTPUT_DIR}/power_vram_timeseries.json")

# ---------------------------
# Quick sample predictions (CoLA-style grammaticality)
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
    label = "ACCEPTABLE" if pred == 1 else "UNACCEPTABLE"
    conf = probs[0][pred].item()
    return label, conf

test_sentences = [
    "The book seems to have been read.",
    "The child laughed happily.",
    "The ideas sleep furiously.",
    "Colorless green ideas sleep quickly.",
    "Mary thinks that John left.",
    "He is difficult to please.",
]

print("\n--- Fine-tuned Model Predictions ---")
for s in test_sentences:
    label, confidence = test_prediction(s, model, tokenizer)
    print(f"'{s}'")
    print(f"→ {label} (confidence: {confidence:.3f})\n")

print("=" * 60)
print("🏁 BITFIT FINE-TUNING SUMMARY (CoLA)")
print("=" * 60)
print("✅ Training completed successfully!")
print("📊 Results:")
print(f"   • Zero-shot MCC: {zs_mcc:.4f}")
print(f"   • Fine-tuned MCC: {ft_mcc:.4f}")
print(f"   • Improvement: +{improvement:.4f}")
print(f"   • Training time: {training_time/60:.1f} minutes")
if avg_power_w is not None:
    print(f"   • Avg power over checkpoints: {avg_power_w:.2f} W (samples: {num_ckpt_samples})")
if avg_vram_used_mb is not None:
    print(f"   • Avg VRAM used over checkpoints: {avg_vram_used_mb:.0f} MiB")
if avg_peak_alloc_mb is not None:
    print(f"   • Avg peak allocator (interval): {avg_peak_alloc_mb:.0f} MiB")
if energy_Wh is not None:
    print(f"   • Estimated energy: {energy_Wh:.2f} Wh")
print(f"   • Trainable parameters: {trainable_params:,} ({trainable_percentage:.4f}%)")
print(f"💾 Model saved to: {OUTPUT_DIR}")
print("\n🎉 BitFit fine-tuning complete!")
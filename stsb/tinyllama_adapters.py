#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import torch
import evaluate
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)

from peft import PrefixTuningConfig, get_peft_model, TaskType

# ----------------------------
# Optional NVML (power/VRAM)
# ----------------------------
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

print("=" * 90)
print("RUN: task=stsb model_backend=tinyllama method=adapters_equiv_prefix_tuning")
print("=" * 90)

# ----------------------------
# Config
# ----------------------------
TASK_NAME = "stsb"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./stsb-tinyllama-adapters-prefix"

MAX_LENGTH = 128
EPOCHS = 5

# IMPORTANT: bs=32 is very aggressive for TinyLlama on 16GB. Start smaller.
BATCH_SIZE = 32
GRAD_ACCUM = 8

LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

LOG_STEPS = 50
GPU_INDEX = 0

# Prefix-tuning params (adapter-equivalent)
NUM_VIRTUAL_TOKENS = 20

# Disk safety
SAVE_STRATEGY = "no"
LOAD_BEST = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
print(f"Epochs={EPOCHS} bs={BATCH_SIZE} grad_accum={GRAD_ACCUM} lr={LEARNING_RATE} max_len={MAX_LENGTH}")
print(f"Prefix-tuning: num_virtual_tokens={NUM_VIRTUAL_TOKENS}")
print(f"NVML available: {NVML_OK}")
print(f"save_strategy={SAVE_STRATEGY} load_best_model_at_end={LOAD_BEST}")

# ----------------------------
# Dataset: GLUE STS-B
# ----------------------------
print("\nLoading GLUE STS-B...")
raw = load_dataset("glue", "stsb")
print(f"Train={len(raw['train'])}  Val={len(raw['validation'])}")

# ----------------------------
# Tokenizer
# ----------------------------
print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# LLaMA-family often has no pad token: must define for batching
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def tokenize(batch):
    enc = tok(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = batch["label"]  # float scores in [0,5]
    return enc

print("Tokenizing...")
remove_cols = [c for c in raw["train"].column_names if c not in ["sentence1", "sentence2", "label"]]
ds = raw.map(tokenize, batched=True, remove_columns=remove_cols)
data_collator = DataCollatorWithPadding(tokenizer=tok)

# ----------------------------
# Metrics: Pearson + Spearman
# ----------------------------
metric = evaluate.load("glue", "stsb")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.squeeze(np.array(preds))
    labels = np.squeeze(np.array(labels))
    return metric.compute(predictions=preds, references=labels)

# ----------------------------
# Model: regression head
# ----------------------------
print("\nLoading model...")
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
dtype = torch.bfloat16 if bf16_ok else (torch.float16 if torch.cuda.is_available() else None)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)

# CRITICAL: set pad token id on MODEL CONFIG (this is what fixes your previous crash)
model.config.pad_token_id = tok.pad_token_id
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# Some builds consult generation_config too
if hasattr(model, "generation_config") and model.generation_config is not None:
    model.generation_config.pad_token_id = tok.pad_token_id

# ----------------------------
# PEFT Prefix Tuning (adapter-equivalent)
# ----------------------------
print("\nApplying PEFT Prefix Tuning (adapter-equivalent)...")
peft_cfg = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,  # works for seq-regression head too in practice
    inference_mode=False,
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,
)
model = get_peft_model(model, peft_cfg)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_pct = 100.0 * trainable_params / max(1, total_params)
print(f"Params: trainable={trainable_params:,} total={total_params:,} ({trainable_pct:.4f}%)")

# ----------------------------
# NVML + per-epoch JSON callback
# ----------------------------
class EpochJSONAndNVMLCallback(TrainerCallback):
    """
    Writes one JSON per epoch evaluation.
    Samples NVML at each evaluation so it works with save_strategy='no'.
    """
    def __init__(self, out_dir: str, gpu_index: int = 0):
        self.out_dir = out_dir
        self.gpu_index = gpu_index
        self.nvml_ok = False
        self.train_t0 = None

        self.samples_power_w = []
        self.samples_vram_used_mb = []
        self.timeseries = []
        self.summary = {}

        self.zero_shot = None
        self.last_train_loss = None

    def _nvml_read(self):
        h = nvmlDeviceGetHandleByIndex(self.gpu_index)
        power_w = nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(h)
        used_mb = float(mem.used) / (1024 ** 2)
        total_mb = float(mem.total) / (1024 ** 2)
        return float(power_w), float(used_mb), float(total_mb)

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_t0 = time.time()
        self.nvml_ok = False
        if NVML_OK:
            try:
                nvmlInit()
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                self.last_train_loss = float(logs["loss"])
            except Exception:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = float(state.epoch) if state.epoch is not None else 0.0

        power_w = used_mb = total_mb = None
        if self.nvml_ok:
            try:
                power_w, used_mb, total_mb = self._nvml_read()
                self.samples_power_w.append(power_w)
                self.samples_vram_used_mb.append(used_mb)
                self.timeseries.append({
                    "when": "on_evaluate",
                    "epoch": epoch,
                    "global_step": int(state.global_step),
                    "power_watts": power_w,
                    "vram_used_mb": used_mb,
                    "vram_total_mb": total_mb,
                })
            except Exception:
                pass

        # persist timeseries
        try:
            with open(os.path.join(self.out_dir, "power_vram_timeseries.json"), "w") as f:
                json.dump({"samples": self.timeseries}, f, indent=2)
        except Exception:
            pass

        if metrics is None:
            metrics = {}

        elapsed_sec = (time.time() - self.train_t0) if self.train_t0 is not None else None
        pearson = metrics.get("eval_pearson", metrics.get("pearson", None))
        spearmanr = metrics.get("eval_spearmanr", metrics.get("spearmanr", None))

        out = {
            "task": "stsb",
            "method": "adapters_equiv_prefix_tuning",
            "model_backend": "tinyllama",
            "model_name": MODEL_NAME,

            "max_length": MAX_LENGTH,
            "epochs_total": int(EPOCHS),
            "epoch": epoch,

            "batch_size": int(BATCH_SIZE),
            "grad_accum": int(GRAD_ACCUM),
            "learning_rate": float(LEARNING_RATE),
            "warmup_ratio": float(WARMUP_RATIO),
            "weight_decay": float(WEIGHT_DECAY),

            "num_virtual_tokens": int(NUM_VIRTUAL_TOKENS),

            "zero_shot_pearson": None if self.zero_shot is None else float(self.zero_shot["pearson"]),
            "zero_shot_spearmanr": None if self.zero_shot is None else float(self.zero_shot["spearmanr"]),
            "eval_pearson": None if pearson is None else float(pearson),
            "eval_spearmanr": None if spearmanr is None else float(spearmanr),

            "elapsed_train_time_minutes": None if elapsed_sec is None else float(elapsed_sec / 60.0),
            "last_logged_train_loss": None if self.last_train_loss is None else float(self.last_train_loss),

            "trainable_parameters": int(trainable_params),
            "total_parameters": int(total_params),
            "trainable_percentage": float(trainable_pct),

            "nvml_power_watts_sample": power_w,
            "nvml_vram_used_mb_sample": used_mb,
        }

        # 1 file per epoch (1..EPOCHS); epoch can be float so round safely
        epoch_int = max(0, int(round(epoch)))
        fname = f"benchmark_results_epoch_{epoch_int:02d}.json"
        with open(os.path.join(self.out_dir, fname), "w") as f:
            json.dump(out, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        import statistics
        self.summary = {
            "avg_power_watts_over_evals": float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None,
            "avg_vram_used_mb_over_evals": float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None,
            "num_eval_samples": int(len(self.samples_power_w)),
        }
        try:
            path = os.path.join(self.out_dir, "power_vram_timeseries.json")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"samples": []}
            data["summary"] = self.summary
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

epoch_cb = EpochJSONAndNVMLCallback(out_dir=OUTPUT_DIR, gpu_index=GPU_INDEX)

# ----------------------------
# Older-transformers compatible Trainer: force labels into compute_loss
# ----------------------------
class PeftRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Make sure labels are passed, even if Trainer can't infer label names for PEFT.
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None

        if loss is None and labels is not None:
            # Fallback MSE
            logits = outputs.logits
            logits = logits.view(-1)
            labels = labels.view(-1).to(logits.dtype)
            loss = torch.nn.functional.mse_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ----------------------------
# TrainingArguments
# ----------------------------
print("\nBuilding TrainingArguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    logging_steps=LOG_STEPS,
    logging_strategy="steps",

    eval_strategy="epoch",
    save_strategy=SAVE_STRATEGY,
    load_best_model_at_end=LOAD_BEST,

    report_to=None,
    dataloader_drop_last=False,

    bf16=bool(bf16_ok),
    fp16=bool(torch.cuda.is_available() and not bf16_ok),

    gradient_checkpointing=True,
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

trainer = PeftRegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,  # FutureWarning only
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[epoch_cb],
)

# ----------------------------
# Zero-shot eval
# ----------------------------
print("\nPre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
zs_pearson = float(pre_eval.get("eval_pearson", pre_eval.get("pearson", 0.0)))
zs_spearmanr = float(pre_eval.get("eval_spearmanr", pre_eval.get("spearmanr", 0.0)))
epoch_cb.zero_shot = {"pearson": zs_pearson, "spearmanr": zs_spearmanr}
print(f"Zero-shot: pearson={zs_pearson:.4f} spearmanr={zs_spearmanr:.4f}")

# ----------------------------
# Train (per-epoch JSON saved automatically)
# ----------------------------
print("\nTraining...")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
final_train_loss = getattr(train_result, "training_loss", None)
print(f"Training time: {train_secs/60.0:.2f} min")
if final_train_loss is not None:
    print(f"Final training loss: {final_train_loss:.6f}")

# ----------------------------
# Final eval
# ----------------------------
print("\nPost-training evaluation...")
post_eval = trainer.evaluate()
ft_pearson = float(post_eval.get("eval_pearson", post_eval.get("pearson", 0.0)))
ft_spearmanr = float(post_eval.get("eval_spearmanr", post_eval.get("spearmanr", 0.0)))
print(f"Fine-tuned: pearson={ft_pearson:.4f} spearmanr={ft_spearmanr:.4f}")

# ----------------------------
# NVML summary + Energy (Wh)
# ----------------------------
avg_power_w = None
avg_vram_used_mb = None
num_samples = 0
if hasattr(epoch_cb, "summary") and isinstance(epoch_cb.summary, dict):
    avg_power_w = epoch_cb.summary.get("avg_power_watts_over_evals")
    avg_vram_used_mb = epoch_cb.summary.get("avg_vram_used_mb_over_evals")
    num_samples = int(epoch_cb.summary.get("num_eval_samples", 0))

energy_Wh = None
if avg_power_w is not None:
    energy_Wh = float(avg_power_w) * (train_secs / 3600.0)

# ----------------------------
# Save adapter + tokenizer
# ----------------------------
print("\nSaving PEFT adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)

# ----------------------------
# Final benchmark JSON
# ----------------------------
final_results = {
    "task": "stsb",
    "method": "adapters_equiv_prefix_tuning",
    "model_backend": "tinyllama",
    "model_name": MODEL_NAME,

    "max_length": MAX_LENGTH,
    "epochs": int(EPOCHS),
    "batch_size": int(BATCH_SIZE),
    "grad_accum": int(GRAD_ACCUM),
    "learning_rate": float(LEARNING_RATE),
    "warmup_ratio": float(WARMUP_RATIO),
    "weight_decay": float(WEIGHT_DECAY),
    "num_virtual_tokens": int(NUM_VIRTUAL_TOKENS),

    "zero_shot_pearson": zs_pearson,
    "zero_shot_spearmanr": zs_spearmanr,
    "fine_tuned_pearson": ft_pearson,
    "fine_tuned_spearmanr": ft_spearmanr,

    "training_time_minutes": float(train_secs / 60.0),
    "final_training_loss": float(final_train_loss) if final_train_loss is not None else None,

    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": float(trainable_pct),

    "avg_gpu_power_watts_over_epochs": avg_power_w,
    "avg_gpu_vram_used_mb_over_epochs": avg_vram_used_mb,
    "num_epoch_samples": int(num_samples),

    "estimated_energy_Wh": energy_Wh,

    "save_strategy": SAVE_STRATEGY,
    "load_best_model_at_end": LOAD_BEST,
}

with open(os.path.join(OUTPUT_DIR, "benchmark_results_final.json"), "w") as f:
    json.dump(final_results, f, indent=2)

print("\nDONE.")
print(f"Per-epoch JSONs: {OUTPUT_DIR}/benchmark_results_epoch_XX.json")
print(f"Final JSON:      {OUTPUT_DIR}/benchmark_results_final.json")
print(f"Timeseries:      {OUTPUT_DIR}/power_vram_timeseries.json")

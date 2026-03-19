#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, json, numpy as np, torch, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, TrainerCallback
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer

# ---------- Optional NVML (power/VRAM) ----------
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

print("=" * 80)
print("RUN: task=stsb model_backend=bert method=adapters")
print("=" * 80)

# ---- Config ----
MODEL_NAME   = "bert-base-uncased"
TASK_NAME    = "stsb"
OUTPUT_DIR   = "./stsb-bert-adapters"
ADAPTER_KIND = "double_seq_bn"     # "double_seq_bn" or "seq_bn"
REDUCTION    = 16
NONLIN       = "relu"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
GRAD_ACCUM   = 2
EPOCHS       = 5
LR           = 1e-5
WARMUP       = 0.1
LOG_STEPS    = 50
GPU_INDEX    = 0

print(f"Model: {MODEL_NAME}")
print(f"Adapter: {ADAPTER_KIND} reduction={REDUCTION} nonlin={NONLIN}")
print(f"Output: {OUTPUT_DIR}")
print(f"NVML available: {NVML_OK}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data ----
raw = load_dataset("glue", "stsb")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.padding_side = "right"

def tokenize(batch):
    enc = tok(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = np.array(batch["label"], dtype=np.float32)
    return enc

ds = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tok)

metric = evaluate.load("glue", "stsb")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.array(preds).squeeze()
    labels = np.array(labels).squeeze()
    return metric.compute(predictions=preds, references=labels)

# ---- NVML sampler on LOG steps (works with save_strategy="no") ----
class LogStepNVMLCallback(TrainerCallback):
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.nvml_ok = False
        self.samples_power_w = []
        self.samples_vram_used_mb = []
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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.nvml_ok:
            return
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
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "power_vram_timeseries.json"), "w") as f:
                json.dump({"samples": self.timeseries}, f, indent=2)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        import statistics
        self.summary = {
            "avg_gpu_power_watts_over_logs": (
                float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None
            ),
            "avg_gpu_vram_used_mb_over_logs": (
                float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None
            ),
            "num_log_samples": len(self.samples_power_w),
        }
        try:
            path = os.path.join(args.output_dir, "power_vram_timeseries.json")
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

nvml_cb = LogStepNVMLCallback(gpu_index=GPU_INDEX)

# ---- Model + Adapters ----
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
model = AutoAdapterModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if bf16_ok else None,
    device_map="auto" if torch.cuda.is_available() else None,
)

# IMPORTANT: regression head via classification head num_labels=1
model.add_classification_head(
    TASK_NAME,
    num_labels=1,
    id2label={0: "SCORE"},   # keep if your version accepts it; otherwise remove too
)

model.config.id2label = {0: "SCORE"}
model.config.label2id = {"SCORE": 0}
model.config.problem_type = "regression"
model.config.num_labels = 1

adapter_cfg = AdapterConfig.load(
    ADAPTER_KIND,
    reduction_factor=REDUCTION,
    non_linearity=NONLIN,
)
model.add_adapter(TASK_NAME, config=adapter_cfg)
model.set_active_adapters(TASK_NAME)
model.train_adapter(TASK_NAME)  # freeze base, train adapter + head

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
pct = 100.0 * trainable_params / total_params
print(f"Trainable params: {trainable_params:,} / {total_params:,} ({pct:.4f}%)")

# ---- TrainingArguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=WARMUP,
    logging_steps=LOG_STEPS,
    eval_strategy="epoch",     # STS-B: epoch eval is typically enough
    save_strategy="no",              # no checkpoints => no disk bloat
    load_best_model_at_end=False,
    report_to=None,
    bf16=bf16_ok,
    fp16=bool(not bf16_ok),
    remove_unused_columns=True,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[nvml_cb],
)

# ---- Pre-eval ----
pre = trainer.evaluate()
zs_pearson = float(pre.get("eval_pearson", 0.0))
zs_spear   = float(pre.get("eval_spearmanr", 0.0))
print(f"Zero-shot: pearson={zs_pearson:.4f} spearmanr={zs_spear:.4f}")

# ---- Train ----
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
final_train_loss = getattr(train_result, "training_loss", None)

# ---- Post-eval ----
post = trainer.evaluate()
ft_pearson = float(post.get("eval_pearson", 0.0))
ft_spear   = float(post.get("eval_spearmanr", 0.0))
print(f"Fine-tuned: pearson={ft_pearson:.4f} spearmanr={ft_spear:.4f}")

# ---- NVML aggregates ----
avg_power_w = nvml_cb.summary.get("avg_gpu_power_watts_over_logs")
avg_vram_mb = nvml_cb.summary.get("avg_gpu_vram_used_mb_over_logs")
num_samples = nvml_cb.summary.get("num_log_samples", 0)

energy_Wh = avg_power_w * (train_secs / 3600.0) if avg_power_w is not None else None

# ---- Save adapter + head ----
model.save_adapter(OUTPUT_DIR, TASK_NAME, with_head=True)
tok.save_pretrained(OUTPUT_DIR)

# ---- Results JSON ----
results = {
    "task": "stsb",
    "method": "adapters",
    "model_backend": "bert",
    "model_name": MODEL_NAME,
    "adapter_kind": ADAPTER_KIND,
    "reduction_factor": REDUCTION,
    "non_linearity": NONLIN,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LR,
    "warmup_ratio": WARMUP,

    "zero_shot_pearson": zs_pearson,
    "zero_shot_spearmanr": zs_spear,
    "fine_tuned_pearson": ft_pearson,
    "fine_tuned_spearmanr": ft_spear,

    "training_time_minutes": train_secs / 60.0,
    "final_training_loss": float(final_train_loss) if final_train_loss is not None else None,

    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": float(pct),

    # NVML aggregated across LOG samples (works without checkpoints)
    "avg_gpu_power_watts_over_checkpoints": avg_power_w,     # keep your key name stable
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_mb,    # keep your key name stable
    "num_checkpoint_samples": int(num_samples),
    "estimated_energy_Wh": energy_Wh,
}

with open(os.path.join(OUTPUT_DIR, "benchmark_results_adapters.json"), "w") as f:
    json.dump(results, f, indent=2)

print("✓ Saved:", os.path.join(OUTPUT_DIR, "benchmark_results_adapters.json"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, json, numpy as np, torch, evaluate
from datasets import load_dataset
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

print("🚀 BERT QNLI Full Fine-tuning (with checkpoint-averaged power/VRAM)")
print("=" * 60)

# ------------ Config ------------
MODEL_NAME   = "bert-base-uncased"
OUTPUT_DIR   = "./bert-qnli-fullft"
MAX_LENGTH   = 128
EPOCHS       = 5
BATCH_SIZE   = 32
GRAD_ACCUM   = 2
LEARNING_RATE= 1e-5
WARMUP_RATIO = 0.1
LOG_STEPS    = 50
EVAL_STEPS   = 200
SAVE_STEPS   = 200
GPU_INDEX    = 0
OPTIM        = "adamw_torch"  # recommended for BERT; keep as string for TrainingArguments.optim

print("📋 Configuration")
print(f" • Model: {MODEL_NAME}")
print(f" • Task: GLUE/QNLI")
print(f" • Max length: {MAX_LENGTH}")
print(f" • Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
print(f" • Micro-batch: {BATCH_SIZE} × grad_accum {GRAD_ACCUM} (effective {BATCH_SIZE*GRAD_ACCUM})")
print(f" • Optimizer: {OPTIM}")
print(f" • NVML available: {NVML_OK}")

# ------------ Data ------------
print("\n📁 Loading GLUE/QNLI…")
raw = load_dataset("glue", "qnli")
print(f"   Train={len(raw['train'])}, Val={len(raw['validation'])}")

# ------------ Tokenizer ------------
print("\n🔤 Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.padding_side = "right"

def tokenize(batch):
    # QNLI is a pair task: (question, sentence)
    enc = tok(
        batch["question"],
        batch["sentence"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = batch["label"]
    return enc

print("🔧 Tokenizing…")
ds = raw.map(
    tokenize,
    batched=True,
    remove_columns=["question", "sentence", "idx", "label"],
)
data_collator = DataCollatorWithPadding(tokenizer=tok)

# ------------ Metrics (QNLI uses accuracy) ------------
acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return acc.compute(predictions=preds, references=labels)

# ------------ Model ------------
print("\n🤖 Loading base model (full train)…")
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    # Keep consistent with GLUE/QNLI labels used in your script:
    # 0 = NOT_ENTAILMENT, 1 = ENTAILMENT
    id2label={0: "NOT_ENTAILMENT", 1: "ENTAILMENT"},
    label2id={"NOT_ENTAILMENT": 0, "ENTAILMENT": 1},
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   📊 Parameters: {trainable_params:,} trainable / {total_params:,} total (100.00%)")

# ------------ NVML checkpoint sampler callback ------------
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
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

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
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/power_vram_timeseries.json", "w") as f:
                    json.dump({"samples": self.timeseries}, f, indent=2)
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
        import statistics, os
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
            path = f"{args.output_dir}/power_vram_timeseries.json"
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

nvml_cb = CheckpointNVMLCallback(track_torch_peaks=True, gpu_index=GPU_INDEX)

# ------------ Training Args ------------
print("\n⚙️  Building TrainingArguments…")
from packaging import version
import transformers

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOG_STEPS,
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    dataloader_num_workers=4,
    bf16=bool(bf16_ok),
    fp16=bool(not bf16_ok),
    gradient_checkpointing=True,
    optim=OPTIM,
)

args_kwargs["eval_strategy"] = "steps"
args = TrainingArguments(**args_kwargs)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[nvml_cb],
)

# ------------ Pre-train eval ------------
print("\n📊 Pre-training evaluation (zero-shot)…")
pre_eval = trainer.evaluate()
zs_acc = float(pre_eval.get("eval_accuracy", 0.0))
print(f"   Zero-shot accuracy: {zs_acc:.4f}")

# ------------ Train ------------
print("\n🚀 Starting FULL fine-tuning (QNLI)…")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
final_train_loss = getattr(train_result, "training_loss", None)
print("   ✅ Done.")
print(f"   ⏱️  Training time: {train_secs/60:.1f} min")
if final_train_loss is not None:
    print(f"   📉 Final training loss: {final_train_loss:.4f}")

# ------------ Post eval ------------
print("\n📊 Post-training evaluation…")
post_eval = trainer.evaluate()
ft_acc = float(post_eval.get("eval_accuracy", 0.0))
impr = ft_acc - zs_acc
print(f"   Fine-tuned accuracy: {ft_acc:.4f}")
print(f"   📈 Improvement: +{impr:.4f}")

# ------------ Aggregate NVML stats ------------
avg_power_w = avg_vram_used_mb = avg_peak_alloc_mb = None
num_ckpt_samples = 0
if hasattr(nvml_cb, "summary"):
    s = nvml_cb.summary or {}
    avg_power_w = s.get("avg_power_watts_over_checkpoints")
    avg_vram_used_mb = s.get("avg_vram_used_mb_over_checkpoints")
    avg_peak_alloc_mb = s.get("avg_peak_allocator_mb_between_checkpoints")
    num_ckpt_samples = s.get("num_checkpoints_sampled", 0)

energy_Wh = None
if avg_power_w is not None:
    energy_Wh = avg_power_w * (train_secs / 3600.0)

# ------------ Save ------------
print("\n💾 Saving full fine-tuned model…")
trainer.save_model()
tok.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# ------------ JSON dump ------------
results = {
    "method": "full_finetune",
    "task": "GLUE-QNLI",
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "zero_shot_accuracy": zs_acc,
    "fine_tuned_accuracy": ft_acc,
    "improvement": impr,
    "training_time_minutes": train_secs / 60.0,
    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": 100.0,
    "avg_gpu_power_watts_over_checkpoints": avg_power_w,
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_used_mb,
    "avg_peak_allocator_mb_between_checkpoints": avg_peak_alloc_mb,
    "num_checkpoint_samples": num_ckpt_samples,
    "estimated_energy_Wh": energy_Wh,
}

with open(f"{OUTPUT_DIR}/benchmark_results_fullft.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ JSON saved to {OUTPUT_DIR}/benchmark_results_fullft.json")
print(f"   ✓ Per-checkpoint timeseries saved to {OUTPUT_DIR}/power_vram_timeseries.json")

# ------------ Quick predictions (QNLI) ------------
print("\n🎯 Sample predictions (QNLI)")
def predict(question: str, sentence: str):
    enc = tok(question, sentence, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0]
        cls = torch.argmax(probs).item()
        label = "ENTAILMENT" if cls == 1 else "NOT_ENTAILMENT"
        conf = probs[cls].item()
    return label, conf

pairs = [
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
    ("Where is Mount Everest located?", "Mount Everest is located in the Himalayas."),
    ("Is water a metal?", "Water is a chemical compound consisting of hydrogen and oxygen."),
]

for q, s in pairs:
    lab, conf = predict(q, s)
    print(f"Q: {q}\nS: {s}\n→ {lab} ({conf:.3f})\n")

print("\n🏁 FULL FINE-TUNING SUMMARY (QNLI)")
print(f" • Zero-shot accuracy: {zs_acc:.4f}")
print(f" • Fine-tuned accuracy: {ft_acc:.4f}  (Δ {impr:.4f})")
print(f" • Time: {train_secs/60:.1f} min | Trainable params: {trainable_params:,}")
if avg_power_w is not None:
    print(f" • Avg power over checkpoints: {avg_power_w:.2f} W (samples: {num_ckpt_samples})")
if avg_vram_used_mb is not None:
    print(f" • Avg VRAM used over checkpoints: {avg_vram_used_mb:.0f} MiB")
if avg_peak_alloc_mb is not None:
    print(f" • Avg peak allocator (interval): {avg_peak_alloc_mb:.0f} MiB")
if energy_Wh is not None:
    print(f" • Estimated energy: {energy_Wh:.2f} Wh")
print(f" • Saved to: {OUTPUT_DIR}")

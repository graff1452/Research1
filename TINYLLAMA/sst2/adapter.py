#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, json, numpy as np, torch, evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    TrainerCallback,
)
from adapters import (
    AutoAdapterModel,
    AdapterConfig,
    AdapterTrainer,
)

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

print("🚀 TinyLlama SST-2 Adapter Fine-tuning (with checkpoint-averaged power/VRAM)")
print("=" * 72)

# ---- Config ----
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK_NAME    = "sst2"
OUTPUT_DIR   = "./tinyllama-sst2-adapters"
ADAPTER_KIND = "double_seq_bn"     # "double_seq_bn" (Houlsby) or "seq_bn" (Pfeiffer)
REDUCTION    = 16                  # bottleneck: hidden_dim / reduction
NONLIN       = "relu"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
GRAD_ACCUM   = 2
EPOCHS       = 3
LR           = 1e-5
WARMUP       = 0.1
LOG_STEPS    = 50
EVAL_STEPS   = 200
SAVE_STEPS   = 200
GPU_INDEX    = 0

print("📋 Configuration")
print(f" • Model: {MODEL_NAME}")
print(f" • Adapter: {ADAPTER_KIND} (reduction={REDUCTION}, nonlin={NONLIN})")
print(f" • Max length: {MAX_LENGTH}")
print(f" • BS: {BATCH_SIZE} × grad_accum {GRAD_ACCUM}  |  LR: {LR}  |  Epochs: {EPOCHS}")
print(f" • NVML available: {NVML_OK}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data ----
print("\n📁 Loading GLUE/SST-2…")
raw = load_dataset("glue", "sst2")
print(f"   Train={len(raw['train'])}, Val={len(raw['validation'])}")

# ---- Tokenizer ----
print("\n🔤 Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def tokenize(batch):
    enc = tok(
        batch["sentence"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = batch["label"]
    return enc

print("🔧 Tokenizing…")
ds = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
data_collator = DataCollatorWithPadding(tokenizer=tok)
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return acc_metric.compute(predictions=preds, references=labels)

# ---- Model + Adapters ----
print("\n🤖 Loading base model with adapter support…")
bf16_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
model = AutoAdapterModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if bf16_ok else None,
    device_map="auto",
)

# Classification head + adapter (same name: TASK_NAME)
print("🧩 Adding classification head + adapter…")
model.add_classification_head(
    TASK_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
)
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

adapter_cfg = AdapterConfig.load(
    ADAPTER_KIND,
    reduction_factor=REDUCTION,
    non_linearity=NONLIN,
)
model.add_adapter(TASK_NAME, config=adapter_cfg)
model.set_active_adapters(TASK_NAME)
model.train_adapter(TASK_NAME)  # freeze base, train adapter + head

# Count params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
pct = 100.0 * trainable_params / total_params
print(f"   📊 Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

# ---- NVML checkpoint sampler ----
from transformers import TrainerCallback
class CheckpointNVMLCallback(TrainerCallback):
    """
    Sample NVML power (W) and VRAM (MiB) at each checkpoint save; write a timeseries
    and compute averages at train end.
    """
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

    def on_save(self, args, state, control, **kwargs):
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
            # persist incrementally
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "power_vram_timeseries.json"), "w") as f:
                json.dump({"samples": self.timeseries}, f, indent=2)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        import statistics
        self.summary = {
            "avg_power_watts_over_checkpoints": (
                float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None
            ),
            "avg_vram_used_mb_over_checkpoints": (
                float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None
            ),
            "num_checkpoints_sampled": len(self.samples_power_w),
        }
        # append summary to file
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

nvml_cb = CheckpointNVMLCallback(gpu_index=GPU_INDEX)

# ---- Training args (version-compatible eval key) ----
from packaging import version
import transformers

print("\n⚙️  Building TrainingArguments…")
args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=WARMUP,
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
    bf16=bool(bf16_ok),
    remove_unused_columns=True,
)

# choose right kwarg for your transformers version
# if version.parse(transformers.__version__) >= version.parse("4.18.0"):
#     args_kwargs["evaluation_strategy"] = "steps"
# else:
args_kwargs["eval_strategy"] = "steps"

training_args = TrainingArguments(**args_kwargs)

# ---- Trainer ----
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

# ---- Pre-train eval ----
print("\n📊 Pre-training evaluation (zero-shot)…")
pre_eval = trainer.evaluate()
zs_acc = float(pre_eval.get("eval_accuracy", 0.0))
print(f"   Zero-shot accuracy: {zs_acc:.4f} ({zs_acc*100:.2f}%)")

# ---- Train ----
print("\n🚀 Starting Adapter fine-tuning…")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
print("   ✅ Done.")
print(f"   ⏱️  Training time: {train_secs/60:.1f} min")
print(f"   📉 Final training loss: {getattr(train_result, 'training_loss', float('nan')):.4f}")

# ---- Post eval ----
print("\n📊 Post-training evaluation…")
post_eval = trainer.evaluate()
ft_acc = float(post_eval.get("eval_accuracy", 0.0))
impr = ft_acc - zs_acc
print(f"   Fine-tuned accuracy: {ft_acc:.4f} ({ft_acc*100:.2f}%)")
print(f"   📈 Improvement: +{impr:.4f} ({impr*100:.2f} pts)")

# ---- Aggregate NVML stats (averaged over checkpoints) ----
avg_power_w = avg_vram_used_mb = None
num_ckpt_samples = 0
if hasattr(nvml_cb, "summary"):
    s = nvml_cb.summary or {}
    avg_power_w = s.get("avg_power_watts_over_checkpoints")
    avg_vram_used_mb = s.get("avg_vram_used_mb_over_checkpoints")
    num_ckpt_samples = s.get("num_checkpoints_sampled", 0)

# Optional energy estimate (Wh)
energy_Wh = None
if avg_power_w is not None:
    energy_Wh = avg_power_w * (train_secs / 3600.0)

# ---- Save adapter + head ----
print("\n💾 Saving adapter + head…")
model.save_adapter(OUTPUT_DIR, TASK_NAME, with_head=True)
tok.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Saved to: {OUTPUT_DIR}")

# ---- JSON dump (augmented) ----
results = {
    "method": "adapters",
    "adapter_kind": ADAPTER_KIND,
    "reduction_factor": REDUCTION,
    "non_linearity": NONLIN,
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LR,
    "epochs": EPOCHS,
    "zero_shot_accuracy": zs_acc,
    "fine_tuned_accuracy": ft_acc,
    "improvement": impr,
    "training_time_minutes": train_secs / 60.0,
    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": pct,

    # Aggregated NVML metrics (averaged across checkpoint saves)
    "avg_gpu_power_watts_over_checkpoints": avg_power_w,
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_used_mb,
    "num_checkpoint_samples": num_ckpt_samples,

    # Optional derived metric
    "estimated_energy_Wh": energy_Wh,
}
with open(os.path.join(OUTPUT_DIR, "benchmark_results_adapters.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ JSON saved to {OUTPUT_DIR}/benchmark_results_adapters.json")

# ---- Quick predictions ----
print("\n🎯 Sample predictions")
def predict(text: str):
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0]
        cls = torch.argmax(probs).item()
        label = "POSITIVE" if cls == 1 else "NEGATIVE"
        conf = probs[cls].item()
    return label, conf

for s in [
    "This movie is amazing!",
    "I hate this boring film.",
    "The acting was okay, nothing special.",
]:
    lab, conf = predict(s)
    print(f"'{s}' → {lab} ({conf:.3f})")

print("\n🏁 ADAPTER FINE-TUNING SUMMARY")
print(f" • Zero-shot acc: {zs_acc*100:.2f}%")
print(f" • Fine-tuned acc: {ft_acc*100:.2f}%")
print(f" • +{impr*100:.2f} pts | {train_secs/60:.1f} min")
if avg_power_w is not None:
    print(f" • Avg power over checkpoints: {avg_power_w:.2f} W (samples: {num_ckpt_samples})")
if avg_vram_used_mb is not None:
    print(f" • Avg VRAM used over checkpoints: {avg_vram_used_mb:.0f} MiB")
if energy_Wh is not None:
    print(f" • Estimated energy: {energy_Wh:.2f} Wh")
print(f" • Trainable params: {trainable_params:,} ({pct:.2f}%)")
print(f" • Saved to: {OUTPUT_DIR}")

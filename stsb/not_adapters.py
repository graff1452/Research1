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
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ===================== REQUIRED ENV VARS =====================
# MODEL_BACKEND: "bert" | "tinyllama"
# METHOD:        "full" | "adapters" | "lora" | "loraplus" | "qlora" | "bitfit"
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "").strip().lower()
METHOD = os.environ.get("METHOD", "").strip().lower()

if MODEL_BACKEND not in {"bert", "tinyllama"}:
    raise ValueError("MODEL_BACKEND must be set to 'bert' or 'tinyllama' (environment variable).")
if METHOD not in {"full", "lora", "loraplus", "qlora", "bitfit"}:
    raise ValueError("METHOD must be one of: full, lora, loraplus, qlora, bitfit (environment variable).")

# -------------------- Models --------------------
MODEL_NAME = {
    "bert": "bert-base-uncased",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}[MODEL_BACKEND]

# -------------------- Output --------------------
OUTPUT_DIR = f"./stsb-{MODEL_BACKEND}-{METHOD}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Config --------------------
TASK = "stsb"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
LR = float(os.environ.get("LR", "1e-5"))
WARMUP = float(os.environ.get("WARMUP", "0.1"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "50"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "200"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
GPU_INDEX = int(os.environ.get("GPU_INDEX", "0"))

# Optimizer choice for full fine-tune (optional)
OPTIM = os.environ.get("OPTIM", "adamw_torch")  # e.g., "paged_adamw_8bit" if your env supports it

print("=" * 80)
print(f"RUN: task={TASK} model_backend={MODEL_BACKEND} method={METHOD}")
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
print(f"Epochs={EPOCHS} bs={BATCH_SIZE} grad_accum={GRAD_ACCUM} lr={LR} max_len={MAX_LENGTH}")
print("=" * 80)

# -------------------- Optional NVML --------------------
NVML_OK = False
try:
    from pynvml import (  # type: ignore
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo,
    )
    NVML_OK = True
except Exception:
    try:
        from nvidia_ml_py import (  # type: ignore
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetPowerUsage,
            nvmlDeviceGetMemoryInfo,
        )
        NVML_OK = True
    except Exception:
        NVML_OK = False

print(f"NVML available: {NVML_OK}")

class CheckpointNVMLCallback(TrainerCallback):
    """
    Samples NVML power (W) and VRAM used (MiB) at each checkpoint save.
    Writes power_vram_timeseries.json incrementally and appends summary at end.
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
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "power_vram_timeseries.json"), "w") as f:
                json.dump({"samples": self.timeseries}, f, indent=2)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        import statistics
        self.summary = {
            "avg_gpu_power_watts_over_checkpoints": (
                float(statistics.mean(self.samples_power_w)) if self.samples_power_w else None
            ),
            "avg_gpu_vram_used_mb_over_checkpoints": (
                float(statistics.mean(self.samples_vram_used_mb)) if self.samples_vram_used_mb else None
            ),
            "num_checkpoint_samples": len(self.samples_power_w),
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

nvml_cb = CheckpointNVMLCallback(gpu_index=GPU_INDEX)

# -------------------- Dataset: GLUE STS-B --------------------
print("Loading GLUE STS-B...")
raw = load_dataset("glue", "stsb")
print(f"Train={len(raw['train'])}  Val={len(raw['validation'])}")

# -------------------- Tokenizer --------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# TinyLlama/LLaMA tokenizers often have no PAD token; define it.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


def tokenize(batch):
    enc = tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    # STS-B labels are floats in [0, 5]
    enc["labels"] = np.array(batch["label"], dtype=np.float32)
    return enc

# Remove all original columns; keep only model inputs + labels
ds = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------- Metrics --------------------
metric = evaluate.load("glue", "stsb")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.array(logits).squeeze()
    labels = np.array(labels).squeeze()
    # evaluate expects python lists / numpy arrays; both OK
    return metric.compute(predictions=preds, references=labels)

# -------------------- Load model (regression head) --------------------
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

quant_cfg = None
if METHOD == "qlora":
    # bitsandbytes required
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        raise RuntimeError("QLoRA requires bitsandbytes installed and working.") from e

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id


if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# -------------------- Apply method --------------------
def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    return total, trainable, pct

if METHOD in {"lora", "loraplus", "qlora"}:
    if METHOD == "qlora":
        model = prepare_model_for_kbit_training(model)

    # target_modules differ across architectures
    if MODEL_BACKEND == "bert":
        # BERT attention Linear modules are typically named query/key/value
        target_modules = ["query", "key", "value"]
    else:
        # TinyLlama uses q_proj/k_proj/v_proj/o_proj
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,   # still fine; head is sequence-level regression
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

elif METHOD == "bitfit":
    # Freeze all; unfreeze biases + LayerNorm + regressor head
    for p in model.parameters():
        p.requires_grad = False

    def is_bitfit_param(name: str) -> bool:
        n = name.lower()
        return (
            n.endswith(".bias") or ".bias" in n or
            "layernorm.weight" in n or "layernorm.bias" in n
        )

    for name, p in model.named_parameters():
        # BERT head is "classifier.*"
        if name.startswith("classifier.") or ".classifier." in name:
            p.requires_grad = True
        if is_bitfit_param(name):
            p.requires_grad = True

elif METHOD == "adapters":
    # Adapters require adapter-transformers; keep this run separate.
    # We raise here so the launcher can still run other methods, but you should
    # use a dedicated adapters script (I can provide it) if you want adapters for STS-B.
    raise RuntimeError(
        "METHOD=adapters requires AutoAdapterModel/AdapterTrainer from adapter-transformers. "
        "Use the dedicated adapters STS-B script (separate file) so imports are correct."
    )

# full: no changes

total_params, trainable_params, trainable_pct = count_params(model)
print(f"Params: trainable={trainable_params:,} total={total_params:,} ({trainable_pct:.4f}%)")

# -------------------- Training args --------------------
# Note: for STS-B, HF sometimes uses eval_spearmanr as "best model" metric; set that.
# Some versions name it "eval_spearmanr". evaluate("glue","stsb") returns keys:
# "pearson" and "spearmanr". Trainer prefixes with "eval_" => eval_pearson, eval_spearmanr.
args = TrainingArguments(
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
    save_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=1,
    load_best_model_at_end=False,
    metric_for_best_model="eval_spearmanr",
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    bf16=bf16_ok,
    fp16=bool(not bf16_ok),
    gradient_checkpointing=True if METHOD in {"full", "qlora"} else False,
    optim="adafactor" if (METHOD == "full" and MODEL_BACKEND == "tinyllama") else "adamw_torch",
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[nvml_cb],
)

# -------------------- Pre-eval (zero-shot) --------------------
print("Pre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
zs_pearson = float(pre_eval.get("eval_pearson", 0.0))
zs_spearman = float(pre_eval.get("eval_spearmanr", 0.0))
print(f"Zero-shot: pearson={zs_pearson:.4f} spearmanr={zs_spearman:.4f}")

# -------------------- Train --------------------
print("Training...")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
final_train_loss = getattr(train_result, "training_loss", None)
print(f"Train time: {train_secs/60.0:.1f} min")

# -------------------- Post-eval --------------------
print("Post-training evaluation...")
post_eval = trainer.evaluate()
ft_pearson = float(post_eval.get("eval_pearson", 0.0))
ft_spearman = float(post_eval.get("eval_spearmanr", 0.0))
print(f"Fine-tuned: pearson={ft_pearson:.4f} spearmanr={ft_spearman:.4f}")

# -------------------- Aggregate NVML + energy --------------------
avg_power_w = None
avg_vram_used_mb = None
num_ckpt_samples = 0
if hasattr(nvml_cb, "summary"):
    s = nvml_cb.summary or {}
    avg_power_w = s.get("avg_gpu_power_watts_over_checkpoints")
    avg_vram_used_mb = s.get("avg_gpu_vram_used_mb_over_checkpoints")
    num_ckpt_samples = s.get("num_checkpoint_samples", 0)

energy_Wh = None
if avg_power_w is not None:
    energy_Wh = float(avg_power_w) * (train_secs / 3600.0)

# -------------------- Save model artifacts --------------------
print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

# -------------------- Results JSON --------------------
results = {
    "task": "stsb",
    "method": METHOD,
    "model_backend": MODEL_BACKEND,
    "model_name": MODEL_NAME,

    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LR,
    "warmup_ratio": WARMUP,

    "zero_shot_pearson": zs_pearson,
    "zero_shot_spearmanr": zs_spearman,
    "fine_tuned_pearson": ft_pearson,
    "fine_tuned_spearmanr": ft_spearman,

    "training_time_minutes": train_secs / 60.0,
    "final_training_loss": float(final_train_loss) if final_train_loss is not None else None,

    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": float(trainable_pct),

    "avg_gpu_power_watts_over_checkpoints": avg_power_w,
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_used_mb,
    "num_checkpoint_samples": num_ckpt_samples,

    "estimated_energy_Wh": energy_Wh,
}

with open(os.path.join(OUTPUT_DIR, f"benchmark_results_{METHOD}.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Wrote: {OUTPUT_DIR}/benchmark_results_{METHOD}.json")
print("DONE.")
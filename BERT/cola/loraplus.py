#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, json, numpy as np, torch, evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---- optional bitsandbytes (for 8-bit optimizer) ----
try:
    import bitsandbytes as bnb
    BNB_OK = True
except Exception:
    BNB_OK = False

print("🚀 BERT CoLA LoRA+ Fine-tuning (with JSON & NVML like sample)")
print("=" * 70)

# --------------------------- Config ---------------------------
MODEL_NAME     = "bert-base-uncased"
OUTPUT_DIR     = "./bert-cola-loraplus"
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-5
EPOCHS         = 5
MAX_LENGTH     = 128
LOG_STEPS      = 50
EVAL_STEPS     = 200
SAVE_STEPS     = 200
GPU_INDEX      = 0
LORAPLUS_RATIO = 16   # LR_B = ratio * LR_A

print("📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE} (LoRA+ ratio={LORAPLUS_RATIO})")
print(f"   Epochs: {EPOCHS}")

# --------------------------- NVML optional imports ---------------------------
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

print(f"   NVML available: {NVML_OK}")

# --------------------------- Dataset ---------------------------
print("\n📁 Loading GLUE CoLA dataset...")
dataset = load_dataset("glue", "cola")
print(f"   Train samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['validation'])}")

# --------------------------- Tokenizer ---------------------------
print("\n🤖 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# BERT already defines a PAD token.
tokenizer.padding_side = "right"

# --------------------------- Base model ---------------------------
print("\n🧠 Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "UNACCEPTABLE", 1: "ACCEPTABLE"},
    label2id={"UNACCEPTABLE": 0, "ACCEPTABLE": 1},
)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

try:
    base_total_params = model.num_parameters()
except Exception:
    base_total_params = sum(p.numel() for p in model.parameters())

print(f"   ✓ Base model loaded")
print(f"   📊 Base parameters: {base_total_params:,}")

# --------------------------- LoRA config ---------------------------
print("\n🔧 Configuring LoRA...")
# BERT attention projections: "...attention.self.query|key|value"
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # CHANGED for BERT
    bias="none",
)
model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
trainable_pct    = 100.0 * trainable_params / total_params
print(f"   ✓ LoRA applied")
print(f"   📊 Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
print(f"   📊 Total parameters (with LoRA): {total_params:,}")

# --------------------------- Tokenize ---------------------------
print("\n🔤 Tokenizing dataset...")
def tokenize_function(examples):
    enc = tokenizer(
        examples["sentence"],            # CoLA uses "sentence"
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = examples["label"]
    return enc

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"],
    desc="Tokenizing"
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --------------------------- Metrics (CoLA uses MCC) ---------------------------
mcc_metric = evaluate.load("matthews_correlation")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return mcc_metric.compute(predictions=preds, references=labels)

print("   ✓ Metrics and data collator ready")

# --------------------------- NVML checkpoint sampler ---------------------------
class CheckpointNVMLCallback(TrainerCallback):
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
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/power_vram_timeseries.json", "w") as f:
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

nvml_cb = CheckpointNVMLCallback(gpu_index=GPU_INDEX)

# --------------------------- TrainingArguments ---------------------------
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=LOG_STEPS,
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_matthews_correlation",
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    bf16=bool(bf16_ok),
    fp16=bool(not bf16_ok),
    gradient_checkpointing=False,
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)
args_kwargs["eval_strategy"] = "steps"
training_args = TrainingArguments(**args_kwargs)
print("\n✓ Training arguments configured")

# --------------------------- LoRA+ optimizer ---------------------------
optimizer = None
scheduler = None

def build_manual_loraplus_param_groups(model, lr, ratio):
    a_params, b_params, rest = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in n:
            a_params.append(p)
        elif "lora_B" in n:
            b_params.append(p)
        else:
            rest.append(p)
    groups = [
        {"params": a_params, "lr": lr},
        {"params": b_params, "lr": lr * ratio},
    ]
    if rest:
        groups.append({"params": rest, "lr": lr})
    return groups

try:
    from peft.optimizers import create_loraplus_optimizer
    opt_cls = bnb.optim.Adam8bit if BNB_OK else torch.optim.AdamW
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=opt_cls,
        lr=training_args.learning_rate,
        loraplus_lr_ratio=LORAPLUS_RATIO,
        weight_decay=training_args.weight_decay,
    )
except Exception:
    print("   ⚠️ Falling back to manual LoRA+ param groups.")
    param_groups = build_manual_loraplus_param_groups(model, training_args.learning_rate, LORAPLUS_RATIO)
    if BNB_OK:
        optimizer = bnb.optim.Adam8bit(param_groups, weight_decay=training_args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=training_args.weight_decay)

from transformers.optimization import get_linear_schedule_with_warmup

# Correct step-count with gradient accumulation
effective_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
updates_per_epoch = max(1, len(tokenized["train"]) // effective_bs)
num_training_steps = updates_per_epoch * int(training_args.num_train_epochs)
num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# --------------------------- Trainer ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[nvml_cb],
    optimizers=(optimizer, scheduler),
)
print("✓ Trainer created")

# --------------------------- Pre-training eval ---------------------------
print("\n📊 Pre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
zs_mcc = float(pre_eval.get("eval_matthews_correlation", 0.0))
print(f"   Zero-shot MCC: {zs_mcc:.4f}")

# --------------------------- Train ---------------------------
print("\n🚀 Starting LoRA+ fine-tuning (CoLA)...")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
final_train_loss = getattr(train_result, "training_loss", None)
print("   ✅ Fine-tuning completed!")
print(f"   ⏱️  Training time: {train_secs/60:.1f} minutes")
if final_train_loss is not None:
    print(f"   📈 Final training loss: {final_train_loss:.4f}")

# --------------------------- Post-training eval ---------------------------
print("\n📊 Post-training evaluation...")
post_eval = trainer.evaluate()
ft_mcc = float(post_eval.get("eval_matthews_correlation", 0.0))
improvement = ft_mcc - zs_mcc
print(f"   Fine-tuned MCC: {ft_mcc:.4f}")
print(f"   📈 Improvement: +{improvement:.4f}")

# --------------------------- Aggregate NVML stats ---------------------------
avg_power_w = avg_vram_used_mb = None
num_ckpt_samples = 0
if hasattr(nvml_cb, "summary"):
    s = nvml_cb.summary or {}
    avg_power_w = s.get("avg_power_watts_over_checkpoints")
    avg_vram_used_mb = s.get("avg_vram_used_mb_over_checkpoints")
    num_ckpt_samples = s.get("num_checkpoints_sampled", 0)

energy_Wh = None
if avg_power_w is not None:
    energy_Wh = avg_power_w * (train_secs / 3600.0)

# --------------------------- Save model ---------------------------
print("\n💾 Saving fine-tuned model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# --------------------------- JSON dump ---------------------------
results = {
    "method": "LoRA+",
    "task": "GLUE-CoLA",
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "zero_shot_mcc": zs_mcc,
    "fine_tuned_mcc": ft_mcc,
    "improvement_mcc": improvement,
    "training_time_minutes": train_secs / 60.0,
    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": float(trainable_pct),
    "avg_gpu_power_watts_over_checkpoints": avg_power_w,
    "avg_gpu_vram_used_mb_over_checkpoints": avg_vram_used_mb,
    "num_checkpoint_samples": num_ckpt_samples,
    "estimated_energy_Wh": energy_Wh,
    "final_training_loss": float(final_train_loss) if final_train_loss is not None else None,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/benchmark_results_loraplus.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ Benchmark results saved to {OUTPUT_DIR}/benchmark_results_loraplus.json")
print(f"   ✓ Per-checkpoint timeseries saved to {OUTPUT_DIR}/power_vram_timeseries.json")

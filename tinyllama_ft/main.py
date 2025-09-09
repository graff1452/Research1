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
)

# Optional NVML snapshot (power/VRAM)
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

print("🚀 TinyLlama SST-2 Full Fine-tuning")
print("=" * 50)

# ------------ Config ------------
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR   = "./tinyllama-sst2-fullft"
MAX_LENGTH   = 128
EPOCHS       = 5
BATCH_SIZE   = 32          # micro-batch; keep small for 16 GB
GRAD_ACCUM   = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE= 1e-5
WARMUP_RATIO = 0.1
LOG_STEPS    = 50
EVAL_STEPS   = 200
SAVE_STEPS   = 200

# 8-bit optimizer saves a ton of memory for full FT
OPTIM = "paged_adamw_8bit"  # fallback to "adamw_torch" if you don't have bitsandbytes

print("📋 Configuration")
print(f" • Model: {MODEL_NAME}")
print(f" • Max length: {MAX_LENGTH}")
print(f" • Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
print(f" • Micro-batch: {BATCH_SIZE}  × grad_accum {GRAD_ACCUM} (effective {BATCH_SIZE*GRAD_ACCUM})")
print(f" • Optimizer: {OPTIM}")

# ------------ Data ------------
print("\n📁 Loading GLUE/SST-2…")
raw = load_dataset("glue", "sst2")
print(f"   Train={len(raw['train'])}, Val={len(raw['validation'])}")

# ------------ Tokenizer ------------
print("\n🔤 Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def tokenize(batch):
    enc = tok(batch["sentence"], truncation=True, padding=False, max_length=MAX_LENGTH)
    enc["labels"] = batch["label"]  # Trainer expects 'labels'
    return enc

print("🔧 Tokenizing…")
ds = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tok)

# ------------ Metrics ------------
acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return acc.compute(predictions=preds, references=labels)

# ------------ Model ------------
print("\n🤖 Loading base model (full train)…")
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    pad_token_id=tok.pad_token_id,
)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False  # needed when using gradient checkpointing
model.to("cuda" if torch.cuda.is_available() else "cpu")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   📊 Parameters: {trainable_params:,} trainable / {total_params:,} total (100.00%)")

# ------------ Training Args ------------
print("\n⚙️  Building TrainingArguments…")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOG_STEPS,
    eval_strategy="steps",     # if older HF, switch to evaluation_strategy
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    report_to=None,            # no W&B
    dataloader_drop_last=False,
    dataloader_num_workers=4,
    bf16=bf16_ok,
    fp16=(not bf16_ok),
    gradient_checkpointing=True,       # big memory saver
    optim=OPTIM,                        # requires bitsandbytes for *_8bit
    # torch.compile can help speed on 2.0+, but skip unless you want to try:
    # torch_compile=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ------------ Pre-train eval ------------
print("\n📊 Pre-training evaluation (zero-shot)…")
pre_eval = trainer.evaluate()
zs_acc = float(pre_eval.get("eval_accuracy", 0.0))
print(f"   Zero-shot accuracy: {zs_acc:.4f} ({zs_acc*100:.2f}%)")

# ------------ Train ------------
print("\n🚀 Starting FULL fine-tuning…")
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
print(f"   Fine-tuned accuracy: {ft_acc:.4f} ({ft_acc*100:.2f}%)")
print(f"   📈 Improvement: +{impr:.4f} ({impr*100:.2f} pts)")

# ------------ Save ------------
print("\n💾 Saving full fine-tuned model…")
trainer.save_model()
tok.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# ------------ NVML snapshot ------------
power_w = vram_used_mb = vram_total_mb = None
if NVML_OK and torch.cuda.is_available():
    try:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        power_w = nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(h)
        vram_used_mb = mem.used / (1024 ** 2)
        vram_total_mb = mem.total / (1024 ** 2)
    except Exception:
        pass

# ------------ JSON dump ------------
results = {
    "method": "full_finetune",
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
    "gpu_power_watts": float(power_w) if power_w is not None else None,
    "gpu_vram_used_mb": float(vram_used_mb) if vram_used_mb is not None else None,
    "gpu_vram_total_mb": float(vram_total_mb) if vram_total_mb is not None else None,
}

with open(f"{OUTPUT_DIR}/benchmark_results_fullft.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ JSON saved to {OUTPUT_DIR}/benchmark_results_fullft.json")

# ------------ Quick predictions ------------
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

print("\n🏁 FULL FINE-TUNING SUMMARY")
print(f" • Zero-shot acc: {zs_acc*100:.2f}%")
print(f" • Fine-tuned acc: {ft_acc*100:.2f}%  (+{impr*100:.2f} pts)")
print(f" • Time: {train_secs/60:.1f} min | Trainable params: {trainable_params:,}")
print(f" • Saved to: {OUTPUT_DIR}")

import os
import time
import json
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from adapters import (
    AutoAdapterModel,
    AdapterConfig,
    AdapterTrainer,
)

# Optional power/VRAM snapshot
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

print("🚀 TinyLlama SST-2 Adapter Fine-tuning")
print("=" * 50)

# ---- Config ----
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK_NAME    = "sst2"
OUTPUT_DIR   = "./tinyllama-sst2-adapters"
ADAPTER_KIND = "double_seq_bn"   # "double_seq_bn" = Houlsby, "seq_bn" = Pfeiffer
REDUCTION    = 16                 # bottleneck: hidden_dim / reduction
NONLIN       = "relu"
MAX_LENGTH   = 128
BATCH_SIZE   = 8                  # fits 16GB comfortably; use grad_accum to scale
GRAD_ACCUM   = 2
EPOCHS       = 3
LR           = 3e-4
WARMUP       = 0.1
LOG_STEPS    = 50
EVAL_STEPS   = 200
SAVE_STEPS   = 200

print("📋 Configuration")
print(f" • Model: {MODEL_NAME}")
print(f" • Adapter: {ADAPTER_KIND} (reduction={REDUCTION}, nonlin={NONLIN})")
print(f" • Max length: {MAX_LENGTH}")
print(f" • BS: {BATCH_SIZE} × grad_accum {GRAD_ACCUM}  |  LR: {LR}  |  Epochs: {EPOCHS}")

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

def tokenize(batch):
    enc = tok(
        batch["sentence"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = batch["label"]  # Trainer expects 'labels'
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
    device_map="auto"
)

# Add classification head and adapter (same name 'sst2')
print("🧩 Adding classification head + adapter…")
model.add_classification_head(
    TASK_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # supported
    # no label2id here
)

# (optional) also set on config for clarity/consumers:
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Build adapter config: Houlsby (double_seq_bn) or Pfeiffer (seq_bn)
adapter_cfg = AdapterConfig.load(
    ADAPTER_KIND,
    reduction_factor=REDUCTION,
    non_linearity=NONLIN,
)
model.add_adapter(TASK_NAME, config=adapter_cfg)
# Activate and set trainable modules
model.set_active_adapters(TASK_NAME)   # also selects head with same name
model.train_adapter(TASK_NAME)         # freezes base model, trains adapter + head

# Count params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
pct = 100.0 * trainable_params / total_params
print(f"   📊 Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

# ---- Training args ----
print("\n⚙️  Building TrainingArguments…")
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
    eval_strategy="steps",            # use eval_strategy in recent Transformers
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    report_to=None,
    dataloader_drop_last=False,
    bf16=bf16_ok,
    remove_unused_columns=True,
)

# ---- Trainer ----
trainer = AdapterTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---- Pre-train eval ----
print("\n📊 Pre-training evaluation (zero-shot)…")
pre_eval = trainer.evaluate()
print(f"   Zero-shot accuracy: {pre_eval['eval_accuracy']:.4f} ({pre_eval['eval_accuracy']*100:.2f}%)")

# ---- Train ----
print("\n🚀 Starting Adapter fine-tuning…")
t0 = time.time()
train_result = trainer.train()
train_secs = time.time() - t0
print("   ✅ Done.")
print(f"   ⏱️  Training time: {train_secs/60:.1f} min")
print(f"   📉 Final training loss: {train_result.training_loss:.4f}")

# ---- Post eval ----
print("\n📊 Post-training evaluation…")
post_eval = trainer.evaluate()
impr = post_eval["eval_accuracy"] - pre_eval["eval_accuracy"]
print(f"   Fine-tuned accuracy: {post_eval['eval_accuracy']:.4f} ({post_eval['eval_accuracy']*100:.2f}%)")
print(f"   📈 Improvement: +{impr:.4f} ({impr*100:.2f} pts)")

# ---- Save adapter + head ----
print("\n💾 Saving adapter + head…")
# Saves both the adapter weights and the matching classification head
model.save_adapter(OUTPUT_DIR, TASK_NAME, with_head=True)  # single folder with adapter_config.json etc.
tok.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Saved to: {OUTPUT_DIR}")

# ---- Power & VRAM snapshot (single sample) ----
power_w = vram_used_mb = vram_total_mb = None
if NVML_OK and torch.cuda.is_available():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
        mem = nvmlDeviceGetMemoryInfo(handle)
        vram_used_mb = mem.used / (1024 ** 2)
        vram_total_mb = mem.total / (1024 ** 2)
    except Exception:
        pass

# ---- JSON dump ----
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
    "zero_shot_accuracy": float(pre_eval["eval_accuracy"]),
    "fine_tuned_accuracy": float(post_eval["eval_accuracy"]),
    "improvement": float(impr),
    "training_time_minutes": train_secs / 60.0,
    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params),
    "trainable_percentage": pct,
    "gpu_power_watts": float(power_w) if power_w is not None else None,
    "gpu_vram_used_mb": float(vram_used_mb) if vram_used_mb is not None else None,
    "gpu_vram_total_mb": float(vram_total_mb) if vram_total_mb is not None else None,
}
with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=4)
print(f"   ✓ JSON saved to {OUTPUT_DIR}/benchmark_results.json")

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

samples = [
    "This movie is amazing!",
    "I hate this boring film.",
    "The acting was okay, nothing special.",
]
for s in samples:
    lab, conf = predict(s)
    print(f"'{s}' → {lab} ({conf:.3f})")

print("\n🏁 ADAPTER FINE-TUNING SUMMARY")
print(f" • Zero-shot acc: {pre_eval['eval_accuracy']*100:.2f}%")
print(f" • Fine-tuned acc: {post_eval['eval_accuracy']*100:.2f}%")
print(f" • +{impr*100:.2f} pts | {train_secs/60:.1f} min")
print(f" • Trainable params: {trainable_params:,} ({pct:.2f}%)")
print(f" • Saved to: {OUTPUT_DIR}")

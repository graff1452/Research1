import os
from typing import Dict, Any
import inspect

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import transformers

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np

# ---- Env knobs ----
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # or "TinyLlama/TinyLlama-1.1B"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./tinyllama-sst2-lora")
BATCH = int(os.getenv("BATCH", "8"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
LR = float(os.getenv("LR", "2e-4"))
EPOCHS = float(os.getenv("EPOCHS", "3"))
FP16 = os.getenv("FP16", "1") == "1"
BF16 = os.getenv("BF16", "0") == "1"
USE_4BIT_ENV = os.getenv("USE_4BIT", "1") == "1"

# ---- Try bitsandbytes only if available ----
BNB_AVAILABLE = False
if USE_4BIT_ENV:
    try:
        import bitsandbytes as bnb  # noqa: F401
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False

# ------------ Dataset ------------
ds = load_dataset("glue", "sst2")
label2id = {"negative": 0, "positive": 1}
id2label = {v: k for k, v in label2id.items()}

# ------------ Tokenizer ------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure pad token exists (LLaMA-style models lack one by default)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def preprocess(example):
    return tok(example["sentence"], truncation=True, max_length=256)

keep_cols = ["input_ids", "attention_mask", "label"]
ds = ds.map(preprocess, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep_cols])

# ------------ Config & Model ------------
cfg = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
if getattr(cfg, "pad_token_id", None) is None:
    cfg.pad_token_id = tok.pad_token_id

load_kwargs: Dict[str, Any] = {"config": cfg}

if BNB_AVAILABLE and USE_4BIT_ENV:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if FP16 else torch.bfloat16 if BF16 else torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    load_kwargs["device_map"] = "auto"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **load_kwargs)
model.config.pad_token_id = tok.pad_token_id

if BNB_AVAILABLE and USE_4BIT_ENV:
    model = prepare_model_for_kbit_training(model)

# ------------ PEFT LoRA ------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ------------ Build TrainingArguments safely ------------
# Inspect __init__ to see which arg name exists
params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
supports_eval_strategy = "eval_strategy" in params
supports_evaluation_strategy = "evaluation_strategy" in params

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=FP16,
    bf16=BF16,
    report_to="none",
)

# Add only the supported evaluation arg
if supports_eval_strategy:
    args_kwargs["eval_strategy"] = "steps"
elif supports_evaluation_strategy:
    args_kwargs["evaluation_strategy"] = "steps"
# else: very old/edge build; Trainer will still run eval when eval_steps/save_steps align

args = TrainingArguments(**args_kwargs)

data_collator = DataCollatorWithPadding(tokenizer=tok, padding=True)
metric = evaluate.load("glue", "sst2")

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return metric.compute(predictions=preds, references=pred.label_ids)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,  # v5 warns to use processing_class; fine to keep
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if getattr(model.config, "use_cache", None):
    model.config.use_cache = False

trainer.train()

# ------------ Save adapter ------------
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)

adapter_out = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_out)

print("=== Done ===")
print("LoRA adapter saved to:", adapter_out)
print("Transformers:", transformers.__version__, "| bitsandbytes used:", BNB_AVAILABLE and USE_4BIT_ENV)
print("pad_token:", tok.pad_token, "pad_token_id:", tok.pad_token_id, "model pad:", model.config.pad_token_id)

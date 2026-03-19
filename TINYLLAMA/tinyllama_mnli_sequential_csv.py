import os, time, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------
# Environment safety
# ------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

# ------------------------------------------------
# HF imports
# ------------------------------------------------
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------------------------
# Optional adapters
# ------------------------------------------------
try:
    import adapters
    ADAPTERS_IMPORT_OK = True
except Exception:
    adapters = None
    ADAPTERS_IMPORT_OK = False

# ------------------------------------------------
# Reproducibility
# ------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def bf16_supported():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8

USE_BF16 = bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
print("bf16:", USE_BF16, "fp16:", USE_FP16)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ------------------------------------------------
# Dataset: MNLI
# ------------------------------------------------
dataset = load_dataset("nyu-mll/glue", "mnli")
metric = evaluate.load("glue", "mnli")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
tokenizer.padding_side = "left"
PAD_ID = tokenizer.pad_token_id
print("pad_token:", tokenizer.pad_token, "pad_id:", PAD_ID)

MAX_LEN = 128

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=MAX_LEN,
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
dataset.set_format(type=None)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def prune_columns(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def acc_from_metrics(m):
    return float(m.get("eval_accuracy", m.get("accuracy", float("nan"))))

# ------------------------------------------------
# NVML power sampling
# ------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
    print("NVML OK")
except Exception as e:
    NVML_OK = False
    _nvml_handle = None
    print("NVML unavailable:", repr(e))

def nvml_sample():
    if not NVML_OK:
        return (float("nan"), float("nan"))
    try:
        p_mw = pynvml.nvmlDeviceGetPowerUsage(_nvml_handle)
        power_w = p_mw / 1000.0
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        vram_mb = mem.used / (1024**2)
        return (float(power_w), float(vram_mb))
    except Exception:
        return (float("nan"), float("nan"))

class PowerVRAMCallback(TrainerCallback):
    def __init__(self, every_n_steps=1):
        self.every_n_steps = every_n_steps
        self.power, self.vram = [], []
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.every_n_steps == 0:
            p, m = nvml_sample()
            self.power.append(p)
            self.vram.append(m)

    def summary(self):
        p = np.array(self.power, dtype=float)
        m = np.array(self.vram, dtype=float)
        return (
            float(np.nanmean(p)) if p.size else float("nan"),
            float(np.nanmean(m)) if m.size else float("nan"),
        )

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sam(score, wh, a):
    if wh is None or not np.isfinite(wh) or wh <= 0:
        return float("nan")
    d = math.log10(wh)
    if d == 0 or not np.isfinite(d):
        return float("nan")
    return (float(score) ** float(a)) * float(a) / d

# ------------------------------------------------
# Model builders
# ------------------------------------------------
def build_base_model(load_4bit=False):
    dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)

    quant_cfg = None
    device_map = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )
        device_map = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        problem_type="single_label_classification",
        torch_dtype=None if load_4bit else dtype,
        quantization_config=quant_cfg,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = PAD_ID

    if not load_4bit:
        model.to(device)

    return model

def make_full_ft():
    return build_base_model(False)

def make_adapters():
    if not ADAPTERS_IMPORT_OK:
        raise RuntimeError("adapters not available")
    model = build_base_model(False)
    adapters.init(model)
    model.add_adapter("mnli", config="seq_bn")
    model.set_active_adapters("mnli")
    model.train_adapter("mnli")
    return model

def make_lora(r=8, alpha=16, dropout=0.1):
    base = build_base_model(False)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(base, cfg).to(device)

def make_lora_plus():
    return make_lora(r=16, alpha=32, dropout=0.05)

def make_qlora():
    base = build_base_model(True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(base, cfg)

def make_bitfit():
    model = build_base_model(False)
    for name, p in model.named_parameters():
        p.requires_grad = ("bias" in name) or ("score" in name) or ("classifier" in name)
    return model

# ------------------------------------------------
# CSV configuration
# ------------------------------------------------
CSV_DIR = Path("./runs/mnli/csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

ROLLUP_PATH = CSV_DIR / "_ALL.csv"
SKIP_IF_EXISTS = True

CSV_COLUMNS = [
    "Method",
    "Trainable Params",
    "Time (min)",
    "Zero-shot (Acc matched)",
    "Fine-tuned (Acc matched)",
    "Epochs",
    "LR",
    "Avg Power (W)",
    "Avg VRAM (MB)",
    "Energy (Wh)",
    "SAM@1",
    "SAM@2",
    "SAM@5",
    "Note",
]

def safe_name(name):
    return (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "_")
            .replace(",", "")
    )

def write_csv(row, path):
    df = pd.DataFrame([{k: row.get(k) for k in CSV_COLUMNS}])
    df.to_csv(path, index=False)

def append_rollup(row):
    df_row = pd.DataFrame([{k: row.get(k) for k in CSV_COLUMNS}])
    if ROLLUP_PATH.exists():
        df_old = pd.read_csv(ROLLUP_PATH)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row
    df_new.to_csv(ROLLUP_PATH, index=False)

# ------------------------------------------------
# Runner
# ------------------------------------------------
def run_experiment(method_name, make_model_fn, lr, epochs):
    row = {k: None for k in CSV_COLUMNS}
    row["Method"] = method_name
    row["Epochs"] = epochs
    row["LR"] = lr

    try:
        torch.cuda.empty_cache()
        model = make_model_fn()

        z_m, _ = eval_mnli(model)
        row["Zero-shot (Acc matched)"] = z_m
        row["Trainable Params"] = count_trainable_params(model)

        pwr_cb = PowerVRAMCallback()

        args = TrainingArguments(
            output_dir=f"./runs/mnli/{safe_name(method_name)}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            dataloader_num_workers=0,
            logging_steps=200,
            report_to="none",
            bf16=USE_BF16,
            fp16=USE_FP16,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=prune_columns(dataset["train"]),
            eval_dataset=prune_columns(dataset["validation_matched"]),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[pwr_cb],
        )

        t0 = time.time()
        trainer.train()
        t1 = time.time()

        minutes = (t1 - t0) / 60
        row["Time (min)"] = minutes

        m = trainer.evaluate()
        ft_acc = acc_from_metrics(m)
        row["Fine-tuned (Acc matched)"] = ft_acc

        avg_power, avg_vram = pwr_cb.summary()
        row["Avg Power (W)"] = avg_power
        row["Avg VRAM (MB)"] = avg_vram

        wh = avg_power * (minutes / 60) if np.isfinite(avg_power) else float("nan")
        row["Energy (Wh)"] = wh

        row["SAM@1"] = sam(ft_acc, wh, 1)
        row["SAM@2"] = sam(ft_acc, wh, 2)
        row["SAM@5"] = sam(ft_acc, wh, 5)

        row["Note"] = "OK"

        del trainer, model
        torch.cuda.empty_cache()

    except Exception as e:
        row["Note"] = f"FAILED: {repr(e)}"

    return row

def eval_mnli(model):
    tmp_args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=16,
        report_to="none",
        bf16=USE_BF16,
        fp16=USE_FP16,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    tmp_trainer = Trainer(
        model=model,
        args=tmp_args,
        eval_dataset=prune_columns(dataset["validation_matched"]),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    m1 = tmp_trainer.evaluate()
    return acc_from_metrics(m1), None

# ------------------------------------------------
# Sequential execution
# ------------------------------------------------
CONFIGS = [
    ("Full FT", make_full_ft),
    ("Adapters (r=16)", make_adapters),
    ("LoRA", make_lora),
    ("LoRA+", make_lora_plus),
    ("QLoRA", make_qlora),
    ("BitFit", make_bitfit),
]

for name, fn in CONFIGS:
    csv_path = CSV_DIR / f"{safe_name(name)}.csv"

    if SKIP_IF_EXISTS and csv_path.exists():
        print("Skipping:", name)
        continue

    print("\n=== Running:", name, "===")
    row = run_experiment(name, fn, 1e-5, 5)

    write_csv(row, csv_path)
    append_rollup(row)

    print("Saved:", csv_path)
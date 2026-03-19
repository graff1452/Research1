# tinyllama_mrpc.py  (FIXED: pad_token_id set on model.config; safe adapters; 6 methods)
import os, time, math, random
import numpy as np
import pandas as pd
import torch

# Avoid transformers trying to import deepspeed (can crash if CUDA_HOME not set)
os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True

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

# Adapters may not support TinyLlama in your env; we catch failures and mark the row FAILED.
try:
    import adapters
    ADAPTERS_IMPORT_OK = True
except Exception as _e:
    adapters = None
    ADAPTERS_IMPORT_OK = False

# -----------------------------
# Repro
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 8

USE_BF16 = bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
print("bf16:", USE_BF16, "fp16:", USE_FP16)

# -----------------------------
# Model id
# -----------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# Dataset: MRPC
# -----------------------------
dataset = load_dataset("nyu-mll/glue", "mrpc")
metric = evaluate.load("glue", "mrpc")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    # Llama-family commonly has no PAD; use EOS as PAD for batching stability.
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
tokenizer.padding_side = "left"

PAD_ID = tokenizer.pad_token_id
print("tokenizer.pad_token:", tokenizer.pad_token, "pad_token_id:", PAD_ID)

MAX_LEN = 128

def tokenize(batch):
    return tokenizer(
        batch["sentence1"], batch["sentence2"],
        truncation=True,
        max_length=MAX_LEN,
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
dataset.set_format(type=None)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def acc_f1_from_metrics(m: dict):
    # evaluate() returns keys without eval_ prefix, Trainer adds eval_ prefix
    acc = m.get("eval_accuracy", m.get("accuracy", float("nan")))
    f1  = m.get("eval_f1",       m.get("f1",       float("nan")))
    return float(acc), float(f1)

print(dataset)

# -----------------------------
# NVML power + VRAM sampling
# -----------------------------
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
        self.power = []
        self.vram = []
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.every_n_steps == 0:
            p, m = nvml_sample()
            self.power.append(p)
            self.vram.append(m)

    def summary(self):
        power = np.array(self.power, dtype=float)
        vram = np.array(self.vram, dtype=float)
        avg_power = float(np.nanmean(power)) if power.size else float("nan")
        avg_vram = float(np.nanmean(vram)) if vram.size else float("nan")
        return avg_power, avg_vram

# -----------------------------
# Helpers
# -----------------------------
def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sam(score: float, wh: float, a: float) -> float:
    if wh is None or not np.isfinite(wh) or wh <= 0:
        return float("nan")
    denom = math.log10(wh)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return (float(score) ** float(a)) * float(a) / denom

# -----------------------------
# Base model builder (FIXED)
# -----------------------------
def build_base_model(load_4bit: bool = False):
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
        num_labels=2,
        problem_type="single_label_classification",
        torch_dtype=None if load_4bit else dtype,
        quantization_config=quant_cfg,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # ✅ CRITICAL: Llama-family seq-cls requires config.pad_token_id for batch_size > 1
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = PAD_ID

    # keep generation_config consistent (not strictly required for training)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = PAD_ID

    if not load_4bit:
        model.to(device)

    return model

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def eval_acc(model) -> tuple[float, float]:
    model.eval()
    tmp_args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        report_to="none",
        bf16=USE_BF16,
        fp16=USE_FP16,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    val_ds = dataset["validation"].remove_columns(
        [c for c in dataset["validation"].column_names if c not in keep_cols]
    )
    tmp_trainer = Trainer(
        model=model,
        args=tmp_args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    metrics = tmp_trainer.evaluate()
    return acc_f1_from_metrics(metrics)

# -----------------------------
# Method builders (6)
# -----------------------------
def make_full_ft():
    return build_base_model(load_4bit=False)

def make_adapters():
    if not ADAPTERS_IMPORT_OK:
        raise RuntimeError("adapters package not importable in this environment.")

    model = build_base_model(load_4bit=False)
    # adapters support is model-dependent; this may still fail and will be reported
    adapters.init(model)
    model.add_adapter("mrpc", config="seq_bn")
    model.set_active_adapters("mrpc")
    model.train_adapter("mrpc")
    return model

def make_lora(r=8, alpha=16, dropout=0.1):
    base = build_base_model(load_4bit=False)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],  # standard Llama choice
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, cfg)
    model.to(device)
    return model

def make_lora_plus():
    return make_lora(r=16, alpha=32, dropout=0.05)

def make_qlora(r=8, alpha=16, dropout=0.1):
    base = build_base_model(load_4bit=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, cfg)
    return model

def make_bitfit():
    model = build_base_model(load_4bit=False)
    # Bias-only + keep classifier head trainable
    for name, p in model.named_parameters():
        p.requires_grad = ("bias" in name) or ("score" in name) or ("classifier" in name)
    return model

# -----------------------------
# Runner
# -----------------------------
def run_experiment(method_name: str, make_model_fn, lr: float, epochs: int, optim_name: str = "adamw_torch"):
    row = {
        "Method": method_name,
        "Trainable Params": None,
        "Time (min)": None,
        "Zero-shot (Accuracy)": None,
        "Fine-tuned (Accuracy)": None,
        "Epochs": epochs,
        "LR": lr,
        "Avg Power (W)": None,
        "Avg VRAM (MB)": None,
        "Energy (Wh)": None,
        "SAM@1": None,
        "SAM@2": None,
        "SAM@5": None,
        "Note": "",
    }

    try:
        torch.cuda.empty_cache()
        model = make_model_fn()

        # Zero-shot
        z_acc, z_f1 = eval_acc(model)
        row["Zero-shot (Accuracy)"] = float(z_acc)

        # Trainable params
        row["Trainable Params"] = int(count_trainable_params(model))
        if row["Trainable Params"] == 0:
            raise RuntimeError("0 trainable params (method misconfigured)")

        pwr_cb = PowerVRAMCallback(every_n_steps=1)

        per_device_bs = 4
        grad_accum = 4

        args = TrainingArguments(
            output_dir=f"./runs/mrpc/{method_name.replace(' ','_')}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=grad_accum,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            logging_steps=50,
            report_to="none",
            bf16=USE_BF16,
            fp16=USE_FP16,
            remove_unused_columns=False,
            label_names=["labels"],
            optim=optim_name,
        )

        train_ds = dataset["train"].remove_columns([c for c in dataset["train"].column_names if c not in keep_cols])
        val_ds   = dataset["validation"].remove_columns([c for c in dataset["validation"].column_names if c not in keep_cols])

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[pwr_cb],
        )

        t0 = time.time()
        trainer.train()
        t1 = time.time()

        minutes = (t1 - t0) / 60.0
        row["Time (min)"] = float(minutes)

        metrics = trainer.evaluate()
        ft_acc, ft_f1 = acc_f1_from_metrics(metrics)
        row["Fine-tuned (Accuracy)"] = float(ft_acc)

        avg_power, avg_vram = pwr_cb.summary()
        row["Avg Power (W)"] = float(avg_power)
        row["Avg VRAM (MB)"] = float(avg_vram)

        wh = avg_power * (minutes / 60.0) if np.isfinite(avg_power) else float("nan")
        row["Energy (Wh)"] = float(wh)

        row["SAM@1"] = float(sam(ft_acc, wh, 1))
        row["SAM@2"] = float(sam(ft_acc, wh, 2))
        row["SAM@5"] = float(sam(ft_acc, wh, 5))

        row["Note"] = f"zero_f1={z_f1:.4f}, ft_f1={ft_f1:.4f}"

        del trainer, model
        torch.cuda.empty_cache()

    except Exception as e:
        row["Note"] = f"FAILED: {repr(e)}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return row

# -----------------------------
# Run all 6 methods
# -----------------------------
CONFIGS = [
    ("Full FT",         make_full_ft,  1e-5, 5, "adamw_torch"),
    ("Adapters (r=16)", make_adapters,  1e-5, 5, "adamw_torch"),
    ("LoRA",            make_lora,      1e-5, 5, "adamw_torch"),
    ("LoRA+",           make_lora_plus, 1e-5, 5, "adamw_torch"),
    ("QLoRA",           make_qlora,     1e-5, 5, "adamw_torch"),
    ("BitFit",          make_bitfit,    1e-5, 5, "adamw_torch"),
]

rows = []
for name, fn, lr, ep, optim_name in CONFIGS:
    print("\n===", name, "===")
    row = run_experiment(name, fn, lr=lr, epochs=ep, optim_name=optim_name)
    rows.append(row)
    print({k: row[k] for k in ["Method","Zero-shot (Accuracy)","Fine-tuned (Accuracy)","Time (min)","Energy (Wh)","Note"]})

df = pd.DataFrame(rows)

# Order columns nicely
cols = [
    "Method",
    "Trainable Params",
    "Time (min)",
    "Zero-shot (Accuracy)",
    "Fine-tuned (Accuracy)",
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
df = df[cols]

out_csv = "tinyllama_mrpc_benchmark_results.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df)

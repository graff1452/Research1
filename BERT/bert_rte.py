# bert_rte.py
import os, time, math, random
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

from datasets import load_dataset
import evaluate
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import adapters

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
# Dataset: RTE
# -----------------------------
dataset = load_dataset("nyu-mll/glue", "rte")
metric = evaluate.load("glue", "rte")  # accuracy

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
MAX_LEN = 128

def tokenize(batch):
    return tokenizer(
        batch["sentence1"], batch["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]

def prune_columns(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

train_ds = prune_columns(dataset["train"])
val_ds   = prune_columns(dataset["validation"])

train_ds.set_format("torch", columns=keep_cols)
val_ds.set_format("torch", columns=keep_cols)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def acc_from_metrics(m: dict) -> float:
    return float(m.get("eval_accuracy", m.get("accuracy", float("nan"))))

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

# -----------------------------
# Helpers
# -----------------------------
def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sam(score: float, wh: float, a: float) -> float:
    if wh is None or not np.isfinite(wh) or wh <= 0:
        return float("nan")
    d = math.log10(wh)
    if d == 0 or not np.isfinite(d):
        return float("nan")
    return (float(score) ** float(a)) * float(a) / d

# -----------------------------
# Base model
# -----------------------------
def build_base_model():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        problem_type="single_label_classification",
    )

@torch.no_grad()
def eval_acc(model) -> float:
    model.eval()
    tmp_args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=64,
        dataloader_drop_last=False,
        report_to="none",
        bf16=USE_BF16,
        fp16=USE_FP16,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    tmp_trainer = Trainer(
        model=model,
        args=tmp_args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    metrics = tmp_trainer.evaluate()
    return acc_from_metrics(metrics)

# -----------------------------
# Method builders (6)
# -----------------------------
def make_full_ft():
    return build_base_model().to(device)

def make_adapters():
    model = build_base_model()
    adapters.init(model)
    model.add_adapter("rte", config="seq_bn")
    model.set_active_adapters("rte")
    model.train_adapter("rte")
    model.to(device)
    return model

def make_lora(r=8, alpha=16, dropout=0.1):
    base = build_base_model()
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(base, cfg).to(device)

def make_lora_plus():
    return make_lora(r=16, alpha=32, dropout=0.05)

def make_qlora(r=8, alpha=16, dropout=0.1):
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    base = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        problem_type="single_label_classification",
        quantization_config=qcfg,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(base, cfg)

def make_bitfit():
    model = build_base_model()
    for name, p in model.named_parameters():
        p.requires_grad = ("bias" in name) or ("classifier" in name)
    model.to(device)
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

        z = eval_acc(model)
        row["Zero-shot (Accuracy)"] = float(z)

        row["Trainable Params"] = int(count_trainable_params(model))
        if row["Trainable Params"] == 0:
            raise RuntimeError("0 trainable params")

        pwr_cb = PowerVRAMCallback(every_n_steps=1)

        args = TrainingArguments(
            output_dir=f"./runs/rte/{method_name.replace(' ','_')}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=1,
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

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[pwr_cb],
        )

        t0 = time.time()
        trainer.train()
        t1 = time.time()
        minutes = (t1 - t0) / 60.0
        row["Time (min)"] = float(minutes)

        metrics = trainer.evaluate()
        ft = acc_from_metrics(metrics)
        row["Fine-tuned (Accuracy)"] = float(ft)

        avg_power, avg_vram = pwr_cb.summary()
        row["Avg Power (W)"] = float(avg_power)
        row["Avg VRAM (MB)"] = float(avg_vram)

        wh = avg_power * (minutes / 60.0) if np.isfinite(avg_power) else float("nan")
        row["Energy (Wh)"] = float(wh)

        row["SAM@1"] = float(sam(ft, wh, 1))
        row["SAM@2"] = float(sam(ft, wh, 2))
        row["SAM@5"] = float(sam(ft, wh, 5))

        del trainer, model
        torch.cuda.empty_cache()

    except Exception as e:
        row["Note"] = f"FAILED: {repr(e)}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return row

CONFIGS = [
    ("Full FT",         make_full_ft,  1e-5, 5, "adamw_torch"),
    ("Adapters (r=16)", make_adapters,  1e-5, 5, "adamw_torch"),
    ("LoRA",            make_lora,      1e-5, 5, "adamw_torch"),
    ("LoRA+",           make_lora_plus, 1e-5, 5, "adamw_torch"),
    ("QLoRA",           make_qlora,     1e-5, 5, "adamw_torch"),
    ("BitFit",          make_bitfit,    1e-5, 5, "adamw_torch"),
]

rows = []
for name, fn, lr, ep, optim in CONFIGS:
    print("\n===", name, "===")
    r = run_experiment(name, fn, lr, ep, optim)
    rows.append(r)
    print({k: r[k] for k in ["Method","Zero-shot (Accuracy)","Fine-tuned (Accuracy)","Time (min)","Energy (Wh)","Note"]})

df = pd.DataFrame(rows)
out_csv = "bert_rte_benchmark_results.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df)
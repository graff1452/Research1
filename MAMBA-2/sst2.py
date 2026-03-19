# main_sst2.py  (Mamba2 370M / 16GB) — SST-2, runs 6 methods
import os, time, math, random, json, re
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    major, minor = torch.cuda.get_device_capability(0)
    return major >= 8

USE_BF16 = bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
print("bf16:", USE_BF16, "fp16:", USE_FP16)

# -----------------------------
# Model id
# -----------------------------
MODEL_ID = "benchang1110/mamba2-370m-hf"

# -----------------------------
# Dataset: SST-2 (GLUE)
# -----------------------------
TASK_NAME = "sst2"
METRIC_KEY = "accuracy"

dataset = load_dataset("nyu-mll/glue", TASK_NAME)
metric = evaluate.load("glue", TASK_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
tokenizer.padding_side = "left"

MAX_LEN = 128

def tokenize(batch):
    texts = [f"Sentence: {s}\nSentiment:" for s in batch["sentence"]]
    return tokenizer(texts, truncation=True, max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

def cast_labels(batch):
    batch["labels"] = [int(x) for x in batch["labels"]]
    return batch

dataset = dataset.map(cast_labels, batched=True)

keep_cols = ["input_ids", "attention_mask", "labels"]
dataset.set_format(type=None)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    pred_ids = np.argmax(preds, axis=-1).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    return metric.compute(predictions=pred_ids, references=labels)

def score_from_metrics(m: dict) -> float:
    if f"eval_{METRIC_KEY}" in m:
        return float(m[f"eval_{METRIC_KEY}"])
    if METRIC_KEY in m:
        return float(m[METRIC_KEY])
    return float("nan")

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

def print_mem(tag: str):
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    print(f"[mem {tag}] free={free/1e9:.2f}GB total={total/1e9:.2f}GB")

# -----------------------------
# Patch invalid config.json (Infinity)
# -----------------------------
def load_patched_config_dict(model_id: str) -> dict:
    cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()
    text_fixed = re.sub(r"\bInfinity\b", "1e9", text)
    cfg = json.loads(text_fixed)
    if "time_step_limit" in cfg and isinstance(cfg["time_step_limit"], list):
        cfg["time_step_limit"] = [0.0, float("inf")]
    return cfg

# -----------------------------
# Classification wrapper (binary)
# -----------------------------
class Mamba2ClassificationModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_labels = int(num_labels)
        self.cls_head = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(input_ids=input_ids, **kwargs)
        hidden_states = out.last_hidden_state  # [B, T, H]

        # pool last real token
        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
        else:
            idx = attention_mask.long().sum(dim=1) - 1
            idx = torch.clamp(idx, min=0)
            bsz = hidden_states.size(0)
            pooled = hidden_states[torch.arange(bsz, device=hidden_states.device), idx, :]

        if self.cls_head is None:
            self.cls_head = torch.nn.Linear(pooled.size(-1), self.num_labels).to(pooled.device).to(pooled.dtype)

        logits = self.cls_head(pooled)  # [B, num_labels]

        loss = None
        if labels is not None:
            labels = labels.long().view(-1)
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def ensure_cls_head_initialized(model: Mamba2ClassificationModel):
    model.eval()
    with torch.no_grad():
        batch = tokenizer("init", return_tensors="pt")
        dev = next(model.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))

# -----------------------------
# Backbone builder
# -----------------------------
def build_backbone(load_4bit: bool):
    cfg = load_patched_config_dict(MODEL_ID)

    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

    dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)

    backbone = AutoModel.from_pretrained(
        MODEL_ID,
        config=cfg,
        torch_dtype=dtype if not load_4bit else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        device_map="auto" if load_4bit else None,
    )

    if hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable()
        except Exception:
            pass

    if not load_4bit:
        backbone.to(device)

    return backbone

# -----------------------------
# LoRA targets (Mamba-safe)
# -----------------------------
def mamba_safe_lora_targets(model: torch.nn.Module):
    allowed_suffixes = {"in_proj", "x_proj", "dt_proj"}
    found = set()

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in allowed_suffixes:
                found.add(suffix)

    if not found:
        banned = {"out_proj", "conv1d"}
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                suffix = name.split(".")[-1]
                if suffix not in banned:
                    found.add(suffix)

    return sorted(found)

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def eval_score(model) -> float:
    model.eval()
    tmp_args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=USE_BF16,
        fp16=USE_FP16,
    )
    val_ds = dataset["validation"].remove_columns([c for c in dataset["validation"].column_names if c not in keep_cols])

    tmp_trainer = Trainer(
        model=model,
        args=tmp_args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    metrics = tmp_trainer.evaluate()
    return score_from_metrics(metrics)

# -----------------------------
# Method builders (6)
# -----------------------------
def make_ft():
    backbone = build_backbone(load_4bit=False)
    model = Mamba2ClassificationModel(backbone, num_labels=2)
    model.to(device)
    ensure_cls_head_initialized(model)
    return model

def make_adapters():
    raise RuntimeError("Adapters not supported for Mamba2 (adapter-transformers targets Transformer blocks).")

def make_lora(r=8, alpha=16, dropout=0.1):
    backbone = build_backbone(load_4bit=False)
    targets = mamba_safe_lora_targets(backbone)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    backbone = get_peft_model(backbone, cfg)
    model = Mamba2ClassificationModel(backbone, num_labels=2)
    model.to(device)
    ensure_cls_head_initialized(model)
    return model

def make_lora_plus():
    return make_lora(r=16, alpha=32, dropout=0.05)

def make_qlora():
    # NOTE: load_4bit=False to avoid known 1048576 mismatch for this repo
    backbone = build_backbone(load_4bit=False)
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)

    targets = mamba_safe_lora_targets(backbone)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    backbone = get_peft_model(backbone, cfg)
    model = Mamba2ClassificationModel(backbone, num_labels=2)
    model.to(device)
    ensure_cls_head_initialized(model)
    return model

def make_bitfit():
    # NOTE: load_4bit=False to avoid known 1048576 mismatch for this repo
    backbone = build_backbone(load_4bit=False)
    model = Mamba2ClassificationModel(backbone, num_labels=2)
    model.to(device)
    ensure_cls_head_initialized(model)

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if ("bias" in name) or ("cls_head" in name):
            p.requires_grad = True

    trainable = count_trainable_params(model)
    if trainable == 0:
        raise RuntimeError("BitFit produced 0 trainable params (unexpected).")
    return model

# -----------------------------
# Runner
# -----------------------------
def run_experiment(method_name: str, make_model_fn, lr: float, epochs: int, optim_name: str):
    row = {
        "Method": method_name,
        "Trainable Params": None,
        "Time (min)": None,
        f"Zero-shot ({METRIC_KEY})": None,
        f"Fine-tuned ({METRIC_KEY})": None,
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
        print_mem(f"{method_name} start")

        model = make_model_fn()

        z = eval_score(model)
        row[f"Zero-shot ({METRIC_KEY})"] = float(z)

        row["Trainable Params"] = int(count_trainable_params(model))
        if row["Trainable Params"] == 0:
            raise RuntimeError("0 trainable params (method misconfigured)")

        pwr_cb = PowerVRAMCallback(every_n_steps=1)

        per_device_bs = 4
        grad_accum = 4

        args = TrainingArguments(
            output_dir=f"./runs/{TASK_NAME}/{method_name.replace(' ','_')}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=grad_accum,
            dataloader_num_workers=0,
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
        ft = score_from_metrics(metrics)
        row[f"Fine-tuned ({METRIC_KEY})"] = float(ft)

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
        print_mem(f"{method_name} end")

    except Exception as e:
        row["Note"] = f"FAILED: {repr(e)}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print_mem(f"{method_name} end")

    return row

# -----------------------------
# Run all 6 methods
# -----------------------------
print("FT optimizer: adamw_bnb_8bit")

CONFIGS = [
    ("FT",      make_ft,        1e-5, 5, "adamw_bnb_8bit"),
    ("Adapters", make_adapters,  1e-5, 5, "adamw_torch"),
    ("LoRA",     make_lora,      1e-5, 5, "adamw_torch"),
    ("LoRA+",    make_lora_plus, 1e-5, 5, "adamw_torch"),
    ("QLoRA",    make_qlora,     1e-5, 5, "adamw_torch"),
    ("BitFit",   make_bitfit,    1e-5, 5, "adamw_torch"),
]

rows = []
for name, fn, lr, ep, optim_name in CONFIGS:
    print("\n===", name, "===")
    row = run_experiment(name, fn, lr=lr, epochs=ep, optim_name=optim_name)
    rows.append(row)
    print({k: row.get(k) for k in ["Method", f"Zero-shot ({METRIC_KEY})", f"Fine-tuned ({METRIC_KEY})", "Time (min)", "Energy (Wh)", "Note"]})

df = pd.DataFrame(rows)
cols = [
    "Method",
    "Trainable Params",
    "Time (min)",
    f"Zero-shot ({METRIC_KEY})",
    f"Fine-tuned ({METRIC_KEY})",
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
print(df)

out_csv = "mamba2_370m_sst2_benchmark_results.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

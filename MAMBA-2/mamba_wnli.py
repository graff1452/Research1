# mamba2_wnli.py
import os, time, math, random, json, re
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TRANSFORMERS_NO_DEEPSPEED", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
torch.backends.cuda.matmul.allow_tf32 = True

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

seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

MODEL_ID = "benchang1110/mamba2-370m-hf"

dataset = load_dataset("nyu-mll/glue", "wnli")
metric = evaluate.load("glue", "wnli")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
tokenizer.padding_side = "left"

MAX_LEN = 256

def tokenize(batch):
    texts = [
        f"Sentence 1: {s1}\nSentence 2: {s2}\nTask: entailment or not? Answer:"
        for s1, s2 in zip(batch["sentence1"], batch["sentence2"])
    ]
    return tokenizer(texts, truncation=True, max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
dataset.set_format(type=None)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def prune(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

train_ds = prune(dataset["train"])
val_ds   = prune(dataset["validation"])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def acc_from_metrics(m: dict) -> float:
    return float(m.get("eval_accuracy", m.get("accuracy", float("nan"))))

# NVML
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
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return (p_mw/1000.0, mem.used/(1024**2))
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
            self.power.append(p); self.vram.append(m)
    def summary(self):
        p = np.array(self.power, float); m = np.array(self.vram, float)
        return (float(np.nanmean(p)) if p.size else float("nan"),
                float(np.nanmean(m)) if m.size else float("nan"))

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sam(score: float, wh: float, a: float) -> float:
    if wh is None or not np.isfinite(wh) or wh <= 0:
        return float("nan")
    d = math.log10(wh)
    if d == 0 or not np.isfinite(d):
        return float("nan")
    return (float(score) ** float(a)) * float(a) / d

def load_patched_config_dict(model_id: str) -> dict:
    cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()
    text_fixed = re.sub(r"\bInfinity\b", "1e9", text)
    cfg = json.loads(text_fixed)
    if "time_step_limit" in cfg and isinstance(cfg["time_step_limit"], list):
        cfg["time_step_limit"] = [0.0, float("inf")]
    return cfg

class Mamba2SequenceClassificationModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.cls_head = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(input_ids=input_ids, **kwargs)
        hidden = out.last_hidden_state  # [B,T,H]

        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            idx = attention_mask.long().sum(dim=1) - 1
            idx = torch.clamp(idx, min=0)
            bsz = hidden.size(0)
            pooled = hidden[torch.arange(bsz, device=hidden.device), idx, :]

        if pooled.dim() != 2:
            pooled = pooled.view(pooled.size(0), -1)

        if self.cls_head is None:
            self.cls_head = torch.nn.Linear(pooled.size(-1), self.num_labels).to(pooled.device).to(pooled.dtype)

        logits = self.cls_head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.float(), labels.long().view(-1))
        return {"loss": loss, "logits": logits}

def ensure_head_initialized(model: Mamba2SequenceClassificationModel):
    model.eval()
    with torch.no_grad():
        batch = tokenizer("init", return_tensors="pt", truncation=True, max_length=8)
        dev = next(model.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))

def build_backbone(load_4bit: bool):
    cfg = load_patched_config_dict(MODEL_ID)
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
        device_map = {"": 0} if torch.cuda.is_available() else None

    backbone = AutoModel.from_pretrained(
        MODEL_ID,
        config=cfg,
        torch_dtype=dtype if not load_4bit else None,
        quantization_config=quant_cfg,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if hasattr(backbone, "config"):
        try: backbone.config.use_cache = False
        except Exception: pass
        try: backbone.config.return_dict = True
        except Exception: pass
    if not load_4bit:
        backbone.to(device)
    return backbone

def mamba_safe_lora_targets(model: torch.nn.Module):
    allowed_suffixes = {"in_proj", "x_proj", "dt_proj"}
    found = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            suf = name.split(".")[-1]
            if suf in allowed_suffixes:
                found.add(suf)
    if not found:
        banned = {"out_proj", "conv1d"}
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                suf = name.split(".")[-1]
                if suf not in banned:
                    found.add(suf)
    return sorted(found)

@torch.no_grad()
def eval_acc(model) -> float:
    model.eval()
    tmp_args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=16,
        report_to="none",
        bf16=USE_BF16, fp16=USE_FP16,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    tmp_trainer = Trainer(
        model=model, args=tmp_args, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    m = tmp_trainer.evaluate()
    return acc_from_metrics(m)

def make_ft():
    backbone = build_backbone(False)
    model = Mamba2SequenceClassificationModel(backbone, 2).to(device)
    ensure_head_initialized(model)
    return model

def make_adapters():
    raise RuntimeError("Adapters not supported for Mamba2 (adapter-transformers targets Transformer blocks).")

def make_lora(r=8, alpha=16, dropout=0.1):
    backbone = build_backbone(False)
    targets = mamba_safe_lora_targets(backbone)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets,
                     lora_dropout=dropout, bias="none", task_type="SEQ_CLS")
    backbone = get_peft_model(backbone, cfg)
    model = Mamba2SequenceClassificationModel(backbone, 2).to(device)
    ensure_head_initialized(model)
    return model

def make_lora_plus():
    return make_lora(r=16, alpha=32, dropout=0.05)

def make_qlora():
    backbone = build_backbone(True)
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
    targets = mamba_safe_lora_targets(backbone)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=targets,
                     lora_dropout=0.1, bias="none", task_type="SEQ_CLS")
    backbone = get_peft_model(backbone, cfg)
    model = Mamba2SequenceClassificationModel(backbone, 2)
    ensure_head_initialized(model)
    return model

def make_bitfit():
    backbone = build_backbone(False)
    model = Mamba2SequenceClassificationModel(backbone, 2).to(device)
    ensure_head_initialized(model)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if ("bias" in name) or ("cls_head" in name):
            p.requires_grad = True
    return model

def run_experiment(method_name, make_model_fn, lr, epochs, optim_name="adamw_torch"):
    row = {
        "Method": method_name,
        "Trainable Params": None,
        "Time (min)": None,
        "Zero-shot (Accuracy)": None,
        "Fine-tuned (Accuracy)": None,
        "Epochs": epochs, "LR": lr,
        "Avg Power (W)": None, "Avg VRAM (MB)": None,
        "Energy (Wh)": None,
        "SAM@1": None, "SAM@2": None, "SAM@5": None,
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

        pwr_cb = PowerVRAMCallback(1)

        per_device_bs = 32
        grad_accum = 1

        args = TrainingArguments(
            output_dir=f"./runs/wnli/{method_name.replace(' ','_')}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=grad_accum,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            logging_steps=50,
            report_to="none",
            bf16=USE_BF16, fp16=USE_FP16,
            remove_unused_columns=False,
            label_names=["labels"],
            optim=optim_name,
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            tokenizer=tokenizer, data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[pwr_cb],
        )

        t0 = time.time(); trainer.train(); t1 = time.time()
        minutes = (t1 - t0) / 60.0
        row["Time (min)"] = float(minutes)

        m = trainer.evaluate()
        ft = acc_from_metrics(m)
        row["Fine-tuned (Accuracy)"] = float(ft)

        avg_power, avg_vram = pwr_cb.summary()
        row["Avg Power (W)"] = float(avg_power)
        row["Avg VRAM (MB)"] = float(avg_vram)

        wh = avg_power * (minutes/60.0) if np.isfinite(avg_power) else float("nan")
        row["Energy (Wh)"] = float(wh)
        row["SAM@1"] = float(sam(ft, wh, 1))
        row["SAM@2"] = float(sam(ft, wh, 2))
        row["SAM@5"] = float(sam(ft, wh, 5))

        del trainer, model
        torch.cuda.empty_cache()
    except Exception as e:
        row["Note"] = f"FAILED: {repr(e)}"
        try: torch.cuda.empty_cache()
        except Exception: pass
    return row

CONFIGS = [
    ("Full FT",         make_ft,        1e-5, 5, "adamw_torch"),
    ("Adapters (r=16)", make_adapters,  1e-5, 5, "adamw_torch"),
    ("LoRA",            make_lora,      1e-5, 5, "adamw_torch"),
    ("LoRA+",           make_lora_plus, 1e-5, 5, "adamw_torch"),
    ("QLoRA",           make_qlora,     1e-5, 5, "adamw_torch"),
    ("BitFit",          make_bitfit,    1e-5, 5, "adamw_torch"),
]

rows = []
for name, fn, lr, ep, opt in CONFIGS:
    print("\n===", name, "===")
    r = run_experiment(name, fn, lr, ep, opt)
    rows.append(r)
    print({k: r[k] for k in ["Method","Zero-shot (Accuracy)","Fine-tuned (Accuracy)","Time (min)","Energy (Wh)","Note"]})

df = pd.DataFrame(rows)
out_csv = "mamba2_370m_wnli_benchmark_results.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df)
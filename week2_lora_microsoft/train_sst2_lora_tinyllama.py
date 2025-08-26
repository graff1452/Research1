import os, torch, loralib as lora
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, LlamaForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import evaluate

# ---- Config ----
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_LABELS = 2
RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_LEN = 256

# ---- 1) Load Data ----
ds = load_dataset("nyu-mll/glue", "sst2")
metric = evaluate.load("accuracy")  # SST-2 uses accuracy

# ---- 2) Tokenizer ----
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def preprocess(ex):
    out = tok(ex["sentence"], truncation=True, max_length=MAX_LEN)
    out["labels"] = ex["label"]
    return out

ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

# ---- 3) Model ----
cfg = AutoConfig.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS,
    pad_token_id=tok.pad_token_id,
    problem_type="single_label_classification",
)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_ID, config=cfg, torch_dtype=dtype
)

# ---- 4) Add LoRA ----
def convert_linear_to_lora(module, name_filter=("q_proj", "v_proj")):
    for name, child in list(module.named_children()):
        convert_linear_to_lora(child, name_filter)  # recurse
        if isinstance(child, nn.Linear) and name in name_filter:
            new_lin = lora.Linear(
                child.in_features, child.out_features,
                r=RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                bias=(child.bias is not None)
            )
            with torch.no_grad():
                new_lin.weight.copy_(child.weight)
                if child.bias is not None:
                    new_lin.bias.copy_(child.bias)
            setattr(module, name, new_lin)

convert_linear_to_lora(model)

# ---- 5) Freeze except LoRA + classifier ----
lora.mark_only_lora_as_trainable(model)
for p in model.score.parameters():
    p.requires_grad = True

# ---- 6) Training Args ----
args = TrainingArguments(
    output_dir="out_sst2_tinyllama_lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,     # <<< keeps only last checkpoint
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# ---- 7) Metrics ----
def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": metric.compute(predictions=preds, references=labels)["accuracy"]}

# ---- 8) Trainer ----
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ---- 9) Train ----
trainer.train()

# ---- 10) Save ----
# Full model + tokenizer
trainer.save_model()
tok.save_pretrained(args.output_dir)

# LoRA-only adapter (tiny file)
torch.save(
    lora.lora_state_dict(model),
    os.path.join(args.output_dir, "lora_sst2_adapter.pt")
)

print("Training done. Model + adapter saved in:", args.output_dir)

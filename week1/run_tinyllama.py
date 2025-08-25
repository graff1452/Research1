import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load

# -------------------------
# Configuration
# -------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
subset_size = 50          # number of validation examples to test
max_new_tokens = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model and tokenizer
# -------------------------
print("Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# -------------------------
# Helper: create SST-2 prompt
# -------------------------
def create_prompt(example):
    return f"Classify the sentiment of this sentence as positive or negative:\n\"{example['sentence']}\""

# -------------------------
# Benchmark SST-2
# -------------------------
print("\n=== Benchmarking SST-2 ===")
dataset = load_dataset("glue", "sst2", split=f"validation[:{subset_size}]")
predictions = []

for example in dataset:
    prompt = create_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    # map text output to sentiment labels
    if "positive" in text:
        predictions.append(1)
    else:
        predictions.append(0)

# -------------------------
# Evaluate
# -------------------------
metric = load("glue", "sst2")
results = metric.compute(predictions=predictions, references=dataset["label"])
print("SST-2 results:", results)

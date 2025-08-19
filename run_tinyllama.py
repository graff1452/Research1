import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
import re

# -------------------------
# Configuration
# -------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"
]
subset_size = 50          # examples per validation split for testing
max_new_tokens = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model and tokenizer
# -------------------------
print("Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# -------------------------
# Helper: create prompts
# -------------------------
def create_prompt(task_name, example):
    if task_name == "cola":
        return f"Is the following sentence grammatically correct? Answer 'acceptable' or 'unacceptable'.\n\"{example['sentence']}\""
    elif task_name == "sst2":
        return f"Classify the sentiment of this sentence as positive or negative:\n\"{example['sentence']}\""
    elif task_name in ["mrpc", "qqp"]:
        s1 = example.get('sentence1', example.get('question1', ''))
        s2 = example.get('sentence2', example.get('question2', ''))
        return f"Are these two sentences paraphrases of each other? Answer 'yes' or 'no'.\nSentence 1: \"{s1}\"\nSentence 2: \"{s2}\""
    elif task_name == "stsb":
        s1 = example['sentence1']
        s2 = example['sentence2']
        return f"How similar are these sentences on a scale 0-5?\nSentence 1: \"{s1}\"\nSentence 2: \"{s2}\""
    elif task_name in ["mnli", "qnli", "rte", "wnli"]:
        premise = example.get('premise', example.get('sentence', ''))
        hypothesis = example.get('hypothesis', example.get('question', ''))
        return f"Does the premise entail the hypothesis? Answer 'entailment', 'contradiction', or 'neutral'.\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\""
    else:
        return str(example)

# -------------------------
# Benchmark loop
# -------------------------
for task_name in tasks:
    print(f"\n=== Benchmarking {task_name.upper()} ===")

    # handle MNLI separate validation splits
    if task_name == "mnli":
        validation_splits = ["validation_matched", "validation_mismatched"]
    else:
        validation_splits = ["validation"]

    for val_split in validation_splits:
        print(f"\n--- Split: {val_split} ---")
        dataset = load_dataset("glue", task_name, split=f"{val_split}[:{subset_size}]")
        predictions = []

        for example in dataset:
            prompt = create_prompt(task_name, example)
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

            # map text output to labels
            if task_name == "cola":
                predictions.append(1 if "acceptable" in text else 0)
            elif task_name == "sst2":
                predictions.append(1 if "positive" in text else 0)
            elif task_name in ["mrpc", "qqp"]:
                predictions.append(1 if "yes" in text else 0)
            elif task_name == "stsb":
                match = re.search(r"\d+\.?\d*", text)
                predictions.append(float(match.group(0)) if match else 0.0)
            elif task_name in ["mnli", "qnli", "rte", "wnli"]:
                if "entailment" in text:
                    predictions.append(0)
                elif "contradiction" in text:
                    predictions.append(1)
                elif "neutral" in text:
                    predictions.append(2)
                else:
                    predictions.append(-1)
            else:
                predictions.append(-1)

        # Evaluate
        metric = load("glue", task_name)
        results = metric.compute(predictions=predictions, references=dataset["label"])
        print(f"{val_split} results:", results)

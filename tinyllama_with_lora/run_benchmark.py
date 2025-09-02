import time
import torch
import subprocess
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np
from peft import AutoPeftModelForSequenceClassification

# Step 1: Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fine-tuned model directory
OUTPUT_DIR = "./tinyllama-sst2-lora"
BATCH_SIZE = 16  # Batch size for evaluation

# Step 2: Load Dataset
print(f"\n📁 Loading GLUE SST-2 dataset...")
dataset = load_dataset("glue", "sst2")

# Step 3: Load Tokenizer and Model (Load fine-tuned model from OUTPUT_DIR)
print(f"\n🤖 Loading fine-tuned model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the fine-tuned model with LoRA applied
model = AutoPeftModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    pad_token_id=tokenizer.pad_token_id
)

print(f"   ✓ Fine-tuned model loaded successfully")

# Step 4: Tokenize Dataset (same as during fine-tuning)
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"], 
        truncation=True, 
        padding=False, 
        max_length=512
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"],
    desc="Tokenizing"
)

# Step 5: Data Collator and Metrics
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# Step 6: Training Arguments (for evaluation)
training_args = TrainingArguments(
    output_dir="./results", 
    per_device_eval_batch_size=BATCH_SIZE,  # Use the same batch size
    logging_dir="./logs",
    do_train=False,  # We are not training, just evaluating
    do_eval=True,    # Ensure evaluation is enabled
)

# Step 7: Create Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["validation"],  # Evaluate on the validation set
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Step 8: Evaluation (Zero-shot evaluation)
print(f"\n📊 Running evaluation on fine-tuned model...")

# Function to get GPU memory usage
def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # in GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Reserved memory: {reserved_memory:.2f} GB")

# Function to get power consumption using nvidia-smi
def get_gpu_power():
    power = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    return float(power.strip())

# Start measuring time for evaluation
start_time = time.time()

# Print VRAM usage and power before evaluation
print_gpu_memory()
initial_power = get_gpu_power()
print(f"Initial Power Consumption: {initial_power:.2f} W")

# Run evaluation
eval_result = trainer.evaluate()

# End measuring time
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time

# Print VRAM usage and power after evaluation
print(f"\n📊 Evaluation completed in {total_time:.2f} seconds")
print_gpu_memory()
final_power = get_gpu_power()
print(f"Final Power Consumption: {final_power:.2f} W")

# Power consumption during evaluation
power_consumed = final_power - initial_power
print(f"Power consumed during evaluation: {power_consumed:.2f} W")

# Print the final evaluation results
print(f"\n📊 Evaluation Results:")
print(f"   Accuracy: {eval_result['eval_accuracy'] * 100:.2f}%")

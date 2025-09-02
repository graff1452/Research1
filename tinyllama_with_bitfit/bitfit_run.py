import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
from pynvml import *

# Initialize NVML for GPU monitoring
nvmlInit()

# Track timing, VRAM, and power usage
def get_gpu_usage():
    handle = nvmlDeviceGetHandleByIndex(0)  # Get the first GPU
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
    return {
        "vram_used": mem_info.used / 1e9,  # Convert to GB
        "vram_total": mem_info.total / 1e9,  # Convert to GB
        "power_usage": power_usage  # Power in Watts
    }

# Step 1: Load the GLUE SST-2 dataset
print("📁 Loading GLUE SST-2 dataset...")
dataset = load_dataset("glue", "sst2")

# Step 2: Load Tokenizer
print("🤖 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Fix padding token setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✅ Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# Step 3: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Load the Pre-Trained Model
print("🤖 Loading pre-trained TinyLlama model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "./bitfit_tinyllama_sst2",  # Path to the fine-tuned model checkpoint
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    torch_dtype=torch.float32,  # Use float32 for stable inference
    device_map="auto" if torch.cuda.is_available() else None,
    pad_token_id=tokenizer.pad_token_id  # Ensure model knows the pad token
)

# Step 5: Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# Step 6: Trainer (No Training, Just Evaluation)
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",  # Save evaluation results
        do_train=False,  # Skip training
        do_eval=True,  # Perform evaluation
    ),
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Step 7: Benchmarking
print("🚀 Starting benchmarking (evaluation)...")
training_results = {
    "benchmark_time": None,
    "vram_usage": [],
    "power_usage": []
}

# Track the start time for benchmarking
start_time = time.time()

# Evaluate the model (Benchmarking)
try:
    eval_results = trainer.evaluate()
    training_results["benchmark_time"] = time.time() - start_time  # Record benchmarking time
    print("✅ Benchmarking completed successfully!")

except Exception as e:
    print(f"❌ Benchmarking failed: {e}")
    raise e  # Re-raise to see full traceback if needed

# Step 8: Display Results
print("\n=== FINAL BENCHMARK RESULTS ===")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Step 9: Test Predictions (Optional)
def test_prediction(text):
    """Test a single prediction with proper device handling"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    confidence = probabilities[0][predicted_class].item()
    
    return label, confidence

# Step 10: Test Example Predictions
test_sentences = [
    "This movie is absolutely fantastic!",
    "Worst movie I've ever seen, terrible acting.",
    "The film was okay, not great but watchable.",
    "I loved every minute of this amazing story!",
    "Boring and predictable plot."
]

print("\n🧪 Testing predictions:")
for sentence in test_sentences:
    try:
        label, confidence = test_prediction(sentence)
        print(f"Text: '{sentence}'")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()
    except Exception as e:
        print(f"❌ Error testing sentence: {e}")

print("🎉 Benchmarking complete!")

import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
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

# Step 4: Load the Model
print("🤖 Loading TinyLlama model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    torch_dtype=torch.float32,  # Use float32 for stable training
    device_map="auto" if torch.cuda.is_available() else None,
    pad_token_id=tokenizer.pad_token_id  # Ensure model knows the pad token
)

# Step 5: Enhanced BitFit Implementation
print("🔧 Applying Enhanced BitFit...")

def apply_enhanced_bitfit(model):
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if 'score' in name or 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✅ Training: {name} ({param.numel()} params)")
        
        elif 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✅ Training: {name} ({param.numel()} params)")
        
        elif 'norm' in name.lower() or 'layer_norm' in name.lower():
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✅ Training: {name} ({param.numel()} params)")
        
        else:
            param.requires_grad = False
    
    print(f"\n📊 Training {trainable_params:,} out of {total_params:,} parameters")
    print(f"📊 Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return trainable_params, total_params

trainable_params, total_params = apply_enhanced_bitfit(model)

# Step 6: Data Collator for better batching
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 7: Training Arguments (GPU optimized)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Larger batch size
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=500,
    save_steps=1000,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    dataloader_drop_last=False,
    fp16=True,  # Enable FP16
    bf16=False,
    dataloader_num_workers=4,  # Increase number of workers for data loading
    max_grad_norm=1.0,
    report_to=None,
    save_total_limit=3
)


# Step 8: Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# Step 9: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Step 10: Start Training
print("🚀 Starting BitFit training...")
print(f"📊 Tokenizer pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"📊 Model pad_token_id: {model.config.pad_token_id}")

# Initialize results dictionary
training_results = {
    "training_time": None,
    "vram_usage": [],
    "power_usage": []
}

# Track the start time
start_time = time.time()

# Training Loop
try:
    trainer.train()
    training_results["training_time"] = time.time() - start_time  # Record training time
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("💡 Common fixes:")
    print("  - Check tokenizer padding setup")
    print("  - Try reducing batch size")
    print("  - Ensure GPU memory is sufficient")
    raise e  # Re-raise to see full traceback if needed

# Step 11: Save the Model
print("💾 Saving the fine-tuned model...")
trainer.save_model("./bitfit_tinyllama_sst2")
tokenizer.save_pretrained("./bitfit_tinyllama_sst2")

# Step 12: Final Evaluation
print("📊 Final evaluation...")
eval_results = trainer.evaluate()
print("\n=== FINAL RESULTS ===")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Step 13: Save results in JSON
print("💾 Saving results to JSON...")
gpu_usage = get_gpu_usage()  # Get current GPU usage
training_results["vram_usage"].append(gpu_usage["vram_used"])
training_results["power_usage"].append(gpu_usage["power_usage"])

# Save to JSON
with open("training_results.json", "w") as f:
    json.dump(training_results, f, indent=4)

# Step 14: Test Predictions
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

# Step 15: Test Examples
test_sentences = [
    "This movie is absolutely fantastic!",
    "Worst movie I've ever seen, terrible acting.",
    "The film was okay, not great but watchable.",
    "I loved every minute of this amazing story!",
    "Boring and predictable plot."
]

print("\n🧪 Testing predictions:")
print("-" * 50)
for sentence in test_sentences:
    try:
        label, confidence = test_prediction(sentence)
        print(f"Text: '{sentence}'")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()
    except Exception as e:
        print(f"❌ Error testing sentence: {e}")

print("🎉 BitFit training complete!")

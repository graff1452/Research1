import time
import torch
import json
import numpy as np
import pynvml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import adapters

# Initialize GPU monitoring
pynvml.nvmlInit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Function to get current VRAM usage
def get_vram_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024 ** 2  # in MB

# Function to get current power usage
def get_power_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    power = pynvml.nvmlDeviceGetPowerUsage(handle)
    return power / 1000  # in Watts

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

# Step 4: Load the Model with Adapter Configuration
print("🤖 Loading TinyLlama model with adapters...")
model = AutoModelForSequenceClassification.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    pad_token_id=tokenizer.pad_token_id
)

# Initialize adapter functionality
adapters.init(model)

# Add adapter layer
model.add_adapter("sst2_adapter", config="pfeiffer")
model.train_adapter("sst2_adapter")

# Freeze all parameters except for adapter layers
for name, param in model.named_parameters():
    if "sst2_adapter" not in name:
        param.requires_grad = False

# Step 5: Data Collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 6: Training Arguments (GPU optimized)
training_args = TrainingArguments(
    output_dir="./results",       
    num_train_epochs=3,            
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=16, 
    gradient_accumulation_steps=2, 
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
    fp16=False,                   
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    dataloader_num_workers=0,      
    max_grad_norm=1.0,             
    report_to=None,               
    save_total_limit=3             
)

# Step 7: Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# Step 8: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Initialize tracking variables for JSON output
training_info = {
    "start_time": time.time(),
    "vram_usage_start": get_vram_usage(),
    "power_usage_start": get_power_usage(),
    "steps": [],
    "total_time": 0,
    "final_vram_usage": 0,
    "final_power_usage": 0,
    "final_accuracy": 0.0
}

# Step 9: Start Training
try:
    print("🚀 Starting training with adapter layers...")
    trainer.train()

    # Log final metrics
    training_info["total_time"] = time.time() - training_info["start_time"]
    training_info["final_vram_usage"] = get_vram_usage()
    training_info["final_power_usage"] = get_power_usage()
    training_info["final_accuracy"] = trainer.evaluate()["eval_accuracy"]

    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")

# Step 10: Save the Model and Tokenizer
print("💾 Saving the fine-tuned model with adapter...")
trainer.save_model("./bitfit_tinyllama_sst2")
tokenizer.save_pretrained("./bitfit_tinyllama_sst2")

# Step 11: Save Training Data to JSON
print("💾 Saving training metrics to JSON...")
with open("training_metrics.json", "w") as f:
    json.dump(training_info, f, indent=4)

# Step 12: Final Evaluation
print("📊 Final evaluation...")
eval_results = trainer.evaluate()
print("\n=== FINAL RESULTS ===")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Step 13: Test Predictions
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

# Step 14: Test Examples
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

print("🎉 Training complete!")

import time
import torch
import psutil
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo

print("🚀 TinyLlama SST-2 LoRA Fine-tuning")
print("=" * 40)

# Step 3: Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-sst2-lora"
BATCH_SIZE = 16  # Can use larger batch size with LoRA
LEARNING_RATE = 3e-4  # Higher LR often works better with LoRA
EPOCHS = 3

print(f"📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Epochs: {EPOCHS}")

# Step 4: Load Dataset
print(f"\n📁 Loading GLUE SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
print(f"   Train samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['validation'])}")

# Step 5: Load Model and Tokenizer
print(f"\n🤖 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto"  # Automatically handle device placement
)

print(f"   ✓ Base model loaded")
print(f"   ✓ Model parameters: {model.num_parameters():,}")

# Step 6: Configure LoRA
print(f"\n🔧 Configuring LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.1,  # LoRA dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
    bias="none",  # Don't add bias to LoRA layers
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percentage = 100 * trainable_params / total_params

print(f"   ✓ LoRA applied successfully")
print(f"   📊 Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
print(f"   📊 Total parameters: {total_params:,}")

# Step 7: Tokenize Dataset
print(f"\n🔤 Tokenizing dataset...")

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

print(f"   ✓ Tokenization completed")

# Step 8: Data Collator and Metrics
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

print(f"   ✓ Metrics and data collator ready")

# Step 9: Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    report_to=None,  # Disable wandb
    dataloader_drop_last=False,
    bf16=True,  # Use bfloat16 instead of fp16 for better stability
    gradient_checkpointing=False,  # Disable gradient checkpointing for LoRA
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,  # Helps with LoRA training
)

print(f"   ✓ Training arguments configured")

# Step 10: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"   ✓ Trainer created")

# Step 11: Pre-training Evaluation
print(f"\n📊 Pre-training evaluation (zero-shot)...")
pre_eval = trainer.evaluate()
print(f"   Zero-shot accuracy: {pre_eval['eval_accuracy']:.4f} ({pre_eval['eval_accuracy']*100:.2f}%)")

# Step 12: Start Fine-tuning
print(f"\n🚀 Starting LoRA fine-tuning...")
print(f"   This will take approximately 10-20 minutes...")

start_time = time.time()
train_result = trainer.train()
training_time = time.time() - start_time

print(f"   ✅ Fine-tuning completed!")
print(f"   ⏱️  Training time: {training_time/60:.1f} minutes")
print(f"   📈 Final training loss: {train_result.training_loss:.4f}")

# Step 13: Post-training Evaluation
print(f"\n📊 Post-training evaluation...")
post_eval = trainer.evaluate()
print(f"   Fine-tuned accuracy: {post_eval['eval_accuracy']:.4f} ({post_eval['eval_accuracy']*100:.2f}%)")

# Calculate improvement
improvement = post_eval['eval_accuracy'] - pre_eval['eval_accuracy']
print(f"   📈 Improvement: +{improvement:.4f} ({improvement*100:.2f} percentage points)")

# Step 14: Save Model
print(f"\n💾 Saving fine-tuned model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Model saved to: {OUTPUT_DIR}")

# Step 15: Power and VRAM Consumption
# Initialize NVIDIA Management Library for power usage
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

# Get power and VRAM usage
def get_gpu_metrics():
    power = nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to Watts
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    vram_used = memory_info.used / (1024 ** 2)  # Convert to MB
    vram_total = memory_info.total / (1024 ** 2)  # Convert to MB
    return power, vram_used, vram_total

# Monitor during training
power, vram_used, vram_total = get_gpu_metrics()

# Step 16: Save Results to JSON
results = {
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "zero_shot_accuracy": pre_eval["eval_accuracy"],
    "fine_tuned_accuracy": post_eval["eval_accuracy"],
    "improvement": improvement,
    "training_time_minutes": training_time / 60,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "trainable_percentage": trainable_percentage,
    "gpu_power_watts": power,  # Power in Watts
    "gpu_vram_used_mb": vram_used,  # VRAM used in MB
    "gpu_vram_total_mb": vram_total,  # Total VRAM in MB
}

# Save the results to a JSON file
with open(f"{OUTPUT_DIR}/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"   ✓ Benchmark results saved to {OUTPUT_DIR}/benchmark_results.json")

# Step 17: Test Individual Predictions
print(f"\n🎯 Testing individual predictions...")

def test_prediction(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    confidence = predictions[0][predicted_class].item()
    return label, confidence

test_sentences = [
    "This movie is amazing!",
    "I hate this boring film.",
    "The acting was okay, nothing special.", 
    "Absolutely terrible movie, waste of time.",
    "Outstanding performance and great story!",
    "Boring and predictable plot."
]

print(f"\n--- Fine-tuned Model Predictions ---")
for sentence in test_sentences:
    label, confidence = test_prediction(sentence, model, tokenizer)
    print(f"'{sentence}'")
    print(f"→ {label} (confidence: {confidence:.3f})\n")

# Final Summary
print("=" * 50)
print("🏁 LORA FINE-TUNING SUMMARY")
print("=" * 50)
print(f"✅ Training completed successfully!")
print(f"📊 Results:")
print(f"   • Zero-shot accuracy: {pre_eval['eval_accuracy']*100:.2f}%")
print(f"   • Fine-tuned accuracy: {post_eval['eval_accuracy']*100:.2f}%") 
print(f"   • Improvement: +{improvement*100:.2f} percentage points")
print(f"   • Training time: {training_time/60:.1f} minutes")
print(f"   • Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
print(f"💾 Model saved to: {OUTPUT_DIR}")

print(f"\n🎉 LoRA fine-tuning complete!")

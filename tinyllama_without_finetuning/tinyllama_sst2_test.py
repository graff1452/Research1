# TinyLlama SST-2 Benchmark - Power, VRAM, and Timing Analysis

import torch
import psutil
import time
import os
import subprocess
import threading
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
import GPUtil

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.gpu_power_readings = []
        self.gpu_memory_readings = []
        self.gpu_utilization_readings = []
        self.cpu_readings = []
        self.ram_readings = []
        self.timestamps = []
        
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _get_gpu_power(self):
        """Get GPU power consumption using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=power.draw', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            power = float(result.stdout.strip())
            return power
        except:
            return None
    
    def _monitor_loop(self):
        while self.monitoring:
            timestamp = time.time()
            
            # GPU monitoring
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    self.gpu_memory_readings.append(gpu.memoryUsed)
                    self.gpu_utilization_readings.append(gpu.load * 100)
                    
                    # Get power consumption
                    power = self._get_gpu_power()
                    if power:
                        self.gpu_power_readings.append(power)
            except:
                pass
            
            # CPU and RAM monitoring
            self.cpu_readings.append(psutil.cpu_percent())
            self.ram_readings.append(psutil.virtual_memory().percent)
            self.timestamps.append(timestamp)
            
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
    def get_stats(self):
        stats = {}
        
        # GPU Stats
        if self.gpu_power_readings:
            stats['gpu_power_avg'] = np.mean(self.gpu_power_readings)
            stats['gpu_power_max'] = np.max(self.gpu_power_readings)
            stats['gpu_power_min'] = np.min(self.gpu_power_readings)
        
        if self.gpu_memory_readings:
            stats['gpu_memory_avg'] = np.mean(self.gpu_memory_readings)
            stats['gpu_memory_max'] = np.max(self.gpu_memory_readings)
        
        if self.gpu_utilization_readings:
            stats['gpu_util_avg'] = np.mean(self.gpu_utilization_readings)
            stats['gpu_util_max'] = np.max(self.gpu_utilization_readings)
        
        # CPU/RAM Stats
        if self.cpu_readings:
            stats['cpu_avg'] = np.mean(self.cpu_readings)
            stats['cpu_max'] = np.max(self.cpu_readings)
        
        if self.ram_readings:
            stats['ram_avg'] = np.mean(self.ram_readings)
            stats['ram_max'] = np.max(self.ram_readings)
        
        return stats

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return None

def benchmark_tinyllama_sst2():
    print("=== TinyLlama SST-2 Benchmark ===")
    print("Testing: Power consumption, VRAM usage, and timings\n")
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    # Timing dictionary
    timings = {}
    
    # === STEP 1: Initial Setup ===
    print("📊 Starting system monitoring...")
    monitor.start_monitoring()
    
    print("🔧 Initial GPU memory:")
    initial_gpu_mem = get_gpu_memory()
    if initial_gpu_mem:
        print(f"   Allocated: {initial_gpu_mem['allocated']:.2f} GB")
        print(f"   Reserved: {initial_gpu_mem['reserved']:.2f} GB")
    
    # === STEP 2: Load Dataset ===
    print("\n📁 Loading GLUE SST-2 dataset...")
    start_time = time.time()
    dataset = load_dataset("glue", "sst2")
    timings['dataset_loading'] = time.time() - start_time
    print(f"   ✓ Dataset loaded in {timings['dataset_loading']:.2f}s")
    
    # === STEP 3: Load Model ===
    print("\n🤖 Loading TinyLlama model...")
    start_time = time.time()
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        pad_token_id=tokenizer.eos_token_id
    )
    
    timings['model_loading'] = time.time() - start_time
    print(f"   ✓ Model loaded in {timings['model_loading']:.2f}s")
    
    # Check GPU memory after model loading
    print("\n💾 GPU memory after model loading:")
    model_gpu_mem = get_gpu_memory()
    if model_gpu_mem:
        print(f"   Allocated: {model_gpu_mem['allocated']:.2f} GB")
        print(f"   Reserved: {model_gpu_mem['reserved']:.2f} GB")
        print(f"   Model size: ~{model_gpu_mem['allocated'] - initial_gpu_mem['allocated']:.2f} GB")
    
    # === STEP 4: Tokenize Dataset ===
    print("\n🔤 Tokenizing dataset...")
    start_time = time.time()
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=False, max_length=512)
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["sentence", "idx"]
    )
    
    timings['tokenization'] = time.time() - start_time
    print(f"   ✓ Tokenization completed in {timings['tokenization']:.2f}s")
    
    # === STEP 5: Setup Training Components ===
    print("\n⚙️  Setting up training components...")
    start_time = time.time()
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_eval_batch_size=8,
        dataloader_drop_last=False,
        report_to=None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    timings['setup'] = time.time() - start_time
    print(f"   ✓ Setup completed in {timings['setup']:.2f}s")
    
    # === STEP 6: Run Evaluation ===
    print("\n🧪 Running evaluation on validation set...")
    print("   (This will measure peak power and memory usage)")
    
    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    eval_results = trainer.evaluate()
    timings['evaluation'] = time.time() - start_time
    
    print(f"   ✓ Evaluation completed in {timings['evaluation']:.2f}s")
    print(f"   ✓ Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"   ✓ Throughput: {eval_results.get('eval_samples_per_second', 'N/A'):.1f} samples/sec")
    
    # === STEP 7: Individual Predictions Benchmark ===
    print("\n🎯 Benchmarking individual predictions...")
    
    test_sentences = [
        "This movie is amazing!",
        "I hate this boring film.",
        "The acting was okay, nothing special.",
        "Absolutely terrible movie, waste of time.",
        "Great acting and storyline!",
        "Boring and predictable plot.",
        "Outstanding cinematography!",
        "Terrible waste of two hours.",
        "Brilliant performance by the lead actor.",
        "Could not wait for it to end."
    ]
    
    device = next(model.parameters()).device
    prediction_times = []
    
    for sentence in test_sentences:
        start_time = time.time()
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        prediction_time = time.time() - start_time
        prediction_times.append(prediction_time)
    
    timings['avg_prediction'] = np.mean(prediction_times)
    timings['total_predictions'] = sum(prediction_times)
    
    print(f"   ✓ Average prediction time: {timings['avg_prediction']*1000:.2f} ms")
    print(f"   ✓ Total time for 10 predictions: {timings['total_predictions']:.3f}s")
    
    # === STEP 8: Final Memory Check ===
    print("\n💾 Final GPU memory usage:")
    final_gpu_mem = get_gpu_memory()
    if final_gpu_mem:
        print(f"   Peak allocated: {final_gpu_mem['max_allocated']:.2f} GB")
        print(f"   Current allocated: {final_gpu_mem['allocated']:.2f} GB")
        print(f"   Current reserved: {final_gpu_mem['reserved']:.2f} GB")
    
    # === STEP 9: Stop Monitoring and Get Stats ===
    print("\n📊 Stopping system monitoring...")
    monitor.stop_monitoring()
    stats = monitor.get_stats()
    
    # === STEP 10: Final Report ===
    print("\n" + "="*50)
    print("📈 BENCHMARK RESULTS")
    print("="*50)
    
    # Timing Results
    print("\n⏱️  TIMING RESULTS:")
    print(f"   Dataset Loading:     {timings['dataset_loading']:.2f}s")
    print(f"   Model Loading:       {timings['model_loading']:.2f}s") 
    print(f"   Tokenization:        {timings['tokenization']:.2f}s")
    print(f"   Setup:               {timings['setup']:.2f}s")
    print(f"   Evaluation:          {timings['evaluation']:.2f}s")
    print(f"   Avg Prediction:      {timings['avg_prediction']*1000:.2f}ms")
    print(f"   Total Runtime:       {sum(timings.values()):.2f}s")
    
    # GPU Results
    if final_gpu_mem:
        print("\n🖥️  GPU MEMORY RESULTS:")
        print(f"   Model Size:          ~{model_gpu_mem['allocated'] - initial_gpu_mem['allocated']:.2f} GB")
        print(f"   Peak Memory Usage:   {final_gpu_mem['max_allocated']:.2f} GB")
        print(f"   Current Usage:       {final_gpu_mem['allocated']:.2f} GB")
        print(f"   Reserved Memory:     {final_gpu_mem['reserved']:.2f} GB")
    
    # Power Results
    if 'gpu_power_avg' in stats:
        print("\n⚡ POWER CONSUMPTION:")
        print(f"   Average GPU Power:   {stats['gpu_power_avg']:.1f} W")
        print(f"   Peak GPU Power:      {stats['gpu_power_max']:.1f} W")
        print(f"   Min GPU Power:       {stats['gpu_power_min']:.1f} W")
    
    # System Resource Results  
    if 'gpu_util_avg' in stats:
        print("\n📊 SYSTEM UTILIZATION:")
        print(f"   Avg GPU Utilization: {stats['gpu_util_avg']:.1f}%")
        print(f"   Peak GPU Util:       {stats['gpu_util_max']:.1f}%")
    
    if 'cpu_avg' in stats:
        print(f"   Avg CPU Usage:       {stats['cpu_avg']:.1f}%")
        print(f"   Peak CPU Usage:      {stats['cpu_max']:.1f}%")
    
    if 'ram_avg' in stats:
        print(f"   Avg RAM Usage:       {stats['ram_avg']:.1f}%")
        print(f"   Peak RAM Usage:      {stats['ram_max']:.1f}%")
    
    # Performance Results
    print("\n🎯 PERFORMANCE RESULTS:")
    print(f"   Accuracy:            {eval_results.get('eval_accuracy', 0):.4f} ({eval_results.get('eval_accuracy', 0)*100:.2f}%)")
    print(f"   Throughput:          {eval_results.get('eval_samples_per_second', 0):.1f} samples/sec")
    print(f"   Inference Speed:     {1000/timings['avg_prediction']:.1f} predictions/sec")
    
    print("\n" + "="*50)
    print("✅ Benchmark completed!")

if __name__ == "__main__":
    # Install required packages message
    print("Required packages: pip install GPUtil psutil")
    print("Make sure nvidia-smi is available for power monitoring\n")
    
    benchmark_tinyllama_sst2()
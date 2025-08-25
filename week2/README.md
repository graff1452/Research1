# TinyLlama + LoRA Fine-Tuning on GLUE SST-2

This project fine-tunes TinyLlama-1.1B-Chat-v1.0 on the GLUE SST-2 sentiment classification task using LoRA (PEFT) and benchmarks the resulting model for accuracy, speed, memory, and energy usage.

## 📂 Project Structure
```
week2/
├── train_tinyllama_lora_sst2.py     # Training script
├── bench_sst2_tinyllama_lora.py     # Benchmarking script
├── tinyllama-sst2-lora/             # Training outputs
│   └── lora_adapter/                # Saved LoRA weights
├── bench_reports/                   # Benchmark JSON + CSV
├── requirements.txt                 # Dependencies
└── README.md
```

## 🚀 Setup
1. Create and activate a virtual environment:
```
   python -m venv .venv
   source .venv/bin/activate
```

2. Install dependencies:
```
   pip install -r requirements.txt
```
   or capture your own:
```
   pip freeze > requirements.txt
```
## 🏋️ Training
Fine-tune TinyLlama on SST-2 with LoRA:
```
   python train_tinyllama_lora_sst2.py
```
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  
- Dataset: GLUE SST-2  
- LoRA saved at: ./tinyllama-sst2-lora/lora_adapter/  

Training logs include loss, eval accuracy, grad norms, throughput.

## 📊 Benchmarking
Once training finishes and the LoRA adapter is saved:
```
   python bench_sst2_tinyllama_lora.py \
     --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --adapter_dir ./tinyllama-sst2-lora/lora_adapter \
     --batch_size 32 \
     --max_length 256 \
     --amp
```
Outputs:
- Console report  
- bench_reports/sst2_benchmark.json  
- bench_reports/sst2_benchmark.csv  

Metrics include accuracy, latency & throughput, tokens/sec, VRAM peak, GPU avg/peak power, energy (Wh / J).

## ✅ Example Benchmark Result (validation set)
```
{
  "accuracy": 0.9564,
  "examples_per_sec": 436.9,
  "latency_ms_per_example": 2.29,
  "tokens_per_sec": 12276.9,
  "vram_peak_GB": 4.20,
  "avg_power_W": 200.0,
  "peak_power_W": 259.8,
  "energy_J": 380.1
}
```
## 🔧 Notes & Tips
- Pad token: set to </s> for TinyLlama (pad_token_id=2).  
- Quantization: Works with bitsandbytes (load_in_4bit / load_in_8bit).  
- Power/Energy: Requires pynvml and an NVIDIA GPU.  
- Transformers v5 arg change: use eval_strategy instead of evaluation_strategy.  

## 📌 Next Steps
- Sweep batch sizes for scaling curves (latency vs throughput vs VRAM).  
- Try load_in_8bit / load_in_4bit for lighter inference.  
- Merge LoRA with base model for standalone checkpoint.  
- Extend benchmarks to other GLUE tasks.  

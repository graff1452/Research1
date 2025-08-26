# TinyLlama LoRA Fine-Tuning on GLUE SST-2

This project demonstrates how to fine-tune [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using [LoRA](https://github.com/microsoft/LoRA) for the **GLUE SST-2** sentiment classification task.  

### Features
- Efficient LoRA fine-tuning to reduce memory and compute cost  
- Training on the SST-2 dataset from GLUE  
- Saving both the **full model** and a compact **LoRA adapter**  
- Benchmarking on SST-2 validation set with metrics:
  - Accuracy
  - Runtime
  - VRAM usage
  - Power and energy consumption  
- Exporting results as a structured **JSON report**  
- Preparing `SST-2.tsv` for official GLUE leaderboard submission  

---

## ⚙️ Environment Setup

Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies from requirements.txt:
```
pip install -r requirements.txt
```

🚀 Training

Run the training script to fine-tune TinyLlama with LoRA on SST-2:
```
python train_sst2_lora_tinyllama.py
```

This will:

- Train LoRA adapters on SST-2

- Save checkpoints in out_sst2_tinyllama_lora/

- Export both the full model and lora_sst2_adapter.pt

📊 Benchmarking

Once training is complete, evaluate your model with:
```
python benchmark_sst2_metrics.py --model_path out_sst2_tinyllama_lora --out metrics.json
```

This will:

- Compute accuracy on the SST-2 validation split

- Record runtime, VRAM, and (if supported) GPU power/energy metrics

- Save results into metrics.json for easy reproducibility

📝 GLUE Submission

To generate predictions for the hidden test split and prepare the official GLUE submission file:
```
python make_glue_submission_sst2.py --model_path out_sst2_tinyllama_lora
```

This produces SST-2.tsv with predictions (index, prediction), which you can zip and upload to gluebenchmark.com.

📂 Project Structure
```
week2_lora_microsoft/
├── train_sst2_lora_tinyllama.py     # training script
├── benchmark_sst2_metrics.py        # benchmarking script
├── make_glue_submission_sst2.py     # helper for GLUE test submission
├── requirements.txt                 # dependencies
├── README.md                        # this file
└── out_sst2_tinyllama_lora/         # model outputs & adapters
```
#!/usr/bin/env python3
# benchmark_sst2_metrics.py
# Example:
#   python benchmark_sst2_metrics.py --model_path out_sst2_tinyllama_lora --out metrics.json

import argparse, json, os, sys, time, threading
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding
import evaluate

# ---------- Power sampling (NVIDIA) ----------
class NvmlPowerSampler:
    """
    Samples GPU power via NVML in a background thread and integrates energy.
    Works on NVIDIA GPUs. If NVML is unavailable, acts as a no-op.
    """
    def __init__(self, device_index: int = 0, interval_s: float = 0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples_W: List[float] = []
        self.timestamps: List[float] = []
        self._stop = threading.Event()
        self._thread = None
        self._enabled = False
        # NVML bind
        try:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._enabled = True
        except Exception:
            self.nvml = None
            self._enabled = False

    def _loop(self):
        while not self._stop.is_set():
            try:
                p_mW = self.nvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
                now = time.time()
                self.samples_W.append(p_mW / 1000.0)
                self.timestamps.append(now)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        if not self._enabled: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._enabled: return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summary(self) -> Dict[str, Any]:
        if not self._enabled or len(self.samples_W) < 2:
            return {
                "nvml_available": False,
                "mean_power_W": None,
                "peak_power_W": None,
                "energy_Wh": None,
                "energy_kWh": None,
                "samples": len(self.samples_W),
            }
        W = np.array(self.samples_W, dtype=np.float64)
        t = np.array(self.timestamps, dtype=np.float64)
        dt = np.diff(t)                       # seconds between samples
        W_mid = (W[:-1] + W[1:]) / 2.0        # trapezoid midpoint power
        energy_Wh = float(np.sum(W_mid * dt) / 3600.0)  # Wh
        return {
            "nvml_available": True,
            "mean_power_W": float(np.mean(W)),
            "peak_power_W": float(np.max(W)),
            "energy_Wh": energy_Wh,
            "energy_kWh": energy_Wh / 1000.0,
            "samples": int(len(W)),
            "sample_interval_s": self.interval_s,
        }

# ---------- Utilities ----------
def pick_device(arg: str) -> str:
    if arg == "cuda" or (arg == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"

def load_tok(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def to_dataloader(split, tok, max_len: int, batch_size: int, shuffle: bool = False):
    def _prep(ex):
        out = tok(ex["sentence"], truncation=True, max_length=max_len)
        out["idx"] = ex["idx"]
        if "label" in ex:
            out["labels"] = ex["label"]
        return out
    ds = split.map(_prep, batched=True, remove_columns=split.column_names)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

@torch.no_grad()
def eval_accuracy(model, dataloader, device: str):
    preds, labels = [], []
    for batch in dataloader:
        idx = batch.pop("idx")  # unused for metric
        lb = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        preds.append(pred)
        labels.append(lb.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metric = evaluate.load("glue", "sst2")  # returns accuracy
    return metric.compute(predictions=preds, references=labels)["accuracy"]

def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024.0**2)

def main():
    parser = argparse.ArgumentParser(description="Benchmark TinyLlama LoRA on GLUE/SST-2 (dev) with power/VRAM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", help="'auto'|'cuda'|'cpu'")
    parser.add_argument("--out", type=str, default="metrics.json", help="Output JSON file")
    parser.add_argument("--dtype", type=str, default="auto", help="'auto'|'bf16'|'fp16'|'fp32'")
    parser.add_argument("--power_interval", type=float, default=0.2, help="Seconds between power samples")
    args = parser.parse_args()

    device = pick_device(args.device)

    # Data
    glue = load_dataset("nyu-mll/glue", "sst2")
    tok = load_tok(args.model_path)
    val_ds, val_loader = to_dataloader(glue["validation"], tok, args.max_len, args.batch_size, shuffle=False)

    # Dtype selection
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32" or device == "cpu":
        torch_dtype = torch.float32
    else:
        # auto
        if device == "cuda":
            major = torch.cuda.get_device_capability(0)[0]
            torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            torch_dtype = torch.float32

    # Model
    model = LlamaForSequenceClassification.from_pretrained(args.model_path, torch_dtype=torch_dtype).eval()
    model.to(device)

    # Reset VRAM peak counters
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Start power sampling
    sampler = NvmlPowerSampler(device_index=0, interval_s=args.power_interval)
    sampler.start()
    t0 = time.time()

    # Run evaluation
    acc = eval_accuracy(model, val_loader, device)

    # Stop sampling and sync
    t1 = time.time()
    sampler.stop()
    if device == "cuda":
        torch.cuda.synchronize()

    # Gather VRAM stats
    vram_peak_alloc = None
    vram_peak_reserved = None
    if device == "cuda":
        vram_peak_alloc = int(torch.cuda.max_memory_allocated())
        vram_peak_reserved = int(torch.cuda.max_memory_reserved())

    # Power/Energy summary
    power = sampler.summary()

    # System/GPU info
    sysinfo = {
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cuda":
        sysinfo.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cuda_device_count": torch.cuda.device_count(),
        })

    results = {
        "task": "GLUE/SST-2 (dev)",
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "runtime_seconds": t1 - t0,
        "accuracy": acc,
        "vram": {
            "peak_alloc_bytes": vram_peak_alloc,
            "peak_alloc_MiB": bytes_to_mib(vram_peak_alloc) if vram_peak_alloc is not None else None,
            "peak_reserved_bytes": vram_peak_reserved,
            "peak_reserved_MiB": bytes_to_mib(vram_peak_reserved) if vram_peak_reserved is not None else None,
        },
        "power": power,     # contains mean/peak W, energy Wh/kWh if NVML available
        "system": sysinfo,
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote metrics to: {args.out}")

if __name__ == "__main__":
    # Lazy import to avoid overhead at top-level for datasets
    from datasets import load_dataset  # noqa: E402
    main()

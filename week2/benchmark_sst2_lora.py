import argparse, json, os, time, csv, threading
from contextlib import nullcontext
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import evaluate
import numpy as np

# -------- Power sampling with NVML (GPU only) --------
class PowerSampler:
    def __init__(self, dev_index=0, interval=0.1):
        self.dev_index = dev_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples = []  # (timestamp, watts)

        self.nvml_ok = False
        try:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(dev_index)
            self.nvml_ok = True
        except Exception as e:
            print("[PowerSampler] NVML not available:", e)

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
                self.samples.append((now, mw / 1000.0))
            except Exception:
                # if sampling fails, just skip
                pass
            # sleep to next interval
            time.sleep(max(0.0, self.interval - (time.perf_counter() - now)))

    def start(self):
        if not self.nvml_ok:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.nvml_ok:
            return
        self._stop.set()
        self._thread.join()
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass

    def stats(self):
        """
        Returns (avg_power_W, peak_power_W, energy_Wh, energy_J)
        energy integrates trapezoidally over the sampling window.
        """
        if not self.samples:
            return None, None, 0.0, 0.0
        # trapezoidal integration
        energy_J = 0.0
        peak_W = 0.0
        for i in range(1, len(self.samples)):
            t0, w0 = self.samples[i-1]
            t1, w1 = self.samples[i]
            dt = (t1 - t0)  # seconds
            energy_J += 0.5 * (w0 + w1) * dt  # W*s = J
            peak_W = max(peak_W, w0, w1)
        energy_Wh = energy_J / 3600.0
        avg_W = energy_J / (self.samples[-1][0] - self.samples[0][0]) if len(self.samples) > 1 else self.samples[0][1]
        return avg_W, peak_W, energy_Wh, energy_J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter_dir", default="./tinyllama-sst2-lora/lora_adapter")
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Enable autocast fp16/bf16 on GPU")
    ap.add_argument("--dev_index", type=int, default=0, help="GPU index for power sampling")
    ap.add_argument("--power_interval", type=float, default=0.1, help="Seconds between power samples")
    ap.add_argument("--report_prefix", default="sst2_benchmark")
    args = ap.parse_args()

    # -------- Data --------
    ds = load_dataset("glue", "sst2")[args.split]
    metric = evaluate.load("glue", "sst2")

    # -------- Tokenizer --------
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    def encode(batch):
        out = tok(batch["sentence"], truncation=True, max_length=args.max_length)
        out["labels"] = batch["label"]
        return out

    cols_keep = ["input_ids", "attention_mask", "labels"]
    ds_enc = ds.map(encode, batched=True, remove_columns=[c for c in ds.column_names if c not in cols_keep])

    # -------- Model (+ LoRA) --------
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    model.to(args.device)
    model.config.pad_token_id = tok.pad_token_id

    # -------- Dataloader --------
    def collate(features):
        batch = tok.pad(
            features,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return batch

    from torch.utils.data import DataLoader
    loader = DataLoader(ds_enc, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=(args.device=="cuda"))

    # -------- VRAM accounting --------
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.dev_index)

    # -------- Warmup (small number of batches to stabilize timings) --------
    warmup_batches = min(5, len(loader))
    warmup_idx = 0
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids", "attention_mask", "labels"]:
                batch[k] = batch[k].to(args.device, non_blocking=True)
            with (torch.autocast(device_type="cuda", dtype=torch.float16) if (args.amp and args.device=="cuda") else nullcontext()):
                _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            warmup_idx += 1
            if warmup_idx >= warmup_batches:
                break
    if args.device == "cuda":
        torch.cuda.synchronize(args.dev_index)

    # -------- Time + power measurement --------
    power = PowerSampler(dev_index=args.dev_index, interval=args.power_interval)
    power.start()

    t0 = time.perf_counter()
    n_tokens = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            attn = batch["attention_mask"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            n_tokens += attn.sum().item()

            with (torch.autocast(device_type="cuda", dtype=torch.float16) if (args.amp and args.device=="cuda") else nullcontext()):
                logits = model(input_ids=input_ids, attention_mask=attn).logits

            preds = logits.argmax(dim=-1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    if args.device == "cuda":
        torch.cuda.synchronize(args.dev_index)
    t1 = time.perf_counter()
    power.stop()

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    results = metric.compute(predictions=preds_all, references=labels_all)
    total_time_s = t1 - t0
    n_examples = len(ds_enc)
    tokens_per_sec = n_tokens / total_time_s
    ex_per_sec = n_examples / total_time_s
    latency_ms_per_example = 1000.0 / ex_per_sec

    # VRAM
    vram_bytes = None
    if args.device == "cuda":
        vram_bytes = torch.cuda.max_memory_allocated(args.dev_index)

    avg_W, peak_W, energy_Wh, energy_J = power.stats()

    # -------- Report --------
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "amp": bool(args.amp),
        "num_examples": int(n_examples),
        "total_tokens": int(n_tokens),
        "accuracy": float(results.get("accuracy", float("nan"))),
        "total_time_s": float(total_time_s),
        "examples_per_sec": float(ex_per_sec),
        "latency_ms_per_example": float(latency_ms_per_example),
        "tokens_per_sec": float(tokens_per_sec),
        "vram_peak_bytes": int(vram_bytes) if vram_bytes is not None else None,
        "vram_peak_GB": (float(vram_bytes) / (1024**3)) if vram_bytes is not None else None,
        "avg_power_W": float(avg_W) if avg_W is not None else None,
        "peak_power_W": float(peak_W) if peak_W is not None else None,
        "energy_Wh": float(energy_Wh),
        "energy_J": float(energy_J),
        "power_interval_s": args.power_interval,
    }

    prefix = args.report_prefix
    os.makedirs("bench_reports", exist_ok=True)
    json_path = os.path.join("bench_reports", f"{prefix}.json")
    csv_path = os.path.join("bench_reports", f"{prefix}.csv")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # also a flat CSV row for quick comparisons
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)

    # pretty print
    print("\n=== SST-2 Benchmark Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()

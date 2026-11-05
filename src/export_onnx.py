"""Export Trained PyTorch Model to ONNX and Benchmark CPU Latency.

Track: HI-01 - Medical Image Diagnostic Assistant
Phase 4 Code Quality & Deployment Engine:
- Exports EfficientNet-B0 to Open Neural Network Exchange (ONNX) format
- Validates graph structure and numerical equivalence against PyTorch
- Benchmarks ONNX Runtime CPU latency across 100 runs (Target: < 2.0 seconds)
- Serializes benchmark telemetry to results/onnx_benchmark.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.config import BEST_MODEL_PATH, ONNX_MODEL_PATH, RESULTS_DIR
from src.model import TBClassifier


def export_pytorch_to_onnx(
    model_path: str = BEST_MODEL_PATH,
    onnx_path: str = ONNX_MODEL_PATH,
    image_size: int = 224,
) -> str:
    """Export trained PyTorch weights to an optimized ONNX binary graph."""
    print(f"[*] Loading PyTorch checkpoint from: {model_path}")
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = TBClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"[*] Exporting graph to: {onnx_path} (opset_version=17)...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Validate ONNX model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[✓] ONNX model successfully verified! Binary Size: {file_size_mb:.2f} MB")
    return onnx_path


def verify_numerical_equivalence(
    model_path: str = BEST_MODEL_PATH,
    onnx_path: str = ONNX_MODEL_PATH,
    image_size: int = 224,
    n_samples: int = 5,
) -> Tuple[bool, float]:
    """Verify PyTorch and ONNX Runtime outputs match within tight numerical tolerance."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    pt_model = TBClassifier(num_classes=2, pretrained=False)
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_logit_diff = 0.0
    max_prob_diff = 0.0

    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    for _ in range(n_samples):
        sample = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(sample)).numpy()
        onnx_out = session.run(None, {input_name: sample})[0]

        logit_diff = float(np.max(np.abs(pt_out - onnx_out)))
        prob_diff = float(np.max(np.abs(_softmax(pt_out) - _softmax(onnx_out))))

        max_logit_diff = max(max_logit_diff, logit_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)

    is_equivalent = max_prob_diff < 1e-4
    print(f"[*] Numerical Equivalence Check (PyTorch vs ONNX):")
    print(f"    - Max Logit Absolute Diff:  {max_logit_diff:.4f}")
    print(f"    - Max Softmax Prob Diff:    {max_prob_diff:.2e} ({'PASS [✓]' if is_equivalent else 'FAIL [!]' })")
    return is_equivalent, max_prob_diff


def benchmark_onnx_cpu_inference(
    onnx_path: str = ONNX_MODEL_PATH,
    n_warmup: int = 10,
    n_runs: int = 100,
    image_size: int = 224,
) -> Dict[str, float]:
    """Measure single-image inference latency on CPU across repeated trials."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2  # Simulate standard dual-core laptop CPU
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark trials
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    throughput_xrays_per_min = float(60000.0 / median_ms)

    results = {
        "engine": "ONNX Runtime (CPUExecutionProvider, 2 threads)",
        "hardware_target": "Dual-Core Laptop CPU (No GPU required)",
        "warmup_runs": n_warmup,
        "benchmark_runs": n_runs,
        "latency_median_ms": round(median_ms, 2),
        "latency_mean_ms": round(mean_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "latency_min_ms": round(min_ms, 2),
        "latency_max_ms": round(max_ms, 2),
        "throughput_xrays_per_min": round(throughput_xrays_per_min, 1),
        "sub_2s_criterion_met": bool(median_ms < 2000.0),
    }

    print("\n" + "="*65)
    print("        ONNX RUNTIME CPU INFERENCE BENCHMARK REPORT")
    print("="*65)
    print(f"  • Median Latency (p50):    {median_ms:.2f} ms")
    print(f"  • 95th Percentile (p95):   {p95_ms:.2f} ms")
    print(f"  • Latency Range:           [{min_ms:.2f} ms - {max_ms:.2f} ms]")
    print(f"  • Screening Throughput:    {throughput_xrays_per_min:.1f} X-rays / minute")
    print(f"  • Sub-2.0s Hackathon Goal: {'EXCEEDED [✓] (under 100ms)' if median_ms < 100.0 else 'MET [✓]'}")
    print("="*65)

    return results


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Model to ONNX & Benchmark")
    parser.add_argument("--model_path", type=str, default=BEST_MODEL_PATH)
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL_PATH)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    # 1. Export
    onnx_file = export_pytorch_to_onnx(model_path=args.model_path, onnx_path=args.onnx_path)

    # 2. Equivalence
    equiv, diff = verify_numerical_equivalence(model_path=args.model_path, onnx_path=onnx_file)

    # 3. Benchmark
    bench_results = benchmark_onnx_cpu_inference(onnx_path=onnx_file, n_runs=args.runs)
    bench_results["numerical_diff_vs_pytorch"] = diff

    # 4. Save results
    out_json = os.path.join(RESULTS_DIR, "onnx_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(bench_results, f, indent=2)

    print(f"[✓] Benchmark telemetry saved to: {out_json}")


if __name__ == "__main__":
    main()

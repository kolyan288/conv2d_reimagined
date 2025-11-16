import csv
import os
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import torch
from torch.profiler import profile, record_function


class LatencyMetricsWriter:
    def __init__(self, csv_file: str = "latency_metrics.csv"):
        self.csv_file = csv_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "model_name",
                        "batch_size",
                        "precision",
                        "input_shape",
                        "latency_cpu_ms",
                        "latency_gpu_mean_ms",
                        "latency_gpu_std_ms",
                        "total_params",
                        "val_iou",
                        "test_iou",
                        "device",
                        "notes",
                        "sparsity"
                    ]
                )

    def extract_cpu_latency_from_profiler(self, prof) -> float:
        """Extract CPU latency in milliseconds from profiler output"""
        # Using self_cpu_time_total as you mentioned
        cpu_time_ns = prof.key_averages(group_by_input_shape=True).self_cpu_time_total
        return cpu_time_ns / 1000000  # Convert nanoseconds to milliseconds

    def record_metrics(
        self,
        model,
        model_name: str,
        batch_size: int,
        precision: str,
        input_shape: tuple,
        cpu_profiler: Optional[profile] = None,
        gpu_latency: Optional[Tuple[float, float]] = None,
        total_params: Optional[int] = None,
        notes: str = "",
        val_iou: Optional[float] = None,
        test_iou: Optional[float] = None,
        sparsity : Optional[Dict[str, Any]] = None,
    ):
        """Record latency metrics to CSV"""

        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())

        # Extract CPU latency from profiler
        latency_cpu_ms = (
            self.extract_cpu_latency_from_profiler(cpu_profiler)
            if cpu_profiler
            else None
        )

        # Extract GPU latency
        latency_gpu_mean_ms, latency_gpu_std_ms = (
            gpu_latency if gpu_latency else (None, None)
        )

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "batch_size": batch_size,
            "precision": precision,
            "input_shape": str(input_shape),
            "latency_cpu_ms": latency_cpu_ms,
            "latency_gpu_mean_ms": latency_gpu_mean_ms,
            "latency_gpu_std_ms": latency_gpu_std_ms,
            "total_params": total_params,
            "val_iou": val_iou,
            "test_iou": test_iou,
            "device": "cuda",
            "notes": notes,
            "sparsity": str(sparsity) if sparsity else "",
        }

        # Append to CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)

        print(f"Metrics recorded for {model_name} (batch_size={batch_size})")
        return metrics

    def read_metrics(self) -> pd.DataFrame:
        """Read all metrics as pandas DataFrame"""
        return pd.read_csv(self.csv_file)

    def get_metrics_by_model(self, model_name: str) -> pd.DataFrame:
        """Get metrics for a specific model"""
        df = self.read_metrics()
        return df[df["model_name"] == model_name]

    def get_latest_metrics(self, n: int = 10) -> pd.DataFrame:
        """Get the most recent n metrics"""
        df = self.read_metrics()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp", ascending=False).head(n)


# Usage example that matches your workflow:
# def benchmark_workflow_example():
#     """Example showing how to use with your existing profiling workflow"""
#     writer = LatencyMetricsWriter("model_latency_metrics.csv")

#     # Your existing models and inputs
#     model_p1 = YourModel()  # Original model
#     model_p2 = YourModel()  # Converted model
#     input_tensor = torch.randn(2, 3, 320, 320)

#     # Your existing CPU profiling
#     print("Profiling original model...")
#     lcpu_p1 = latency_cpu_profiler(model_p1, input_tensor, warmup_n=10, benchmark_n=30)

#     print("Profiling converted model...")
#     lcpu_p2 = latency_cpu_profiler(model_p2, input_tensor, warmup_n=10, benchmark_n=30)

#     # Print profiler tables (your existing code)
#     print("--- Original Model ---")
#     print(
#         lcpu_p1.key_averages(group_by_input_shape=True).table(
#             sort_by="cpu_time_total", row_limit=5
#         )
#     )
#     print("\n--- After convert_fx ---")
#     print(
#         lcpu_p2.key_averages(group_by_input_shape=True).table(
#             sort_by="cpu_time_total", row_limit=5
#         )
#     )

#     # Your existing GPU benchmarking
#     print("\n--- GPU Benchmarking ---")
#     gpu_mean_p1, gpu_std_p1 = latency_gpu(model_p1, input_tensor, warmup_n=10, benchmark_n=100)
#     gpu_mean_p2, gpu_std_p2 = latency_gpu(model_p2, input_tensor, warmup_n=10, benchmark_n=100)

#     # Record metrics for both models
#     metrics_p1 = writer.record_metrics(
#         model=model_p1,
#         model_name="OriginalModel",
#         batch_size=input_tensor.shape[0],
#         precision="fp32",
#         input_shape=input_tensor.shape,
#         cpu_profiler=lcpu_p1,
#         gpu_latency=(gpu_mean_p1, gpu_std_p1),
#         notes="Original model before conversion"
#     )

#     metrics_p2 = writer.record_metrics(
#         model=model_p2,
#         model_name="ConvertedModel",
#         batch_size=input_tensor.shape[0],
#         precision="fp32",
#         input_shape=input_tensor.shape,
#         cpu_profiler=lcpu_p2,
#         gpu_latency=(gpu_mean_p2, gpu_std_p2),
#         notes="After convert_fx conversion"
#     )

#     # Print summary
#     print(f"\n=== Summary ===")
#     print(f"Original Model - CPU: {metrics_p1['latency_cpu_ms']:.3f}ms, GPU: {metrics_p1['latency_gpu_mean_ms']:.3f}ms")
#     print(f"Converted Model - CPU: {metrics_p2['latency_cpu_ms']:.3f}ms, GPU: {metrics_p2['latency_gpu_mean_ms']:.3f}ms")

#     return writer

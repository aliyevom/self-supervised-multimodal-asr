"""Metrics tracking and computation for ASR evaluation.

This module provides utilities for computing various ASR metrics including WER,
CER, latency measurements, and memory profiling.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import psutil
import torch.cuda as cuda
from jiwer import wer, cer
from collections import defaultdict

@dataclass
class LatencyMetrics:
    mean_ms: float
    std_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

@dataclass
class MemoryMetrics:
    cpu_mb: float
    gpu_allocated_mb: Optional[float] = None
    gpu_cached_mb: Optional[float] = None

class MetricsTracker:
    """Tracks and aggregates multiple metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.current_batch = {}
        
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            self.current_batch[key] = value
            
    def get_current(self) -> Dict[str, float]:
        """Get metrics for current batch."""
        return self.current_batch
        
    def get_average(self) -> Dict[str, float]:
        """Get averaged metrics over all updates."""
        return {
            key: np.mean(values) for key, values in self.metrics.items()
        }

def compute_wer(
    predictions: List[str],
    references: List[str],
    remove_punctuation: bool = True
) -> float:
    """Compute Word Error Rate."""
    if remove_punctuation:
        # Remove punctuation and normalize spacing
        predictions = [p.lower().strip() for p in predictions]
        references = [r.lower().strip() for r in references]
    
    return wer(references, predictions)

def compute_cer(
    predictions: List[str],
    references: List[str]
) -> float:
    """Compute Character Error Rate."""
    return cer(references, predictions)

def measure_latency(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> LatencyMetrics:
    """Measure model inference latency."""
    model.eval()
    latencies = []
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(**sample_input)
    
    # Measurement runs
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(**sample_input)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start_time)
    
    latencies_ms = np.array(latencies) * 1000
    return LatencyMetrics(
        mean_ms=np.mean(latencies_ms),
        std_ms=np.std(latencies_ms),
        p90_ms=np.percentile(latencies_ms, 90),
        p95_ms=np.percentile(latencies_ms, 95),
        p99_ms=np.percentile(latencies_ms, 99)
    )

def measure_memory() -> MemoryMetrics:
    """Measure CPU and GPU memory usage."""
    process = psutil.Process()
    cpu_mb = process.memory_info().rss / (1024 * 1024)
    
    if torch.cuda.is_available():
        return MemoryMetrics(
            cpu_mb=cpu_mb,
            gpu_allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
            gpu_cached_mb=torch.cuda.memory_reserved() / (1024 * 1024)
        )
    
    return MemoryMetrics(cpu_mb=cpu_mb)

class ASRMetricsTracker(MetricsTracker):
    """Specialized metrics tracker for ASR tasks."""
    
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.references = []
        
    def update_transcriptions(
        self,
        batch_predictions: List[str],
        batch_references: List[str]
    ):
        """Update with new transcriptions."""
        self.predictions.extend(batch_predictions)
        self.references.extend(batch_references)
        
        # Compute batch metrics
        batch_wer = compute_wer(batch_predictions, batch_references)
        batch_cer = compute_cer(batch_predictions, batch_references)
        
        self.update({
            'wer': batch_wer,
            'cer': batch_cer
        })
        
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics including overall WER/CER."""
        metrics = self.get_average()
        
        # Compute overall WER/CER on complete dataset
        metrics['overall_wer'] = compute_wer(self.predictions, self.references)
        metrics['overall_cer'] = compute_cer(self.predictions, self.references)
        
        return metrics
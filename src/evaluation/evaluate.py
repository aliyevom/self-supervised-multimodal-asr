import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import time
import json
from typing import Dict, List
import numpy as np
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from jiwer import wer
import psutil
import torch.cuda as cuda

from src.models.context_encoder import ContextEncoder
from src.models.fusion_layer import DeepFusionLayer
from src.training.train_fusion import ContextEnhancedASR
from src.data.dataset import ContextualSpeechDataset

def add_noise(
    audio: torch.Tensor,
    noise_level: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise to audio signal."""
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

def measure_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """Measure GPU and CPU memory usage."""
    memory_stats = {}
    
    # CPU memory
    process = psutil.Process()
    memory_stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_stats["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
    return memory_stats

def measure_latency(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_runs: int = 100,
) -> Dict[str, float]:
    """Measure model inference latency."""
    latencies = []
    
    # Warmup
    for _ in range(10):
        _ = model(**inputs)
        
    # Measure latency
    torch.cuda.synchronize()
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start_time)
        
    return {
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "p90_latency_ms": np.percentile(latencies, 90) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
    }

@hydra.main(config_path="../../configs", config_name="default")
def evaluate(cfg: DictConfig):
    # Load models and processor
    processor = Wav2Vec2Processor.from_pretrained(cfg.model.asr.pretrained)
    
    # Load context encoder
    context_encoder = ContextEncoder(
        input_channels=cfg.model.context_encoder.input_channels,
        hidden_dim=cfg.model.context_encoder.hidden_dim,
        num_layers=cfg.model.context_encoder.num_layers,
        kernel_size=cfg.model.context_encoder.kernel_size,
        stride=cfg.model.context_encoder.stride,
        context_dim=cfg.model.context_encoder.context_dim,
    )
    
    context_checkpoint = torch.load("checkpoints/context_encoder_best.pt")
    context_encoder.load_state_dict(context_checkpoint["model_state_dict"])
    
    # Load fusion model
    fusion_checkpoint = torch.load("checkpoints/fusion_model_best.pt")
    model = ContextEnhancedASR.load_state_dict(fusion_checkpoint["model_state_dict"])
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Create evaluation dataset
    dataset = ContextualSpeechDataset(
        speech_dir=cfg.data.librispeech.test_clean,
        context_dir=cfg.data.context.test_path,
        sampling_rate=cfg.data.context.sampling_rate,
        max_duration=cfg.data.librispeech.max_duration,
        context_duration=cfg.data.context.clip_duration,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    results = {
        "wer": {},
        "latency": {},
        "memory": measure_memory_usage(model),
    }
    
    # Evaluate for different noise levels
    for noise_level in cfg.evaluation.noise_levels:
        print(f"\nEvaluating with noise level {noise_level}")
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Add noise to speech
                speech = batch["speech"]
                if noise_level > 0:
                    speech = add_noise(speech, noise_level)
                    
                if torch.cuda.is_available():
                    speech = speech.cuda()
                    batch["context"] = batch["context"].cuda()
                    
                # Forward pass
                logits = model(
                    speech=speech,
                    context=batch["context"],
                    attention_mask=torch.ones_like(speech),
                )
                
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions.extend(
                    processor.batch_decode(predicted_ids)
                )
                
                # Get reference transcriptions
                references.extend(
                    processor.batch_decode(
                        processor(batch["speech_path"], return_tensors="pt").input_ids
                    )
                )
                
        # Compute WER
        noise_wer = wer(references, predictions)
        results["wer"][f"noise_{noise_level}"] = noise_wer
        print(f"WER: {noise_wer:.4f}")
        
        # Measure latency
        sample_inputs = {
            "speech": speech[:1],  # Single example
            "context": batch["context"][:1],
            "attention_mask": torch.ones_like(speech[:1]),
        }
        latency_stats = measure_latency(model, sample_inputs)
        results["latency"][f"noise_{noise_level}"] = latency_stats
        print(f"Mean latency: {latency_stats['mean_latency_ms']:.2f} ms")
        
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nEvaluation complete! Results saved to evaluation_results/metrics.json")

if __name__ == "__main__":
    evaluate()
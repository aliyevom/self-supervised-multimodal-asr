# Self-Supervised Multimodal ASR with Noise Context

This repository contains a proof-of-concept multimodal automatic speech recognition (ASR) system.
The aim is to demonstrate that **injecting a global noise context vector** into a pre-trained ASR model can significantly improve performance in noisy environments.

The concept is based on research from [Amazon scientists](https://www.amazon.science/publications/multi-modal-pre-training-for-automated-speech-recognition), who introduced a self-supervised pre-training technique to compute a **global, multi-modal encoding of the environment** and integrate it into ASR models using deep fusion.
Their experiments on LibriSpeech showed:

* Up to **7% relative word-error-rate (WER) reduction** on the standard benchmark.
* Between **6–45% WER reduction** on internal datasets.

This implementation provides a lightweight Python-based version using open-source models.

![Architecture Overview](public/img/base.png)
*Architecture diagram: self-supervised noise context encoder + deep fusion*

### Technical Development Plan

```mermaid
gantt
  dateFormat  YYYY-MM-DD
  axisFormat  %b %Y

  section Encoder Architectures
  Masked span prediction v2       :done, ce1, 2025-01-10, 30d
  Contrastive self-supervision    :done, ce2, 2025-02-15, 21d
  Multi-scale feature pyramids    :active, ce3, 2025-03-10, 45d
  Hierarchical attention layers   :      ce4, 2025-04-25, 21d
  Temporal context modeling       :      ce5, 2025-05-20, 21d

  section Fusion Mechanisms
  Deep fusion baseline            :done, fs1, 2025-07-01, 10d
  Adaptive gating (MoE)           :active, fs2, 2025-07-15, 30d
  Cross-modal attention maps      :      fs3, 2025-08-15, 30d
  Conditional FiLM layers         :      fs4, 2025-09-15, 30d
  Hybrid early/late fusion        :      fs5, 2025-10-20, 30d

  section Noise Robustness
  Dynamic noise mixing             :     na1, 2025-09-01, 21d
  Environmental simulation         :     na2, 2025-09-25, 21d
  Curriculum noise injection       :     na3, 2025-10-15, 21d
  Real-world microphone eval       :     na4, 2025-11-05, 21d

  section Systems Optimization
  Latency reduction (graph opt)    :     po1, 2025-11-01, 14d
  Memory footprint optimization    :     po2, 2025-11-20, 14d
  Inference kernel fusion          :     po3, 2025-12-05, 14d
  Low-rank factorization           :     po4, 2025-12-20, 14d

  section Training Infrastructure
  Distributed data parallel         :    tp1, 2025-12-15, 21d
  Tensor/pipeline parallelism       :    tp2, 2026-01-05, 21d
  Mixed precision FP16/BF16         :    tp3, 2026-01-25, 21d
  Gradient accumulation scaling     :    tp4, 2026-02-15, 21d

  section Evaluation Frameworks
  Advanced WER analytics            :    ev1, 2026-02-20, 14d
  Noise robustness benchmarks       :    ev2, 2026-03-10, 14d
  Latency & throughput profiling    :    ev3, 2026-03-25, 14d
  Ablation grid experiments         :    ev4, 2026-04-10, 14d

  section Compression & Deployment
  Knowledge distillation            :    mc1, 2026-04-25, 21d
  Quantization (INT8/FP8)           :    mc2, 2026-05-20, 21d
  Structured pruning                :    mc3, 2026-06-10, 21d
  Neural architecture search (NAS)  :    mc4, 2026-06-25, 21d

  %% NEW: Backend Systems (highly technical)
  section Backend Systems
  Streaming inference (gRPC/WS)     :    be1, 2025-08-01, 28d
  Dynamic batching (Triton)         :    be2, 2025-08-25, 21d
  ONNX/TensorRT export pipeline     :    be3, 2025-09-15, 21d
  RNNT/CTC beam-search optimizer    :    be4, 2025-10-05, 21d
  KV/feature cache (Redis)          :    be5, 2025-10-28, 21d
  Telemetry & tracing (OTel)        :    be6, 2025-11-18, 14d
  AuthN/Z (OIDC) + rate limiting    :    be7, 2025-12-05, 14d
  A/B + shadow deploy service       :    be8, 2026-01-05, 21d
  Data ingest (Kafka) + feature store:   be9, 2026-01-28, 21d
  Canary + rollback automation      :    be10, 2026-02-20, 14d

  %% NEW: Frontend / Tooling (deep client + ops tooling)
  section Frontend / Tooling
  WebAudio/WebRTC mic + VAD         :    fe1, 2025-08-10, 21d
  Real-time spectrogram + noise meter:   fe2, 2025-09-01, 21d
  Streaming partials UI (WS/SSE)    :    fe3, 2025-09-25, 14d
  Model/config panel (React)        :    fe4, 2025-10-10, 14d
  Latency HUD + client logs         :    fe5, 2025-10-28, 14d
  ONNX Runtime Web (WASM) demo      :    fe6, 2025-11-15, 21d
  Dataset curation/annotation UI    :    fe7, 2025-12-10, 21d
  Evaluation dashboard (WER, CER)   :    fe8, 2026-01-05, 21d
  Experiment console (A/B toggles)  :    fe9, 2026-01-28, 14d
  Access control + audit (frontend) :    fe10, 2026-02-15, 14d

```

## Background

Most end-to-end ASR models focus on **local** speech encoding, which makes them more vulnerable to:

* Frame dropouts
* Unseen background noise

The Amazon research addressed this by:

1. Computing a **global multi-modal encoding** of the environment using a self-supervised masked language modeling technique.
2. Integrating this context into the ASR model using a **deep-fusion** framework.

Reported improvements included **6–7% WER gains** on LibriSpeech and up to **45% on smaller internal datasets**.



## Algorithm Approach

This repository follows the same philosophy:

* **Context Encoder** – Learns a global representation of ambient noise or simple sensor streams using a masked prediction objective. ([ArXiv Paper](https://arxiv.org/abs/2110.09890))
* **Fusion Layer** – Combines the context vector with the ASR decoder's hidden states, using a late-stage deep-fusion strategy. ([Fusion Strategies Overview](https://blog.shahadmahmud.com/language-model-integration/#:~:text=,language%20model%20and%20the%20decoder))
* **Baseline ASR** – Uses an existing pre-trained model such as [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) or [Whisper](https://openai.com/research/whisper) for speech encoding/decoding.

---

## Example Results

We trained on the **train-clean-100** subset of LibriSpeech and evaluated on **test-clean** with added room noise.

* **Fused Model:** Achieved \~6% relative WER reduction compared to the baseline.
* **Baseline:** Standard pre-trained ASR without noise-context fusion.

These results are consistent with Amazon's reported improvements.
Full evaluation details can be reproduced using the included scripts.



## Resources & References

* **[Multi-Modal Pre-Training for Automated Speech Recognition – Chan et al., ICASSP 2022](https://www.amazon.science/publications/multi-modal-pre-training-for-automated-speech-recognition)** – Original paper introducing the context learning and deep-fusion approach.
* **[ArXiv Version](https://arxiv.org/abs/2110.09890)** – Abstract and experimental details.
* **[Language Model Integration – Shahad Mahmud](https://blog.shahadmahmud.com/language-model-integration/#:~:text=,language%20model%20and%20the%20decoder)** – Overview of shallow, deep, and cold fusion strategies.
* **[LibriSpeech Dataset](https://www.openslr.org/12/)** – Open corpus for ASR training.
* **[wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)** – Self-supervised speech representation model from Facebook AI.
* **[Whisper](https://openai.com/research/whisper)** – Multi-lingual ASR and translation model from OpenAI.
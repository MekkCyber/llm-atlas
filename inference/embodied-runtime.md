# Embodied Inference Runtime (Embodied.cpp)
*Depth — a portable C++ runtime for VLA / world-action model inference across heterogeneous robot hardware.*

**TL;DR:** VLA and world-action models are typically deployed via research Python stacks that assume request-response serving, batch >1, and fixed token I/O. Robots need the opposite: closed-loop *multi-rate* execution (perception, planning, action heads ticking at different frequencies), latency-first *batch-1* inference on heterogeneous edge devices, and *typed I/O* that goes beyond tokens (joint states, camera streams, force-torque). Embodied.cpp analyzes the shared execution shape of representative VLAs (HY-VLA, π0.5) and WAMs, then exposes it through five plugin layers — input adapters, sequence builders, backbone execution, head plugins, deployment adapters — so one runtime spans models and robots.

**Prereqs:** [../agents/_vla](../agents/_vla.md)
**Related:** [../agents/action-chunk-correction.md](../agents/action-chunk-correction.md), [../quantization/README.md](../quantization/README.md)

---

## What it is

Serving engines for LLMs (vLLM, SGLang, TRT-LLM) are optimized for text: high-throughput batched autoregressive decoding, paged KV cache, continuous batching. They don't fit robots:

- **Batch size 1 is the norm.** A robot runs one policy; there is nothing to batch with.
- **Latency is the tail metric.** A worst-case 100ms delay in the control loop can kill a manipulation task.
- **Multiple output heads at different rates.** A VLA might update its high-level plan at 5 Hz while emitting motor commands at 200 Hz. LLM runtimes have one output rate: token/step.
- **Heterogeneous edge hardware.** Robots ship on Jetson, custom ASICs, x86 laptops, Apple Silicon. Runtime has to move.
- **Typed I/O beyond tokens.** Perception inputs are images / point clouds / IMU streams; outputs are typed commands. Fixed-token I/O forces round-trips through tokenizer boilerplate that adds latency.

Embodied.cpp attacks these five constraints simultaneously with a C++ runtime and a five-layer plugin architecture.

## How it works

The runtime decomposes any VLA / WAM into a fixed pipeline of five plugin points:

1. **Input adapter.** Ingests raw sensor streams (image, joint state, force-torque, language instruction) and materializes typed tensors. Adapters are per-modality and per-robot.
2. **Sequence builder.** Assembles the model's expected sequence: for a VLA, this typically means interleaving vision tokens, instruction tokens, and proprioceptive state tokens in the format the specific model was trained for.
3. **Backbone execution.** Runs the transformer / hybrid backbone through one *backend abstraction* that maps to CUDA / Metal / custom NPU as available. This is the layer that quantization (FP8, INT4, MXFP4) and speculative decoding hook into.
4. **Head plugin.** Per-model action / value / planning head — discrete tokens, flow-matching decoders, DiT action decoders, whatever the model uses. Multiple heads can tick at different rates; the runtime schedules them.
5. **Deployment adapter.** Emits typed commands to the robot's control stack (ROS, custom low-level control, sim adapters for testing).

**Multi-rate scheduling.** The core novelty vs. LLM runtimes. Perception can run at 20 Hz, the backbone at 5 Hz, action head at 200 Hz — the scheduler runs each block at its own rate, sharing the backbone's cached activations across action-head ticks. This is what makes closed-loop control at high frequency feasible with a single VLA.

**Latency-first fused inference.** Because batch size is 1, the runtime aggressively fuses operators to cut kernel-launch overhead. Common fusions: attention + residual, RMSNorm + projection, output head + activation.

## Why it matters

- **VLA deployment story was broken.** Research pipelines depended on Python + PyTorch on the robot side; edge inference was painful and non-portable. A C++ runtime with a plugin architecture is what the field needs to graduate from prototype demos to real deployment.
- **Multi-rate scheduling is a new pattern for serving.** LLM inference engines all tick at one rate. Any agent that combines slow reasoning with fast reactive execution (VLAs, but also planning-based agents) benefits from this scheduler design.
- **Concrete robotics numbers.** On HY-VLA and π0.5, closed-loop success rates of 100.0% and 91.0% respectively. WAM benchmark: 312.2 MiB → 88.1 MiB per transformer block. Real numbers on real models, not toy examples.
- **A `llama.cpp` for embodied models.** The industry accepted `llama.cpp` as the portable low-level runtime for text LLMs; Embodied.cpp is positioned to play the same role for VLA/WAM.

## Gotchas & tricks

- **Model-specific quirks live in adapters.** Every VLA has bespoke tokenization (image patch layout, action encoding); the plugin layers explicitly isolate this. Adding a new model = writing an adapter, not modifying the runtime.
- **Backbone abstraction is where quantization lives.** FP8 / INT4 / MXFP4 quantization applied at the backbone layer is the biggest lever for edge deployment; the runtime treats quantization as a backend property.
- **Multi-rate scheduling needs cross-rate sync.** If perception updates at 20 Hz while action head reads it at 200 Hz, the runtime has to hold the latest perception activation without race conditions. Lock-free ring buffers are the typical implementation.
- **Deterministic-latency guarantees.** Some robotics deployments require hard-real-time bounds. C++ helps but is not sufficient; jitter analysis on the actual target hardware is still required.
- **Sim / real parity.** The same runtime should serve the model in simulation and on the physical robot. The deployment adapter is the seam; keeping sim and real behind the same interface avoids the classic "works in sim, breaks on robot" trap.

## Sources

- Paper: *Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots* — Xu et al., Southeast U. / Nanjing U. / MSR / Tsinghua AIR, 2026 — [arXiv:2607.02501](https://arxiv.org/abs/2607.02501)
- Related: `llama.cpp` (Gerganov, 2023–) — the closest analog for text LLMs.
- Related: π0 / π0.5 (Physical Intelligence, 2024–25) — one of the target models.

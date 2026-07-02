# Block Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Block Diffusion LMs (BD-LMs like LLaDA2) generate language block-by-block via iterative denoising rather than one token at a time, while still supporting KV caching and flexible-length generation. Multi-Block Diffusion (MBD-LM, 2026) extends BD-LMs so that several consecutive noisy blocks decode *concurrently* using Multi-block Teacher Forcing training and a Block Buffer inference mechanism, translating parallelism into wall-clock speedup on math and code benchmarks.

**Prereqs:** [README](README.md)
**Related:** [diffusion-speculative-decoding](diffusion-speculative-decoding.md) · [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

Two dominant LM families:

- **Autoregressive (AR).** One token per forward pass; KV cache reuse; irreducible sequential latency.
- **Diffusion LMs (D-LMs).** Denoise a fixed-length noisy sequence in $T$ steps. Highly parallel but pays $T$ target-model forwards per generation and doesn't naturally support flexible length.

**Block Diffusion LMs** are the middle: divide the output into fixed-size *blocks* of $B$ tokens; denoise one block at a time conditioned on the clean prefix; treat blocks as the AR unit. Result: KV cache carries across blocks, flexible length via more blocks, and each block delivers $B$ tokens for a small number of denoising forwards.

**Multi-Block Diffusion** extends this to denoise multiple consecutive noisy blocks concurrently, chasing more inter-block parallelism.

## How it works

### Single-block diffusion (SingleBD)

Standard BD-LM training: teacher forcing on one noisy block conditioned on a clean prefix. Inference: denoise block $b$, commit, move to $b{+}1$. Simple; KV-cache friendly.

### Multi-block inference (MultiBD)

At inference, maintain a running set of consecutive noisy blocks; each step advances the denoising of all of them. Once block $b$ is fully denoised it commits; a new noisy block enters at the tail. Slot-wise, blocks are at heterogeneous noise levels: block 0 near clean, block $k{-}1$ still very noisy.

### Multi-block Teacher Forcing (MultiTF)

The training gap MBD-LM closes: SingleBD training exposes the model to only one noisy block; MultiBD inference expects heterogeneous noise across multiple blocks. **MultiTF** trains on bounded noise-groups conditioned on clean prefixes with randomized noise-schedulers, so the training distribution matches inference. Bridges teacher forcing and diffusion forcing.

### Block Buffer decoder

To realize wall-clock speedup, the runtime keeps a fixed-shape input buffer of $k$ blocks; committed blocks slide out and new noisy blocks slide in. Static input shapes preserve prefix-cache reuse, avoid recompilation, and let the increased parallelism land as real tokens-per-second.

## Why it matters

- **Higher throughput than AR at similar quality.** MBD-LLaDA2-Mini raises Tokens Per Forward (TPF) from 3.47 → 6.19 and average accuracy from 79.95% → 81.03%. With DMax, TPF reaches 9.34 at 1.02% accuracy drop on math and code benchmarks.
- **Latency-sensitive serving.** Diffusion LMs were previously suspect for production because AR was faster in wall-clock; MBD-LM's Block Buffer flips the balance.
- **Complements speculative decoding.** BD-LMs also work well as *draft models* for AR targets — see [diffusion-speculative-decoding](diffusion-speculative-decoding.md).

## Gotchas & tricks

- **Block size is a critical knob.** Small blocks → low throughput; big blocks → higher rejection risk in speculative use and higher training-inference mismatch. Match training and deployment blocks.
- **Buffer size trades latency for TTFT.** More blocks in-flight raises TPF but raises time-to-first-token; balance against the deployment SLO.
- **KV cache must be prefix-only.** Anything after the current block boundary changes per denoising step; only the clean prefix cache is truly reusable.
- **Randomized noise schedule during training.** Deterministic training-time schedules that don't match MultiBD's slot-wise heterogeneity leave TPF on the table.

## Sources

- Paper: *Multi-Block Diffusion Language Models* — Jin, Xu, Liu et al., 2026 — SJTU / Huawei; introduces MultiTF and Block Buffer.
- Earlier BD-LM lineage: LLaDA2 and predecessor block-diffusion LMs (2025); the SingleBD substrate.

# Causal autoregressive video diffusion

*Depth — distilling bidirectional video diffusion teachers into causal, few-step, streaming students.*

**TL;DR:** Bidirectional video diffusion gives high quality but requires the full sequence in advance and many denoising steps — incompatible with real-time streaming. Causal autoregressive diffusion generates the video chunk-by-chunk in temporal order with each chunk requiring only a couple of denoising steps. Lip Forcing (2026) shows this can be done by distilling a 14B bidirectional teacher into autoregressive students that hit 31 FPS streaming at 1.3B and 39.8× speedup at 14B, with sub-millisecond time-to-first-frame.

**Prereqs:** (none yet)
**Related:** [README](README.md)

---

## What it is

Two axes of choice for video diffusion:

|  | Few-step | Many-step |
| --- | --- | --- |
| **Bidirectional (full sequence)** | distilled bidirectional (DMD-like) | classical video diffusion (DiT-based) |
| **Causal (chunk-by-chunk)** | **causal autoregressive diffusion (Lip Forcing, Self-Forcing lineage)** | causal but slow — rarely used |

The causal-few-step quadrant is where real-time streaming lives. The chunk-by-chunk generation lets you start emitting frames before the input is complete; the few-step denoising keeps per-chunk latency low; together, time-to-first-frame becomes sub-millisecond and steady-state FPS clears 30.

## How it works

### Chunked autoregressive generation

Partition the target video into temporal chunks of $K$ frames each. The model generates chunk $i$ conditioned on chunks $1, \ldots, i-1$ (causal) and on whatever side conditioning the task requires (audio, text, reference frame). Each chunk goes through a small number ($N$, typically 2) of denoising steps.

### Distillation from a bidirectional teacher

Train a chunk-causal student to imitate a strong bidirectional teacher. Lip Forcing's recipe:

1. **Identify the useful CFG window.** Trajectory analysis of the teacher reveals that CFG predictions help *sync* in a mid-trajectory band but hurt reference fidelity elsewhere. Distillation should only match the teacher inside the useful window.
2. **Sync-Window DMD.** Distribution-matching distillation applied only in that mid-trajectory band; outside the band, the student uses no-CFG predictions for fidelity.
3. **Few-step schedule.** Pick the $N$-step inference schedule that hits the target latency.
4. **Domain-specific reward.** Lip Forcing adds a SyncNet-based reward to the distilled student, post-hoc-tuning it for lip sync.

### Streaming inference

At inference, generate chunk $i+1$ while the user/agent is still consuming chunk $i$. No CFG at inference, no future-context dependency, no inference-time guidance steering. Each new chunk requires just $N$ forward passes.

## Why it matters

- **Real-time video generation is now serveable.** 31 FPS at 1.3B model size, sub-ms time-to-first-frame at 14B. Within range for product deployment.
- **The distillation recipe generalizes.** Trajectory-analysis-derived distillation windows + domain-specific reward = a template for taking any bidirectional video model causal.
- **Reframes the video diffusion problem.** "Big bidirectional model is the only path to quality" was the dominant assumption; Lip Forcing shows a 39.8× speedup at the *same* model size by changing the generation order, not the model.
- **Pairs with hybrid attention.** Long-video understanding stacks ([hybrid linear attention](../architectures/hybrid-linear-attention.md), sparse attention) consume causal video well; this paper closes the *generation* side of the same loop.

## Gotchas & tricks

- **Chunk size $K$ trades latency vs quality.** Smaller chunks → lower latency but more boundary artifacts; larger chunks → smoother but slower start.
- **Number of denoising steps $N$.** $N=2$ is aggressive — works for distilled students with the right reward, fails for non-distilled models.
- **Reference-fidelity drift.** Chunk-by-chunk generation can drift from the reference over long sequences. Periodically re-conditioning on the reference helps.
- **Domain-specific reward is not optional.** The cited paper relies on SyncNet for lip-sync; transferring to another domain (e.g. controllable character animation) needs a domain-appropriate reward, or quality degrades visibly.
- **CFG only helps inside the analysis window.** Using CFG everywhere overshoots and hurts fidelity; trajectory analysis is essential.

## Sources

- Paper: *Lip Forcing: Few-Step Autoregressive Diffusion for Real-time Lip Synchronization* — Cho et al., KAIST AI / AIPARK, 2026 — [arXiv 2606.11180](https://arxiv.org/abs/2606.11180).
- Background: DMD (Distribution Matching Distillation) — Yin et al., 2024.
- Related causal-AR video lineage: Self-Forcing and follow-ons.

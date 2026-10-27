# Hallucination correction via representation surgery
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A *training-free* method for correcting cross-modal hallucinations in **omni-modal LLMs** (audio + vision + text), introduced as **Omni-Woodpecker**. Rather than detecting hallucinations in the decoded text and re-prompting, it intervenes **inside the fused hidden representation**: it identifies conflicting modality cues, separates grounded evidence from learned language priors, and rebalances the fused state before the next decoding step. Works on existing checkpoints with no retraining.

**Prereqs:** [README.md](README.md)
**Related:** [../interpretability/README.md](../interpretability/README.md), [../safety/README.md](../safety/README.md)

---

## What it is

Omni-LLMs (e.g., Qwen2.5-Omni, VideoLLaMA2-AV) fuse audio, vision, and text into a single hidden representation. Hallucinations happen when the fused state is pulled toward **language priors** that disagree with the audio/video evidence — the model answers from what text-like inputs usually look like, not from what is actually in the clip. Representation-level correction attacks the problem at its source, before tokens are decoded.

## How it works

The method runs at inference time, as a wrapper around the omni-LLM forward pass:

1. **Decompose the fused hidden state.** At one or more layers, project the state into *evidence-driven* and *prior-driven* components, using cheap probes derived from the modality tokens themselves (visual-only / audio-only / text-only paths).
2. **Detect conflict.** Measure disagreement between the evidence and prior components for a given token position. If they align, do nothing.
3. **Rebalance.** Where conflict is high, scale down the prior component and scale up the evidence component before passing the state to the next layer / decoding head. The scalar re-weighting is parameter-free in the sense that no network weights are updated.

Because the intervention is a hidden-state transform rather than a retraining step, it composes with any omni-LLM that exposes modality-separated tokens or activations.

## Why it matters

- **Training-free fix on existing checkpoints.** Can be deployed to models like Qwen2.5-Omni-7B and VideoLLaMA2-AV without retraining — the dominant cost model for omni-LLM deployment.
- **Reduces hallucinations with no measured regression on general reasoning benchmarks** on the two tested backbones.
- **Evidence-vs-prior framing is reusable.** The same decomposition idea could apply to text-only hallucination (world knowledge vs. input evidence) or VLM hallucination, giving interpretability work a concrete deployment lever.

## Gotchas & tricks

- **Needs modality-separated signals.** Backbones that fuse too early (no per-modality residual path) are harder to decompose without extra probes.
- **The rebalance coefficient is a tradeoff.** Too aggressive and the model ignores priors even when they are correct (e.g., for abstract reasoning); too soft and the hallucinations persist. Picked per-benchmark in the paper.
- **Not a verifier.** The method does not know *what* the right answer is; it only rebalances the state toward evidence. If the evidence itself is ambiguous, the method does nothing useful.
- **Latency.** Any per-layer intervention adds inference cost. The paper's version is cheap enough to run online, but naive implementations that run the probes multiple times per token can be slow.

## Sources

- Paper: *Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs* — Liu, Fu, Yang, Shan, He, 2026 — https://arxiv.org/abs/2610.27103

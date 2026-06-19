# Bag of Dims — Standard-Basis Feature Decoding

*Depth — sign patterns of the residual-stream standard basis already encode interpretable features, no SAE required.*

**TL;DR:** The transformer residual stream's *standard basis* — the raw hidden-state coordinates — already functions as a training-free, architecture-general feature basis. Each dimension acts as an **independent binary register**: the **sign** ($\pm 1$) carries semantic content, the **magnitude** carries confidence. Replacing all magnitudes with unity preserves 72–93% top-5 next-token accuracy. Sign patterns alone organize into 175 semantic categories (mean AUC 0.80) discovered with zero training across Qwen 3.5-4B, Gemma 3-4B, and Mistral 7B.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [sparse-autoencoders](sparse-autoencoders.md)

---

## What it is

A counter-claim to the SAE program: you don't need a learned dictionary to find interpretable features in a transformer. The dimensions of the residual stream, as-is, carry the structure SAEs go looking for — and you can read it with sign patterns and Hamming distance.

## How it works

Treat each hidden-state dimension $h_d$ as a register with two states $\text{sign}(h_d) \in \{-1, +1\}$ plus a magnitude $|h_d|$.

**Sign-only test.** Replace $h$ with $\text{sign}(h)$ (all magnitudes set to 1) and run through the LM head. Achieves 72–93% top-5 next-token accuracy across models — almost all of the head's predictive power is in the signs.

**Decoder-free Hamming scoring.** Don't even use the LM head. Compute $h_{\text{query}}$ and $h_{\text{candidate}}$ for each vocabulary token, score by Hamming distance over signs. Reaches 80–90% top-4096 accuracy.

**Single-token type cache.** One forward pass per vocabulary token (no context) gives a `vocab_size × d` matrix of sign vectors. From 50 hand-picked semantic anchors, per-dimension sign consistency discovers 175 categories with mean AUC 0.80 — feature discovery with **zero training**.

The interpretation: the model writes information into the residual stream as a near-binary code (high-magnitude positive or negative), and downstream layers read mostly the sign. Magnitudes act as confidence / gating.

## Why it matters

- **Baseline for SAE papers.** Any SAE result needs to clear the standard-basis baseline. Many published feature discoveries can probably be replicated with sign patterns alone.
- **Free interpretability.** Training an SAE on a frontier model costs millions of activations and weeks of compute. Bag of Dims costs one inference pass per anchor.
- **Architecture-general.** Works on Qwen, Gemma, Mistral with the same recipe — no per-model dictionary training.
- **Mechanistic story.** Forces a rethink of *what the residual stream is*. If it's already a binary code, the SAE narrative ("the model uses a low-dimensional manifold; SAEs recover the dictionary") is at least incomplete.

## Gotchas & tricks

- **Sign-only is lossy.** Magnitudes do carry useful confidence — for some downstream tasks (e.g. graded retrieval), magnitude matters and signs alone fall short.
- **Single-token cache is a stripped-down setting.** Context-dependent features (anaphora, grammatical role) need context-conditioned activations and aren't directly readable from the type cache.
- **Not a complete replacement for SAEs.** SAEs still win for *causal* steering (clamping a specific concept in a specific token position). Bag of Dims is more about *what's already there to be read*.
- **Sign-consistency category discovery is the cleanest experiment** — try it first on any new model to see if the standard-basis features look as good as claimed before going to SAE training.

## Sources

- Paper: *Bag of Dims: Training-Free Mechanistic Interpretability via Dimension-Level Sign Patterns* — 2026 — [arXiv:2606.12629](https://arxiv.org/abs/2606.12629).

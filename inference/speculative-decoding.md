# Speculative Decoding
*Depth — accelerate autoregressive decoding by drafting multiple tokens cheaply and verifying them in parallel.*

**TL;DR:** The target model's forward pass is the decode bottleneck. Speculative decoding has a small draft model propose $K$ next tokens, the target model verifies all $K$ in a single forward pass (parallel verification), and accepts the longest prefix that matches the target's own distribution. Average speedups of 2–3× at matched quality are routine for batch-size-1 serving; higher for well-matched draft/target pairs. A recent variant, Feedback Shortcut (2026), extends the recipe to latent-feedback transformers whose hidden-state dependencies normally break the draft pathway.

**Prereqs:** [README](README.md)
**Related:** [MTP](../pre-training/mtp.md)

---

## What it is

An acceleration scheme for autoregressive token generation that produces **bit-identical outputs** to the target model while doing most of the work with a cheaper draft. First introduced by Leviathan et al. and Chen et al. (2023); now standard in vLLM, SGLang, and most serving frameworks.

## How it works

Per decode iteration:

1. **Draft.** A small draft model proposes $K$ tokens autoregressively (cheap).
2. **Verify.** The target model runs a single forward pass on the drafted sequence, producing distributions at each position in parallel.
3. **Accept.** At each drafted position $i$, sample acceptance with probability $\min(1, p_{\text{target}}(x_i) / p_{\text{draft}}(x_i))$. Accept the longest prefix; on first rejection, sample a correction from the adjusted residual.

Quality is preserved exactly (rejection sampling guarantees the output distribution equals the target's). Only the speed changes.

Draft-model choices: distilled small model (e.g., EAGLE, Medusa heads), MTP head sharing weights with the target, or a self-speculating variant where drafts come from the same model's earlier layers.

## Why it matters

Decode throughput at batch size 1 is latency-limited by target FLOPs per token. Speculation shifts the bottleneck to draft FLOPs plus one target pass per $K$ tokens. For $K=4$ and 80% average acceptance, throughput rises ~3× with no quality cost. This is the dominant inference acceleration today for single-user serving.

**Interaction with other serving tricks** — speculation composes with paged attention (both target and draft share a KV cache), continuous batching (draft+verify both batch across requests), and quantization (draft can be heavily quantized since errors are corrected). It does *not* compose trivially with latent-feedback architectures, which Feedback Shortcut (2026) fixes.

## Gotchas & tricks

- **Draft-target alignment is everything.** A poorly matched draft wastes target forward passes on rejections. Drafts distilled from the target (EAGLE, Medusa) dominate off-the-shelf small models.
- **$K$ is a bias-variance knob.** Larger $K$ → more wins when accepted, more waste on first-reject. Typical sweet spot is 4–6.
- **Batch sizes dilute the win.** As batch grows, target's forward is already amortized across requests; speculation's relative benefit shrinks. At very large batch sizes (≥ 64), plain batched decode can win.
- **Correctness proofs assume temperature-matched sampling.** Using a different sampling policy for draft vs. target can break the equivalence to pure target sampling.
- **Latent feedback transformers** (where the top-layer state feeds the next position's input) break the draft pathway — the draft needs top-layer states it hasn't computed yet. Feedback Shortcut (2026) adds a lightweight projection to supply stand-in states plus pipelined correction inside the verification pass, recovering 2.4–2.8× speedups for ~2% extra training compute.

## Sources

- Paper: Fast Inference from Transformers via Speculative Decoding — Leviathan et al., 2023 — https://arxiv.org/abs/2211.17192
- Paper: Accelerating Large Language Model Decoding with Speculative Sampling — Chen et al., 2023 — https://arxiv.org/abs/2302.01318
- Paper: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty — Li et al., 2024 — https://arxiv.org/abs/2401.15077
- Paper: Feedback Shortcut: Speculative Decoding for Latent Feedback Transformers — 2026 — [arXiv:2611.28227](https://arxiv.org/abs/2611.28227)

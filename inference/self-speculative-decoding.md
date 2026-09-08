# Self-Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a **subset of the model's own layers** as the drafter and the full layer stack as the verifier. No separate draft model, no auxiliary training run — provided the target model was pre-trained with layer dropout so its shallow forward is already a well-calibrated stand-in for the full forward. Delivers up to ≈1.5× inference speedup with negligible accuracy loss on layer-dropout-trained backbones.

**Prereqs:** [../pre-training/layer-dropout.md](../pre-training/layer-dropout.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

Classical speculative decoding uses a small draft model $M_\text{draft}$ to propose $k$ tokens and the target model $M_\text{target}$ to verify them in parallel with a single forward. The catch: you need a draft model of a compatible tokenizer, comparable-distribution behavior, and separate serving infra.

Self-speculative decoding removes the separate draft model. The idea: run the target model with only its **shallow prefix** of layers (or with dropout-skipped middle layers) to propose $k$ tokens; then run the *full* stack once to verify. Whenever the shallow forward already agrees with the deep forward on a token, you skip that token's full compute.

## How it works

- **Draft phase.** Forward the model with only a shallow layer subset — e.g. layers $[0, L/2)$, or a stochastically-picked mask matching the training-time layer-drop pattern. Emit $k$ candidate tokens greedily (or by sampling).
- **Verify phase.** Run the *full* $L$-layer forward on the prefix + $k$ candidates in a single batched forward. Compare each drafted token's logits against the full-stack logits under the same acceptance rule as classical speculative decoding (rejection-sampled or greedy match).
- **Accept prefix.** All drafted tokens matching the full-stack prediction are accepted; the first mismatch is replaced with the full-stack sample, and drafting resumes.

The trick is **compatibility**: the shallow forward must produce logits close enough to the full forward that acceptance rates stay high. Layer-dropout pre-training explicitly trains the model to survive shallow forwards, so this compatibility holds by construction.

## Why it matters

- **No auxiliary artifacts.** No separately-trained draft model, no separate weights, no tokenizer mismatch. One checkpoint serves both roles.
- **Composes with layer-drop pre-training.** The same training-time investment (layer dropout) that reduces pre-training FLOPs *also* enables this inference speedup — a rare same-lever-two-wins.
- **Simpler serving stack.** Speculative decoding's operational cost has historically been maintaining a *second* model in sync with the target. Self-speculative decoding erases that.
- **Tunes trivially against latency vs. quality.** Vary the shallow subset size to trade acceptance rate against draft cost.

## Gotchas & tricks

- **Requires layer-dropout pre-training.** On a backbone trained *without* layer dropout, shallow forwards are meaningless — acceptance rates collapse and you get no speedup.
- **Shallow subset choice matters.** A contiguous prefix is simple but not always optimal; the mask that most closely matches training-time layer-drop patterns tends to accept best.
- **Batch dynamics.** Speedup depends on acceptance length distribution; low-acceptance regions (specialized domains, rare tokens) can be slower than plain decoding. Fall back to standard decoding when rolling acceptance drops below a threshold.
- **KV-cache reuse.** Draft and verify pass must share the KV cache correctly — dropped-layer KVs are absent, and the verify pass has to reconstitute them. Off-by-one bugs here silently degrade quality.
- **Interacts with quantization.** FP8/INT4 KV caches magnify shallow-vs-deep disagreement; expect lower acceptance rates than in FP16.

## Sources

- Paper: *Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference* — Elhoushi et al., Cerebras Systems — arXiv:2609.05275 (2026) — establishes the layer-dropout backbone that makes self-speculative decoding work.
- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — the classical two-model speculative decoding baseline.

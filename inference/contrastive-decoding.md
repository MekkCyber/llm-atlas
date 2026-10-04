# Contrastive Decoding
*Depth — bias token selection by contrasting a strong model's logits against a weaker reference.*

**TL;DR:** **Contrastive decoding (CD)** selects the next token based on `log p_strong - λ · log p_weak`, boosting tokens the strong model prefers *more than* a weaker reference. Classical CD uses a smaller model (or earlier checkpoint) as the "weak expert." The **LoopCD** variant for looped transformers uses the *same* model at an earlier recurrent loop as the weak expert — contrastive decoding at zero or near-zero extra cost.

**Prereqs:** [README.md](./README.md)
**Related:** [../architectures/looped-transformer.md](../architectures/looped-transformer.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Standard decoding picks the next token from $\arg\max_v \log p_{\text{strong}}(v)$ (greedy) or samples from it. **Contrastive decoding** modifies the scoring to:

$$
\text{score}(v) \;=\; \log p_{\text{strong}}(v \mid \text{context}) \;-\; \lambda \cdot \log p_{\text{weak}}(v \mid \text{context})
$$

- Tokens the strong model prefers and the weak one doesn't score high.
- Tokens both models prefer (common-sense completions) score flat — the contrast cancels.
- Tokens the weak model is also sure about (surface-level habits) get *penalized*.

A plausibility constraint usually restricts the arg-max to tokens the strong model considers plausible, to avoid rewarding bizarre tokens the weak model also happens to dislike.

## How it works

### Classical CD (Li et al. 2022)

- `p_strong` = a large, capable model (e.g., Llama-70B).
- `p_weak` = a smaller sibling (e.g., Llama-7B) or an earlier checkpoint.
- Both run per decoding step; cost ≈ `p_strong` + `p_weak`.

### DoLa (Chuang 2023)

- `p_weak` = the same model's *earlier-layer* logits (via the LM head applied to a middle layer).
- Only one forward pass needed — the earlier layer is already computed.

### LoopCD (Liu et al. 2026)

For looped (recurrent) transformers, the strong signal is the final-loop prediction and the weak signal is an earlier loop's prediction.

- **LoopCD-Logits** — do one extra output pass at an earlier loop; cost ≈ 1 LM-head call extra.
- **LoopCD-Hidden** — subtract hidden-state representations before the LM head; zero output-pass overhead.

LoopCD reaches 73.33 pass@1 on AIME 2024 (Ouro-2.6B-Thinking, from 61.88) and 31.71 on HumanEval (Huginn, from 22.56), and lets looped models halve their recurrent depth at matched quality (22.5–48.2% FLOPs saved).

## Why it matters

- **Reasoning-task boosts without training.** CD improves open-ended generation and reasoning by biasing against the weak model's "safe but shallow" preferences.
- **Compatible with any decoding strategy.** CD is a scoring modification; combine with greedy, top-p, beam, or MBR as the sampler.
- **The weak expert is almost free in the right architectures.** LoopCD and DoLa reuse what the strong model has already computed, eliminating the second-model cost that held classical CD back from production.

## Gotchas & tricks

- **$\lambda$ is sensitive.** Too small → effect vanishes; too large → degenerate outputs. Typical range 0.3–1.0; tune per task.
- **Plausibility floor.** Without a `p_strong(v) ≥ α · max p_strong` constraint, CD can promote random low-probability tokens the weak model dislikes.
- **Expert mismatch bites.** The weak expert must be *weak at the same thing* the strong one is strong at. A weak model from a different family often doesn't contrast cleanly.
- **Decoder-only setting assumed.** CD for encoder-decoder architectures needs the weak expert aligned to the same decoder tokenizer.
- **Combine with length penalties.** CD tends to favor longer rarer completions; pair with a length normalizer if you need bounded outputs.

## Sources

- Paper: *Contrastive Decoding: Open-ended Text Generation as Optimization* — Li et al., 2022 — the original strong/weak CD recipe.
- Paper: *DoLa: Decoding by Contrasting Layers* — Chuang et al., 2023 — uses the same model's earlier layers as the weak expert.
- Paper: *Decoding Looped Transformers Better for (Almost) Free* — Liu et al., Apple, 2026 — [arXiv:2610.02185](https://arxiv.org/abs/2610.02185). The looped-transformer CD variant.

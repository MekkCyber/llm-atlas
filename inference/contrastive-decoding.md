# Contrastive Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** At each generation step, compute logits from a *strong* model and a *weak* model (or two depths of the same model), and sample from the difference. The weak model's probability on fluent-but-generic continuations cancels out of the strong model's, leaving a sharpened distribution that focuses on where the strong model is distinctively confident. Introduced by Li et al. (2022) for improving fluency+faithfulness; the modern frontier applies it inside recurrent-depth and looped transformers for free gains.

**Prereqs:** [README.md](README.md) (inference overview)
**Related:** [../architectures/looped-transformer.md](../architectures/looped-transformer.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Standard decoding samples from $p_{\text{strong}}(x_t | x_{<t})$. Contrastive decoding (CD) instead samples from a tempered log-ratio:

$$
\text{score}(x_t) = \log p_{\text{strong}}(x_t | x_{<t}) - \lambda \cdot \log p_{\text{weak}}(x_t | x_{<t})
$$

$\lambda \ge 0$ controls the contrast strength. Usually a mask is applied so that only tokens whose strong-model probability exceeds a threshold are candidates — otherwise the ratio blows up for rare tokens.

The weak model can be:

- A smaller model from the same family (classical CD).
- The same model with layers skipped (DoLa, 2024).
- An earlier recurrent pass of a looped / recurrent-depth transformer (LoopCD, 2026).

## How it works

For each decoding step:

1. Compute $p_{\text{strong}}$ in the usual way.
2. Compute $p_{\text{weak}}$. Classical CD pays a second forward pass; LoopCD-Hidden reads the hidden state of an earlier loop that's already been computed — zero extra forward cost. LoopCD-Logits pays one extra loop's worth of logits.
3. Apply a plausibility mask: $x_t$ is admissible only if $p_{\text{strong}}(x_t) \ge \alpha \cdot \max_v p_{\text{strong}}(v)$ for some $\alpha$ (e.g. 0.1).
4. Sample from the contrastive score, usually with top-$k$ or nucleus on the admissible set.

The intuition is subtractive: anything *both* models rate highly is generic. Anything the strong model rates highly *more than the weak one* is where the strong model adds value.

## Why it matters

- **Reasoning-depth free gains.** For looped transformers, intermediate loop states are weak predictions that are *already* paid for. LoopCD-Logits raises Ouro-2.6B-Thinking's AIME 2024 pass@1 from **61.88% → 73.33%**; LoopCD-Hidden raises Huginn's HumanEval pass@1 from **22.56% → 31.71%**.
- **Compute win.** With contrastive-boosted quality, you can halve the number of recurrent loops and match or exceed full-depth baselines, cutting forward FLOPs 22.5%–48.2%.
- **No training.** CD is a decoder-side change; no fine-tuning or re-training required.

## Gotchas & tricks

- **The plausibility mask is non-optional.** Without it, CD samples unbounded-probability-ratio tokens — nonsense that happens to score zero in both models.
- **Choice of weak model matters a lot.** The weak model must share enough with the strong one that their log-ratio is informative. Totally unrelated weaks can hurt.
- **Loop-based variants need recurrent architectures.** Standard GPT-style transformers don't produce aligned weak/strong pairs for free; you pay the second forward pass (or pre-compute via layer skipping).
- **Temperature interacts.** Contrastive scores can look like very low-entropy logits; sampling with a low $T$ on top collapses outputs. Tune $\lambda$ and $T$ together.

## Sources

- Paper: *Contrastive Decoding: Open-ended Text Generation as Optimization* — Li et al., 2022 — original formulation with a smaller model as the weak expert.
- Paper: *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models* — Chuang et al., 2024 — contrasts a later layer against an earlier one of the *same* model.
- Paper: *Decoding Looped Transformers Better for (Almost) Free (LoopCD)* — Liu, Zheng, Chen, Dilip, Bai, Jiao, Wang, Zhang; Apple, 2026 — [arXiv:2610.02185](https://arxiv.org/abs/2610.02185). Loop-based weak expert at zero (hidden) or one-loop (logits) extra cost.

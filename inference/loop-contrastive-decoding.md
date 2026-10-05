# LoopCD — Contrastive Decoding for Looped Transformers
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Looped transformers run a shared block N times; standard decoding only consumes the final loop's prediction. LoopCD contrasts the final prediction against an earlier recurrent pass to form a training-free contrastive-decoding signal — no auxiliary draft model, often zero extra output overhead, and the quality boost lets you halve the loops while matching full-depth baselines.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Liu et al. (Apple, 2026) observe that looped-transformer families (Ouro, Huginn) produce a *decodable* intermediate state at every loop. Standard decoding discards all but the final loop. Because earlier loops embody less computation, recurrence inherently supplies aligned weak-and-strong prediction pairs — perfect inputs to contrastive decoding, without needing an external smaller model.

## How it works

Two variants of the same idea.

**LoopCD-Logits** (one extra output pass):
- Decode an intermediate state from an early loop $k$ into logits $\ell_k$.
- Decode the final-loop state into logits $\ell_N$.
- The contrastive signal is $\ell_N - \alpha \ell_k$ for some $\alpha$; sample from the resulting distribution.

**LoopCD-Hidden** (zero extra output overhead):
- Contrast in **hidden-state space** rather than logit space — a subtraction / gating of hidden states before the single output projection.
- Avoids running the output head twice; cost is dominated by the recurrent loops themselves.

The contrast pushes the model to amplify tokens where additional recurrence *changed* the prediction (likely the ones the extra computation got right) and suppresses tokens the model would have produced early (likely rote patterns).

## Why it matters

- **Big quality gains at full depth.** LoopCD-Logits lifts Ouro-2.6B-Thinking's AIME 2024 pass@1 from **61.88% → 73.33%**. LoopCD-Hidden lifts Huginn's HumanEval pass@1 from **22.56% → 31.71%**.
- **Compute savings at matched quality.** With LoopCD, half the loops match or beat full-depth unguided baselines — **22.5%–48.2% FLOPs saved**.
- **No auxiliary model.** Classical contrastive decoding needs a smaller "amateur" model; LoopCD gets its amateur branch from the same model's early loops.
- Delivers a reason to adopt looped-transformer architectures beyond parameter-efficiency: free, high-quality contrastive-decoding support is built in.

## Gotchas & tricks

- Choice of the "early loop" $k$ matters — too early and the amateur is uninformative; too late and the contrast is weak. The paper reports $k \approx N/2$ as a decent default.
- Only applies to architectures that produce a decodable state per loop; standard transformers have one output.
- LoopCD-Hidden is essentially free compute-wise but a bit less interpretable than the logit-space version.
- Interacts with speculative decoding — both exploit a weaker auxiliary signal, but LoopCD doesn't do verification/rollback; it's a pure sampler modifier.

## Sources

- Paper: *Decoding Looped Transformers Better for (Almost) Free* — Liu, Zheng, Chen, Dilip, Bai, Jiao, Wang, Zhang; Apple, 2026 — [arXiv:2610.02185](https://arxiv.org/abs/2610.02185).
- Context: *Contrastive Decoding* — Li et al., 2022 — the external-amateur ancestor LoopCD internalises.
- Context: *Ouro / Huginn* — looped-transformer families on which LoopCD is evaluated.

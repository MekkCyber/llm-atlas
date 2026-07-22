# Token-Level Off-Policy Labeling (TOPL)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Off-policy post-training methods (SFT-on-collected-trajectories, DPO on off-policy pairs) suffer under distribution shift: forcing the model to *generate* tokens it wouldn't naturally produce drags it into distribution-shifted territory and degrades faithfulness. TOPL reframes the problem: instead of training the model to *produce* off-policy tokens, train it to *label* each of its own candidate tokens as "good" or "bad." That token-level classification signal implicitly steers generation without ever asking the model to imitate off-policy tokens.

**Prereqs:** [dpo.md](./dpo.md), [_rl.md](./_rl.md)
**Related:** [grpo.md](./grpo.md) · [_post-training.md](./_post-training.md) · [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

A LoRA-based training paradigm that turns post-training into a per-token binary correctness classification. The LoRA adapters trained this way double as steering vectors and interpretable classification heads at inference time.

## How it works

Given an off-policy trajectory $(x, y)$ with per-token correctness labels $c_t \in \{0, 1\}$ (obtained from a verifier, judge, or human annotation):

1. Attach LoRA adapters to the base model, keeping the base frozen.
2. Train the LoRA'd model to *classify* whether each token in $y$ is correct given the prefix — i.e. output a per-token score aligned with $c_t$, not to reproduce $y$ token by token.
3. At inference, the LoRA acts as a linear classifier / steering direction on the base model's activations, biasing generation toward tokens that would classify as "good."

The critical bit: the loss is a per-token classification loss, *not* a per-token language-modelling loss on off-policy tokens. That's what avoids distribution shift.

## Why it matters

Every off-policy post-training method (DPO, IPO, SimPO, plain SFT on collected trajectories) fights the same distribution shift. Token-level correctness prediction sidesteps the issue by never asking the model to generate off-policy content — only to *judge* it. The learned adapters are also inspectable: because the training signal is a linear classification head, you can literally read off which activation directions the LoRA discovered as "good vs bad token" indicators.

On document summarization, TOPL beats sequence-level and token-level baselines across 11 OOD datasets; the technique transfers to machine translation without task-specific tuning.

## Gotchas & tricks

- The paper explicitly ablates sequence-level analogues (mark the whole response bad/good) — they do *not* work. Token granularity is load-bearing.
- Getting per-token correctness labels is the hard part. The paper uses summarization-specific verifiers; general applicability depends on availability of a token-level oracle.
- The LoRA-as-steering-vector story is nice for interpretability, but it means the technique inherits LoRA's usual capacity limits — very complex behaviour changes may saturate.

## Sources

- Paper: *Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift* — Zitong Huang, Gustavo Lucas Carvalho, Deqing Fu, Robin Jia (University of Southern California), 2026 — [arXiv:2607.17524](https://arxiv.org/abs/2607.17524) · [HF](https://huggingface.co/papers/2607.17524)

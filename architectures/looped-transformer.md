# Looped Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **looped Transformer** applies the same block-stack multiple times per forward pass — an alternative scaling axis to width and depth. Historical result: an N-loop model is beaten by an N×-wider unrolled model at the same compute, which killed the design for years. Loopie (2026) is the first recipe that closes the loop-vs-unroll gap by combining looping with fine-grained MoE and a training curriculum that keeps looped depth genuinely useful.

**Prereqs:** [transformer-block.md](transformer-block.md), [_moe.md](_moe.md)
**Related:** [hyper-connections.md](hyper-connections.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Given a stack of blocks $B_1, \ldots, B_L$, a looped Transformer executes $\underbrace{(B_1 \circ \cdots \circ B_L)}_\text{unit block} \circ \cdots \circ (B_1 \circ \cdots \circ B_L)$ for $N$ loops. The parameter count stays the same as a single stack, but the effective compute per token is $N \times$ that of the unlooped baseline. Universal Transformers (2018) were the earliest well-known instance; adaptive-depth variants (PonderNet, ALBERT with weight sharing) sit in the same family.

The persistent finding until Loopie: at the same compute budget, a *wider* model with $N \times$ the parameters beat an $N$-looped model. Looping was more parameter-efficient, but the training dynamics under weight-sharing gave lower final quality than plain unrolling.

## How it works

Loopie's central design choices:

- **Sparse (MoE) unit block.** The looped block is not dense but a fine-grained MoE (20B total / 2B active; also 6B / 0.6B). Sharing weights across loops is much more attractive when those shared weights are heavily specialized experts routed per-token — different loops can hit different experts.
- **Training curriculum for looped depth.** Rather than fix $N$ at training time, use a schedule that starts with fewer loops and grows to full $N$, so the model learns to make each additional loop count. The paper's positioning is that this schedule is what breaks the historical "$N \times$ params always wins" pattern.
- **Post-training for reasoning.** A dedicated post-training pipeline builds long-horizon reasoning on top of the looped stack. At IMO 2025 and IPhO 2025, Loopie 20B-A2B reaches gold-medal performance without tools.

At inference, each token pays the same MoE forward-pass cost per loop, so per-token compute is $N \times$ single-loop. This is where looping pays for reasoning: adaptive per-token loops (early-exit) trade compute for quality at test time.

## Why it matters

Looping is a natural fit for reasoning: applying the same block to a working memory is compact "adaptive compute per token." If a training recipe genuinely closes the loop-vs-unroll gap, it opens a scaling axis that has been effectively dormant since Universal Transformers. Parameter-count-per-active-token is decoupled from effective depth, which changes deployment economics — you serve a small footprint but reason at effectively deeper stacks.

## Gotchas & tricks

- **Weight sharing multiplies training-loss curvature.** Same-weight blocks running $N$ times cause interference; MoE routing partially resolves this by giving each loop *effectively* different weights via different active experts.
- **Curriculum matters more than architecture.** Loopie's core argument is that most looped-Transformer failures were training-recipe failures, not architectural. Fixed-$N$ training under-utilizes the loop; curriculum-scheduled $N$ fixes it.
- **Compare against unrolled at matched active parameters** — not matched total parameters. Otherwise the comparison is unfair either way.
- **Adaptive-loop inference is the deployment story.** Fixed $N$ at inference wastes compute on easy tokens. An early-exit head per loop lets the model spend more loops on hard tokens.
- **Depth-vs-loops is task-dependent.** Reasoning benefits from looping; pure recall/completion tends to prefer unrolled depth.

## Sources

- Paper: *Loop the Loopies!* — Gao, Chen, Xiao, Yang, Tao, Zhou, Dai — IQuestLab, 2026 — [arXiv:2607.16051](https://arxiv.org/abs/2607.16051).
- Predecessor: *Universal Transformers* — Dehghani et al., 2018 — the earliest well-known looped Transformer.
- Related: *ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations* — Lan et al., 2019 — cross-layer weight sharing in a non-looped-inference setting.

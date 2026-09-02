# NoRA — Normalized Low-Rank Adaptation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A one-line fix to [lora.md](lora.md): normalize the down-projection matrix $A$ during training. Because LoRA initializes $B = 0$, all early optimization signal flows through $A$; normalizing it stabilizes those first steps, speeds convergence, and mitigates catastrophic forgetting. No extra trainable parameters, no inference cost. Even the "init-only" variant (normalize once at step 0, then standard LoRA thereafter) captures most of the gain.

**Prereqs:** [lora.md](lora.md), [../../architectures/_normalization.md](../../architectures/_normalization.md)
**Related:** [../../pre-training/_training-stability.md](../../pre-training/_training-stability.md)

---

## What it is

LoRA's asymmetric initialization — $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ — means the network output at step 0 equals the frozen base. But every gradient step routes $\partial L / \partial W$ through both branches:

$$
\nabla_A L = B^\top \cdot \nabla_{\Delta W} L, \quad \nabla_B L = \nabla_{\Delta W} L \cdot A^\top
$$

With $B = 0$, $\nabla_A L = 0$ for the very first step — so $A$ doesn't move on step 1. Meanwhile $\nabla_B L$ is scaled by $A$, whose row norms can vary wildly under a random Gaussian init. Early training conditioning is dominated by whatever $A$ happens to look like.

NoRA fixes this by keeping each row (or column) of $A$ on the unit sphere.

## How it works

At each optimizer step, after the standard AdamW update:

$$
A \leftarrow \frac{A}{\|A\|_{\text{row}}}
$$

(row-wise L2 normalization; column-wise is a documented variant). No new parameters, no altered forward pass, no extra memory beyond the normalization op itself.

Two schedules:
- **NoRA-full:** normalize on every optimizer step throughout training.
- **NoRA-init:** normalize once immediately after initialization, then run vanilla LoRA. Captures a large fraction of the benefit at literally zero training-time overhead.

## Why it matters

- **Universal upside across training regimes.** Reported gains hold across pretraining continuation, SFT, and RL fine-tuning — three phases with very different loss landscapes.
- **Zero-cost drop-in.** No new params, no inference change, no LR retuning. Fits inside existing PEFT stacks (peft, TRL) as a one-line change.
- **Reduces forgetting.** By constraining the early trajectory of $A$, NoRA keeps $\Delta W$ from swinging into pathological regions that overwrite base capability.
- **Init-only is nearly free.** Even without touching the training loop, seeding $A$ on the unit sphere improves outcomes — worth flipping on by default.

## Gotchas & tricks

- **Normalization axis matters.** Row-wise is what the paper reports; column-wise gives different early dynamics and is less studied. Match the paper's variant unless you have reason.
- **Interacts with the $\alpha / r$ scale.** Because norms are now constrained, the effective magnitude of $\Delta W$ is more predictable — you may find you can raise $\alpha$ vs vanilla LoRA.
- **Not the same as weight-normalized $A$.** NoRA hard-normalizes each step; weight normalization reparameterizes $A = g \cdot v / \|v\|$ with learned scalar gain $g$. NoRA does not add $g$.
- **Doesn't fix the "wrong matrices" bug.** If you're only applying LoRA to Q, V, NoRA won't rescue you — the problem was your rank/coverage, not init.
- **Verify with the init-only variant before adopting NoRA-full.** If init-only gives 90% of the improvement on your task, keep the training loop untouched.

## Sources

- Paper: *Normalized Low-Rank Adaptation* — Kang, Yue, Zhan, Huang, Liu — 2026 — arxiv.org/abs/2608.31036.

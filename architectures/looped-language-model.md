# Looped language model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **looped language model** applies the same transformer block (or block stack) recurrently at inference, with weight-shared recurrent depth as a run-time knob. Popescu et al. (2026) show that on compositional tool-calling benchmarks (API-Bank, BFCL, NESTful), recurrent depth reliably helps multi-step tool chains, while single-call API tasks see smaller and more model-dependent gains. Adaptive inference — spending extra loops only on inputs that need them — dominates static depth on the compute/quality frontier.

**Prereqs:** [transformer-block.md](./transformer-block.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../agents/README.md](../agents/README.md)

---

## What it is

Take a transformer with a shared block (or a short block stack). At inference, iterate that block $T$ times over the same input before emitting the next token. $T$ can be **fixed** (all inputs looped the same number of times) or **adaptive** (a small controller decides per token/prompt).

Trained end-to-end so the block learns to be *useful under iteration* — not just a stackable layer. The paper considers both:

- **Native looped models** trained from scratch with recurrent depth in the loss.
- **Retrofitted models** where a pretrained transformer is fine-tuned with weight sharing enforced across a subset of layers.

## How it works

- **Weight-shared recursion.** A single set of parameters $\Theta_{\text{loop}}$ implements the recurrent block. Iterating $T$ times is functionally equivalent to a $T$-layer transformer *with all layers tied*.
- **Depth-conditioned loss.** Training samples different $T$ per batch (or per token) so the block generalises across depths — otherwise it overfits to the training $T$.
- **Adaptive controller.** At inference, a small head predicts a stopping decision per token / step. When "keep looping" is chosen, the same block runs again; when "stop" is chosen, the next token is emitted. Halting-style objectives from ACT / Universal Transformers apply.

## Why it matters

- **Runtime depth as an efficiency knob.** Tool-use accuracy on API-Bank / BFCL / NESTful rises monotonically with $T$ on multi-step cases; single-step calls plateau early. Adaptive $T$ captures the multi-step gains without paying static max-depth cost on every prompt.
- **Alternative axis to token-level long-CoT.** Long-CoT RL scales reasoning depth by generating more tokens; looped models scale it by iterating the same block. The two levers combine.
- **Parameter-efficient depth.** Weight sharing means a "20-layer effective" looped model has the parameter count of a much shallower one — attractive for on-device agents.

## Gotchas & tricks

- **Non-looped models retrofitted badly.** Enforcing weight sharing after pretraining hurts base quality unless done gradually or on a subset of layers. Native looped training is easier.
- **Depth generalisation is fragile.** Train on $T \in \{2,4,8\}$ and ask for $T=32$ at inference and the model degrades. Vary $T$ over a wide range during training.
- **KV cache is per-depth.** Naive implementations recompute KV each loop; efficient inference needs per-depth KV reuse just like standard transformers reuse per-layer KV.
- **Adaptive controllers can under-halt.** If the halting head is under-trained, it defaults to always-loop-max and gains disappear behind compute cost.
- **Gains are use-case dependent.** Multi-step tool use, compositional reasoning: yes. Isolated API invocation and short QA: small or model-dependent.

## Sources

- Paper: *Looped Language Models Improve Compositional Tool Calling* — Popescu, Sáez de Ocáriz Borde, Liò (Cambridge), 2026 — [arXiv 2608.18171](https://arxiv.org/abs/2608.18171) — tool-calling study of native and retrofitted looped LMs, with adaptive inference results.
- Earlier reference: *Universal Transformers* — Dehghani et al., 2018 — the recurrent-transformer + halting framework that native looped LMs generalise.

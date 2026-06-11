# Token credit assignment for reasoning RL

*Depth — using interpretability signals to weight per-token RL gradients.*

**TL;DR:** Standard GRPO/RLVR assigns the same trajectory-level advantage to every token in a long CoT — but most of those tokens didn't actually contribute to the answer. FlowTracer (2026) runs attention-flow analysis backward from the answer span to extract a *reasoning backbone* — the tokens whose information flows into the final answer — and upweights gradient only on those tokens. An interpretability tool repurposed as an RL training signal.

**Prereqs:** [grpo](../grpo.md), [rlvr](../rlvr.md), [long-cot-rl](long-cot-rl.md)
**Related:** [prm](prm.md) · [token-level-trust-region](../token-level-trust-region.md) · [orm](orm.md)

---

## What it is

Long-CoT RL has a chronic credit-assignment problem: a 4000-token chain produces one binary reward at the end, and the RLVR update broadcasts that reward to every token (GRPO) or distributes it via a learned value baseline (PPO). Both spread the signal too thinly — most tokens were just punctuation or restatement, a handful actually decided the answer.

Process reward models ([PRM](prm.md)) attack this with a *learned* per-step quality model. FlowTracer attacks it differently: read off token importance from the network's *own attention graph* — no extra model, no extra labels — and use it as a mask on the RL gradient.

## How it works

1. **Forward pass on the trajectory.** Standard rollout produces a response with attention weights at every layer.
2. **Backward attention-flow analysis from the answer span.** For each token in the final answer, recursively trace which earlier tokens contributed via attention edges weighted by attention magnitude. The transitive closure of this trace is the *reasoning backbone*.
3. **Build a per-token mask $m_t \in [0, 1]$.** Backbone tokens get $m_t \approx 1$; pure-boilerplate tokens get $m_t \approx 0$. Smooth weighting, not binary.
4. **Apply to the GRPO/PPO loss:**
   $$ L^\text{targeted} = -\sum_t m_t \cdot A_t \cdot \log \pi_\theta(o_t \mid \ldots) $$
   Tokens off the backbone contribute little or no gradient.

The whole pipeline runs inside the same RL step — no separate model, no offline labelling.

## Why it matters

- **Cheaper than PRMs.** PRMs need their own training, their own data, and their own forward pass. FlowTracer reuses the policy's attention.
- **Complements (not competes with) trust-region work.** [CPPO](../token-level-trust-region.md) widens / narrows the *trust region* per token; FlowTracer scales the *gradient magnitude* per token. The two are orthogonal and stackable.
- **Faster convergence on math/code reasoning.** The paper reports both higher final reward and fewer optimization steps to reach it.
- **Interpretability-as-supervision.** A pattern likely to recur: tools we already use to *explain* model behaviour become signals for *training* model behaviour.

## Gotchas & tricks

- **Attention-flow ≠ causality.** Attention is a soft routing weight, not a guaranteed information pathway. Some tokens contribute through residual connections, MLP-mediated paths, or aggregated effects that simple flow tracing misses. FlowTracer is a *heuristic* importance signal.
- **Mask sharpness matters.** Too binary → only a tiny number of tokens get gradient → high-variance updates. Too smooth → recovers uniform GRPO. The paper picks a particular smoothing; reproductions should sweep it.
- **Doesn't address sparse-reward exploration.** Token-level credit assignment helps when the reward is correct; it doesn't manufacture rewards where there are none. Useful only after rollouts get *some* signal.
- **Adds attention-trace compute** that scales with sequence length squared. For very long CoTs the cost is real — the paper batches the tracing across rollouts.

## Sources

- Paper: *How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs* — Dong et al., SJTU / Alibaba / Shanghai AI Lab, 2026 — [arXiv 2606.10646](https://arxiv.org/abs/2606.10646).
- Background: *DeepSeekMath* — GRPO with uniform per-token advantage.
- Related: process reward models — see [prm](prm.md).

# S2L-PO — Small-to-Large Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a frozen *smaller* model from the same family as the rollout source during GRPO training of a larger model. Small models in the same family exhibit higher **policy-level diversity** (better pass@k as k grows) than their larger siblings — a temporally coherent, behaviorally distinct exploration signal that token-level temperature noise cannot match. A progressive annealing schedule shifts sampling from the small explorer to the large learner over training, avoiding the small model's capacity ceiling. Reported +8.8% on AIME 24 using a 1.7B explorer to train an 8B learner, with *less* rollout compute.

**Prereqs:** [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../rejection-sampling.md](../rejection-sampling.md), [../rl-prompt-curation.md](../rl-prompt-curation.md)

---

## What it is

GRPO needs diversity in the $G$ rollouts per prompt — that's where the group-relative baseline gets its signal. Standard practice cranks temperature / top-p to increase diversity, but token-level noise produces incoherent intermediate steps: a CoT that takes a high-entropy random walk through reasoning space rarely lands anywhere coherent.

S2L-PO uses a different diversity source: a frozen smaller checkpoint in the same model family (e.g. Qwen-1.7B as explorer for Qwen-8B learner). Different parameter count → different inductive biases → genuinely different *policies*, not just different token samples from the same policy.

---

## How it works

### Rollout sourcing

For each prompt $q$ in the RL batch, sample some rollouts from the frozen small explorer $\pi_S$ and some from the current large learner $\pi_L$:

$$
\{o_1, \ldots, o_{G_S}\} \sim \pi_S(\cdot | q), \quad \{o_{G_S+1}, \ldots, o_G\} \sim \pi_L(\cdot | q)
$$

Both groups are scored by the same reward function. Group-relative advantages are computed over the union.

### Importance correction for off-policy rollouts

$\pi_S$ rollouts are off-policy for $\pi_L$. The update applies a clipped importance ratio $\pi_L(o) / \pi_S(o)$ to the small-explorer rollouts in the PPO objective; this is standard machinery and the same clip $\epsilon = 0.2$ as GRPO is reported to work.

### Progressive annealing

Early in training, $G_S = G$ (all rollouts from small explorer). As training progresses, $G_S$ decays to 0 — the learner samples its own rollouts. The schedule is **progressive**: linear or cosine over the first ~70% of training, then pure on-policy. This avoids the failure mode where prolonged small-explorer sampling caps the learner's reasoning ceiling at the small model's.

---

## Why it matters

- **Coherent diversity is qualitatively different from temperature noise.** Pass@k for $\pi_S$ grows with $k$ faster than for $\pi_L$ because $\pi_S$ takes different *strategies*, not just different *wordings*.
- **Cheap.** Inference on a 1.7B model is much cheaper than on an 8B model, so the small-explorer rollouts cut total rollout compute even at the same $G$.
- **No new reward machinery.** Same verifier, same KL term, same group-relative baseline. Just a different sampling distribution.
- **Reframes "small models" in RL pipelines.** Beyond serving and distillation targets, small models are useful as *diversity sources* for training their larger siblings.

---

## Gotchas & tricks

- **Same model family is critical.** A small explorer from a different family (different tokenizer, different pretraining) shifts the support of $\pi_S$ too far from $\pi_L$; importance weights blow up and the clip kills the gradient.
- **Annealing schedule matters.** Hard cutover (small until step T, large after) is unstable — the learner suddenly sees a different rollout distribution and the value-tracking signal jumps. Linear/cosine is empirically smoother.
- **Pass@k advantage requires k > 1.** For single-rollout RL (REINFORCE-style), there's no group to compare; S2L-PO needs the group baseline to exploit policy-level diversity.
- **Importance-clip can mask the gain.** If the small and large policies disagree strongly at some tokens, the clip zeroes out exactly the most informative gradient. Use a slightly wider $\epsilon$ for small-explorer rollouts, or apply token-level masking only on out-of-clip positions.
- **Don't reuse the small explorer's RL checkpoint.** The diversity advantage holds for the *base* (or SFT) small model; an RL-trained small explorer collapses toward narrower distributions and loses the explorer property.

---

## Sources

- Paper: *Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO* — Xu et al., Tsinghua · Shanghai AI Lab, 2026 — [arXiv 2605.30789](https://arxiv.org/abs/2605.30789).
- See also: [../grpo.md](../grpo.md) for the group-relative baseline this extends, and [long-cot-rl.md](long-cot-rl.md) for the reasoning-RL setting where S2L-PO is reported.

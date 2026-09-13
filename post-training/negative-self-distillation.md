# Negative Self-Distillation (NSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy self-distillation (OPSD) trains a model to imitate its own correct rollouts and empirically *degrades* complex reasoning. NSD flips the sign: pick the model's own **flawed** generations, take a gradient **away** from them (positive KL), and use a per-token learned **gate** to restrict the negative signal to reasoning-critical tokens so grammar and fluency aren't damaged.

**Prereqs:** [_post-training.md](./_post-training.md), [_rl.md](./_rl.md)
**Related:** [dpo.md](./dpo.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [rlvr.md](./rlvr.md)

---

## What it is

OPSD's loss minimizes $\mathrm{KL}(\pi_\theta \| \pi_{\text{teacher}})$ on the model's own successful rollouts (a "self-teacher" using privileged info such as ground-truth solutions). Recent work shows this hurts hard-reasoning benchmarks — the model over-imitates surface patterns of correct traces and loses generalizing structure.

NSD asks the symmetric question: instead of *pulling toward* correct rollouts, *push away* from flawed ones. The target of the KL becomes an anti-teacher constructed from the model's own bad traces, so the update is a reasoning-flaw *repellent* rather than a positive imitator.

## How it works

For each prompt, sample a batch of on-policy rollouts and label them (via a verifier or ground-truth check) as "flawed." For each flawed rollout $o^-$:

$$
L_{\text{NSD}} = -\sum_t g_t \cdot \mathrm{KL}\!\big(\pi_\theta(\cdot \mid o^-_{<t}) \,\|\, \pi_{\theta_{\text{old}}}(\cdot \mid o^-_{<t})\big)
$$

The minus sign means we **maximize** the KL — the current policy is trained to disagree with its old flawed self at each position, keeping it away from that trajectory.

The per-token gate $g_t \in [0, 1]$ is a learned lightweight head over the same hidden state that decides which tokens are "reasoning-critical." Grammatical filler (articles, connectives, whitespace) gets $g_t \approx 0$ and receives no negative gradient; content tokens (numerical steps, logical connectives inside math/code) get $g_t \approx 1$. The gate is trained end-to-end.

A small KL to a reference policy $\pi_{\text{ref}}$ regularizes the negative signal so the policy doesn't drift into gibberish while running away from the flaw.

## Why it matters

- **Turns "polish the model with its rollouts" back into a working recipe** for hard reasoning, where standard OPSD was quietly regressing.
- **Cheap alternative to full reward-model RL** for surgical behavior correction — you don't need a preference dataset, only a way to label rollouts as good/bad (often a rule or verifier already available).
- **Gate is the safety valve.** Uniform negative KL over all tokens damages fluency; gating by token role makes negative distillation compatible with language quality. The same gate design should generalize to other repellent-style losses (unlearning, refusal-preservation).

## Gotchas & tricks

- **Without the gate, negative KL destroys fluency.** The paper's central ablation: gate off ⇒ language capabilities regress sharply; gate on ⇒ preserved.
- **KL penalty schedule matters.** Too aggressive a KL and the model overshoots away from *any* pattern that resembles the flawed trace, including good ones; the paper reports a distributional analysis of KL penalty vs. gate active ratio.
- **Requires a labeling signal for flawed rollouts** (verifier, unit tests, ground-truth answer). Not a pure self-improvement algorithm — needs the same labeler that RLVR uses.
- **Complementary, not a replacement, to positive rewards.** In practice, mix with a small positive-signal term to keep the model near correct trajectories; NSD alone is a repellent, not an attractor.
- **Symmetric intuition to DPO.** DPO's loss up-weights preferred and down-weights dispreferred completions simultaneously; NSD is the "dispreferred-only" limit with a per-token gate and on-policy rollouts.

## Sources

- Paper: *Negative Self-Distillation: Learning to Reason by Avoiding Flaws* — Pei, Wei, Xu, Zhu, Chen, Meng — UVA, 2026 — arXiv:2609.11699.

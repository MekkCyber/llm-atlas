# STARE — Surprisal-guided Token-level Advantage Reweighting

*Depth — a token-level patch to GRPO that prevents policy-entropy collapse during long RLVR runs.*

**TL;DR:** Vanilla GRPO assigns the same trajectory-level advantage to every token in a response, which over thousands of steps drives policy entropy to zero — exploration dies and the run silently degrades. STARE does a first-order gradient analysis showing per-token entropy change factors as `trajectory advantage × entropy-sensitivity over the next-token distribution`, identifies the **entropy-critical tokens** via batch-internal surprisal quantiles, **reweights** their effective advantage, and runs a target-entropy closed-loop gate so policy entropy stays in a chosen band over the entire run.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md), [_rewards](_rewards.md)

---

## What it is

A drop-in replacement for GRPO's per-token advantage assignment, plus an outer controller. STARE doesn't change rollouts, doesn't add a value network, doesn't touch the KL term — it just rewrites which token positions get credited with how much advantage, on the fly, to keep policy entropy from collapsing.

## How it works

Token-level entropy change under one GRPO update is, to first order:

$$\Delta H_t \approx A_i \cdot S(p_t)$$

where $A_i$ is the trajectory advantage of response $i$ and $S(p_t)$ is the **entropy-sensitivity** of the next-token distribution at position $t$ — a function of the surprisal $-\log p_t(o_t)$. This factorizes into an advantage × surprisal **four-quadrant** structure: high-advantage / high-surprisal tokens (recoveries) drive entropy up; high-advantage / low-surprisal tokens (confident wins) drive it down; etc. Near the criticality line, small reweightings have outsized control over $\Delta H$.

The recipe:

1. **Per-batch surprisal quantiles** identify entropy-critical token subsets (typically the top and bottom quantiles of $S(p_t)$ — the tokens most able to move entropy).
2. **Selective reweighting** scales each critical token's effective advantage by a learned factor that depends on the desired direction of motion.
3. **Closed-loop gate** measures the policy entropy each step and adjusts the reweighting factor so entropy tracks a target band (e.g. $H^\star \pm \delta$).

Implementation is a few extra lines on top of any GRPO trainer — no extra forward passes, no extra model.

## Why it matters

Policy entropy collapse is the single most common silent failure of long-horizon RLVR runs: by the time you notice (response length explodes uselessly, AIME scores plateau, acceptance rates drop), thousands of GPU-hours have already been spent on a dying policy. STARE turns entropy from "thing you hope stays alive" into a controlled variable. On AIME24 / AIME25, the difference is **4–8% average accuracy** over DAPO and other entropy-aware baselines, and the curves stay healthy for thousands of steps instead of dying around step ~500.

## Gotchas & tricks

- The target entropy band $H^\star$ is the new hyperparameter to tune. Too tight, the gate fights itself; too loose, you're back to vanilla GRPO failure modes.
- Surprisal quantiles are batch-internal — small batches make the quantile estimates noisy and the reweighting jittery.
- Works in both Short-CoT and Long-CoT settings (validated 1.5B–32B), but Multi-Turn Tool Use needs surprisal computed *per turn* rather than over the flat trajectory.
- Reflection-token count and response length grow *in tandem* under STARE — a sign of healthy exploration. Length growing without reflection growth means the gate is too loose.

## Sources

- Paper: *STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability* — Luo, Sun, Wu, Xu, Deng, Hu, Tang, 2026 — [arXiv:2606.19236](https://arxiv.org/abs/2606.19236).
- Code: https://github.com/hp-luo/STARE

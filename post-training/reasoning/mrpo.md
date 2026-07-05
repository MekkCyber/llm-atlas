# MRPO — Medical Reasoning-aware Policy Optimization

*Depth — step-wise process reward with an exponential early-step penalty, targeted at long-CoT failure cascades.*

**TL;DR:** In long chains of thought, a single wrong early step often propagates into a wrong final answer — an **error cascade**. Outcome-only RL gives one scalar per trajectory and can't localize the mistake. MRPO reshapes the process reward: for **failed** trajectories it assigns penalty proportional to $\exp(-\lambda \cdot \text{step\_index})$, so the earlier the invalid step, the larger the penalty. The policy learns to fix upstream errors first. On medical multimodal VQA, early-stage reasoning failures drop from **64.0% → 13.0%**.

**Prereqs:** [prm.md](prm.md), [../grpo.md](../grpo.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../rlvr.md](../rlvr.md), [../_rewards.md](../_rewards.md)

---

## What it is

MRPO is a **process-reward reshaping** that composes with standard PRM-style credit assignment. Instead of a uniform per-step penalty on invalid steps, it weights the penalty by an exponential decay in step position — early mistakes get the largest gradient signal. The idea is domain-general (it applies wherever reasoning errors cascade), but the paper evaluates it in medical multimodal reasoning where cascade dynamics are especially clean.

## How it works

### Step-wise credit assignment with early-step emphasis

For a failed trajectory with steps $s_1, s_2, \ldots, s_T$, identify invalid steps and assign

$$
\text{penalty}(s_i) = c \cdot \exp(-\lambda \cdot i) \quad \text{for each invalid step } s_i
$$

where $c$ is a base magnitude and $\lambda$ controls how sharply the emphasis decays. Step 1 receives the largest penalty; step $T$ a small one. Correct steps in a failed trajectory get no penalty. Successful trajectories get standard reward without the reshape.

### Why exponential and not linear

Linear decay smears the blame; exponential focuses gradient on the *first* faulty step, matching the cascade dynamic (fixing step 1 often makes step 2 correct without a separate signal). It also matches how clinical reasoning errors actually propagate — an initial mis-diagnosis dooms every subsequent inference.

### Integration with the RL loop

MRPO plugs into GRPO-style rollouts. Rewards flow the same way as with a standard PRM; only the per-step penalty for invalid steps in failed trajectories is reshaped. No changes to the KL, group-relative advantage, or PPO clip.

## Why it matters

- **Concrete, plug-in reshape for cascade-prone reasoning.** The exponential-early-step penalty is a two-line change on top of an existing PRM+GRPO setup and turns 64% early-stage failure into 13% on medical VQA.
- **Cross-domain transfer likely.** Nothing in the reshape is medical-specific. Any domain with cascade dynamics — legal reasoning, code generation with dependent steps, long-horizon planning — is a plausible testbed.
- **Aligned with how humans grade cascades.** A grader who sees "step 1 was wrong, everything after was doomed" doesn't equally blame all steps. MRPO codes that intuition into the objective.

## Gotchas & tricks

- **Depends on step-boundary quality.** MRPO inherits PRM's core weakness: what counts as a "step" must be well-defined. Newline-separated math steps are clean; free-form reasoning chunks are not.
- **$\lambda$ is task-sensitive.** Too large and only step 1 gets any signal; too small and it degenerates to uniform PRM. Grid-search on a held-out task.
- **Invalid-step labels come from somewhere.** Human labels, MC rollouts (Math-Shepherd-style), or a learned PRM — MRPO uses whatever the base PRM uses; it doesn't remove the label-source problem.
- **Reward hacking watch.** If the model discovers it can produce a superficially-correct step 1 that the PRM accepts even in failed trajectories, MRPO's front-loaded penalty is exactly the target. Combine with rule verifiers on final answers to catch this.
- **Successful trajectories still need reward.** MRPO only reshapes the failure-side penalty. Positive-reward shaping is unchanged; keep the outcome reward on the correct side.

## Sources

- Paper: *Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning* — Jung et al. (Korea U. / Upstage AI / KAIST / Hanyang / Kyung Hee), 2026 — [arXiv:2606.31825](https://arxiv.org/abs/2606.31825).
- Related: *Math-Shepherd* — Wang et al., 2024 — the PRM+PPO pipeline MRPO reshapes.

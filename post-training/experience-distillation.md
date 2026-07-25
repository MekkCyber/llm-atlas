# Experience Distillation
*Depth — internalize the gains of in-context-learning from an agent's own interaction history into model weights, without any new environment interaction.*

**TL;DR:** In-context learning (ICL) from an agent's trial-and-error history is extremely sample-efficient — the agent gets rapidly better inside a single session by reading its own past mistakes. The improvement vanishes when the context is dropped. **Experience Distillation** applies context distillation to those interaction histories: freeze the collected trajectories and distill the ICL-conditioned policy into the base weights, so gains persist after the context is removed and no fresh environment samples are needed.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [reopd](reopd.md), [../agents/README](../agents/README.md)

---

## What it is

A distillation loss whose teacher is the agent conditioned on its full ICL history, and whose student is the agent without that history:

$$\mathcal{L}(\theta) \;=\; \mathbb{E}_{q, \text{context}}\;\text{KL}\!\big(\pi_{\theta_0}(\cdot \mid q, \text{context}) \,\|\, \pi_\theta(\cdot \mid q)\big).$$

The "context" here is not human-written demonstrations — it's the trajectories the agent itself accumulated while interacting (successes and failures alike). Distilling from the ICL-conditioned policy transfers the in-context adaptation into the parameters.

## How it works

1. **Collect a small experience pool.** Let the agent interact with the environment (or with humans) and record the trajectories. This is the only source of environment cost.
2. **Form ICL-context prompts.** For each held-out task $q$, prepend a curated slice of the experience pool as in-context demonstrations.
3. **Distill.** Compute the teacher distribution $\pi_{\theta_0}(\cdot | q, \text{context})$ and the student distribution $\pi_\theta(\cdot | q)$ (no context), and minimize KL between them.

Critically, no *additional* environment interaction is required after step 1 — steps 2–3 are pure offline training against the frozen teacher's distribution.

The paper shows this preserves substantially more of ICL's gain than naive SFT on the same trajectories. SFT collapses because it forces the student to imitate specific trajectories; distillation lets the student inherit the ICL-induced *policy shift* even in states not covered by any trajectory.

## Why it matters

- **Retains ≥64.8% of ICL gains** across 749 SWE-Bench-style tasks and six text-adventure games, vs 3.8% for direct SFT on the same experience.
- **Matches classical RL baselines with ≥9.6× fewer environment samples.**
- **Complementary to on-policy RL.** RL keeps improving beyond the initial pool; experience distillation is a cheap way to bank the initial ICL gains without waiting for the RL loop to catch up.

## Gotchas & tricks

- **SFT on trajectories is not a substitute.** Cross-entropy on tokens teaches surface behaviors of *specific* trajectories; KL to the ICL-conditioned distribution transfers the underlying capability.
- **Context curation matters.** Selecting relevant slices of the experience pool (matched by task type) beats prepending everything. The context is the teacher — bad context, bad teacher.
- **Watch base-model drift.** Distillation moves the student off its base. Track eval on held-out tasks the experience pool doesn't cover; drift there is a red flag.
- **Small experience pools go far.** The whole point is sample efficiency — the pool doesn't need to be RL-scale. A few thousand well-curated trajectories often suffice.

## Sources

- Paper: *Sample-Efficient Learning from Agent Experience* — Gou, Tu, Fang, Cai, Rezatofighi, 2026 — [arXiv:2607.21051](https://arxiv.org/abs/2607.21051).

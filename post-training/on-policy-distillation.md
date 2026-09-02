# On-Policy Distillation (and OPSA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation samples responses from the *student* policy (rather than a fixed dataset), then trains the student to match the *teacher's* per-token distribution on those samples — a KL-with-teacher objective evaluated on student rollouts. Recent analysis argues the effective signal is *not* teacher knowledge transfer but a **negative advantage on low-probability tokens**, and shows a teacher-free variant (**OPSA**, On-Policy Self-Adaptation) with an entropy-adaptive negative advantage matches — and often beats — the teacher-guided version.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md) · [_rewards.md](_rewards.md)

---

## What it is

Two ideas, easily confused:

- **Offline distillation:** train student on a fixed corpus, minimizing $\mathrm{KL}(p_{\text{teacher}} \| p_{\text{student}})$ over each token. Student never sees its own generations.
- **On-policy distillation:** at each step, sample rollouts from the current student $\pi_{\theta}$, then compute the KL objective against a teacher $\pi_{\text{teacher}}$ *on those student-generated tokens*. Closer to RL than to supervised training — the training distribution is non-stationary and depends on the current policy.

Ding & Zhang (2026) analyze on-policy distillation as a special case of policy-gradient RL. The KL-with-teacher term factorizes into a signal that, in practice, mostly *pushes down* probability mass on tokens the student is over-generating — behaving like a negative advantage rather than a knowledge-transfer channel.

## How it works

### Standard on-policy distillation

At each RL step, sample $G$ responses from $\pi_\theta$ per prompt, then optimize:

$$
L = \mathbb{E}_{q, o \sim \pi_\theta}\!\left[ \sum_t \mathrm{KL}\!\left( \pi_{\text{teacher}}(\cdot \mid q, o_{<t}) \,\|\, \pi_\theta(\cdot \mid q, o_{<t}) \right) \right]
$$

Interpreted as policy gradient: the update reduces $\pi_\theta$'s mass on tokens where teacher disagrees.

### The OPSA reduction

Ding & Zhang show that:
1. Teacher signal contains substantial noise that *increases* with teacher size.
2. Student updates concentrate on tokens with low $\log \pi_\theta$; higher-probability tokens receive near-zero gradient.
3. Replacing the teacher-derived per-token weight with a fixed negative constant (only at low-probability tokens) matches student performance.

**OPSA** builds on this with an entropy-adaptive schedule: assign stronger negative advantages at *high-entropy positions* (where the student is uncertain and low-probability tokens are more likely to be spurious) and redistribute probability mass among frequent tokens.

$$
A_t = -\lambda \cdot H(\pi_\theta(\cdot \mid q, o_{<t})) \cdot \mathbb{1}[\log \pi_\theta(o_t) < \tau]
$$

Applied as a standard policy-gradient loss; no teacher forward pass, no reference model, no rewards.

## Why it matters

- **Reframes distillation as reward-shaping on entropy.** Puts on-policy distillation in the same taxonomy as [rlvr.md](rlvr.md) and [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md).
- **Removes the teacher cost.** Serving a large teacher during student RL is the dominant compute expense in KD pipelines. OPSA drops it entirely.
- **Empirical results on reasoning.** On Qwen3-1.7B: **+35.41 pts AIME24** (263% relative), **>2× Pass@32** across benchmarks, and **+16.77 pts AIME24 over on-policy KD**.
- **Diagnostic value.** Even if OPSA doesn't displace KD in your pipeline, the analysis tells you *what your KD is actually doing*: mostly suppressing low-probability tokens, not transferring the teacher's structured knowledge.

## Gotchas & tricks

- **Noise scales with teacher size.** Bigger teacher $\neq$ better distillation. If your KD is under-performing OPSA, check whether the teacher is adding usable signal at all.
- **Entropy threshold $\tau$ matters.** Too permissive and you're suppressing tokens the student legitimately needs; too strict and you get no signal. Paper reports a specific schedule — start there.
- **Not for early SFT.** Teacher-free negative-advantage training presumes the base student can already produce coherent responses. Cold-starting from a raw pretrained base with OPSA is not what the paper does.
- **Combine with a KL-to-reference term.** Without one, aggressive negative advantages can collapse the policy. Same discipline as GRPO.

## Sources

- Paper: *Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement* — Ding, Zhang — Purdue, 2026 — arxiv.org/abs/2608.31046.
- Predecessor: on-policy distillation lineage from Agarwal et al., *GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models* — 2023 — arxiv.org/abs/2306.13649.

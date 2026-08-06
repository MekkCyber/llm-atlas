# On-Policy Self-Distillation (OPSD)
*Depth — a dense-supervision pattern that turns a trajectory-level RL reward into token-level signal by asking a privileged-context teacher branch to score the student's own rollouts.*

**TL;DR:** In sparse-reward RL for LLMs, a single scalar per trajectory has to explain hundreds or thousands of token choices. OPSD keeps the student on-policy (sample from the current policy, no teacher rollouts) but adds token-level supervision by running the same model in a **teacher branch with privileged context** (ground-truth answer, retrieved skill, hindsight of what actually happened) and using its next-token distribution as a per-token target for a KL / cross-entropy distillation loss. The RL policy-gradient objective and the distillation objective are combined; the distillation term densifies the credit signal without changing what the policy is optimizing for. Two 2026-08-05 papers extend the pattern in different directions: **PCSD** replaces per-token weights with a *local-persistence* aggregate, and **TurnSight** derives the teacher signal from **execution hindsight at turn granularity** and filters it via cross-horizon agreement.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [reasoning/prm.md](./reasoning/prm.md)

---

## What it is

An augmentation on top of policy-gradient RL (GRPO / PPO / RLVR) that supplies **dense per-token supervision** without leaving the on-policy regime. Two distinguishing features vs plain distillation:

1. **Same-model teacher.** The teacher branch is the same network as the student — no external teacher, no capacity gap.
2. **Privileged context.** The teacher is given something the student cannot see at inference (ground truth, oracle CoT, retrieved skill, executed tool output). Its next-token distribution over the student's own rollout provides the token-level supervision.

The RL and distillation losses sum:

$$L = L_{\text{RL}} + \lambda \cdot L_{\text{distill}}$$

where $L_{\text{distill}}$ is a per-token KL (or cross-entropy) between the teacher's and student's distributions on the student-generated tokens.

## How it works

For each prompt in the RL batch:

1. **Sample rollouts.** Draw $G$ trajectories $\{o_1, \dots, o_G\}$ from the current student policy $\pi_\theta$. Compute per-trajectory reward $r_i$ as usual.
2. **Teacher pass.** Re-run each rollout through the *same model* under a privileged context $c^+$: same tokens, but the model gets to see $c^+$ (the ground-truth solution, an oracle rationale, the retrieved best-matching skill, or in TurnSight's case the *executed hindsight* of the tool call).
3. **Distillation signal.** At each position $t$ in trajectory $i$, compute the KL from teacher to student:
   $$L_{\text{distill}}(i, t) = w_{i,t} \cdot D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid o_{i, <t}, c^+) \,\|\, \pi_\theta(\cdot \mid o_{i, <t})\big)$$
   The per-token weight $w_{i,t}$ modulates trust in the teacher at that position.
4. **RL update.** Standard GRPO / PPO advantage update from the trajectory-level reward.
5. **Combine.** Backpropagate $L_{\text{RL}} + \lambda \cdot L_{\text{distill}}$ jointly.

The design axis is the **per-token weight $w_{i,t}$**:

- **Uniform** — trust the teacher everywhere. Simple; sensitive to positions where the privileged context misleads.
- **Isolated token-level discrepancy** — weight by how much the teacher disagrees with the student at position $t$. Sensitive to noise.
- **Shared step-level weight** — one weight for a whole step (turn); overlooks positional variation.
- **PCSD (persistent consistency)** — derive $w_{i,t}$ from how *persistently* the teacher favors the student's choice around position $t$; a local aggregate that filters noisy individual samples.
- **TurnSight (turn-level hindsight)** — the teacher is conditioned on execution hindsight, not oracle context; multiple hindsight views at different lookahead horizons must **agree in direction** for the signal to be used; the surviving signal is normalized across sibling rollouts and modulates the RL advantage without flipping its sign.

## Why it matters

- **Solves the sparse-reward credit-assignment problem for LLM RL.** Trajectory-level rewards give one scalar per hundreds of tokens; distillation supplies a token-level shape.
- **Stays on-policy.** Unlike offline distillation from a separate teacher, the student samples its own tokens — the distillation signal is a *shape correction* on the student's actual behavior, not a fresh dataset that changes what the model is doing.
- **Cheap compared to training a value network.** The teacher pass is a forward-only re-run of the student's rollout with different context; no separate model, no separate optimizer.
- **Composable with GRPO / PPO / RLVR.** OPSD is orthogonal to the policy-optimization choice — plugs into any of them via an added loss term.

## Gotchas & tricks

- **Privileged context choice is the main lever.** Ground-truth answer, retrieved skill, and executed hindsight give *very* different teacher signals. Verify that your $c^+$ meaningfully changes the teacher's next-token distribution vs the student's — otherwise the KL is nearly zero and OPSD does nothing.
- **Teacher can be wrong.** Especially in TurnSight-style hindsight settings, one lookahead horizon can be misleading — the cross-horizon agreement filter is what makes the signal reliable.
- **Weight design matters more than $\lambda$.** Papers spend more effort on *how* to weight per-token signals than on the overall distillation coefficient.
- **Don't flip the RL advantage sign.** TurnSight's design constraint — modulate the RL advantage's magnitude, don't flip its sign — is a general OPSD tip: the RL reward carries the actual optimization direction; the distillation signal only reshapes it.
- **Not the same as offline distillation from a bigger model.** OPSD's teacher = student; the trick is privileged *context*, not privileged *capacity*. Confusing the two leads to over-generous claims about what OPSD can do.

## Sources

- Paper: *PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning* — Lv et al., 2026 — [arXiv 2608.01837](https://arxiv.org/abs/2608.01837). Local-persistence per-token weighting for OPSD.
- Paper: *TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning* — 2026 — [arXiv 2608.04007](https://arxiv.org/abs/2608.04007). Turn-level hindsight teacher, cross-horizon agreement filter.
- Related: on-policy distillation in the CollectionLoRA line (2026-05-29 digest, paper 4) — a multi-teacher variant for diffusion LoRAs; conceptually adjacent.

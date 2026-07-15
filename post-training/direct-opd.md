# Direct On-Policy Distillation (Direct-OPD)
*Depth — reuse a small model's RL run to improve a stronger student without re-running RL.*

**TL;DR:** Run RL (e.g., [RLVR](rlvr.md) + [GRPO](grpo.md)) on a small, cheap model. Take its **pre-RL and post-RL checkpoints** as a "teacher pair," and treat the log-ratio between them as a **dense implicit reward** on the *stronger* student's own on-policy rollouts. The student never sees the sparse verifier; it learns from the *direction* the small RL run discovered. Cuts RL cost when scaling to bigger models.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md), [ppo.md](ppo.md), [_rl.md](_rl.md)
**Related:** [dpo.md](dpo.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [rejection-sampling.md](rejection-sampling.md), [_post-training.md](_post-training.md)

---

## What it is

Sparse-reward RL for reasoning (RLVR + GRPO) is dominated by **rollout cost**. Every step, the target model generates $K$ responses per prompt, each scored by a verifier. When you scale from a 1.5B target to a 32B target, per-step cost blows up even though the *information* recovered from each step (a binary reward) is unchanged.

Direct-OPD's premise: the useful thing an RL run produces is not the final policy but the **policy shift** $\Delta = \log(\pi_\text{RL} / \pi_\text{ref})$. That shift encodes "actions RL made more likely to succeed on the verifier." A stronger student can apply the same shift on its own generations — no verifier calls needed.

## How it works

### 1. Small-scale RL run

Train a small model $\pi^s$ with RLVR + GRPO on the target domain (math, code). Save both:

- $\pi_\text{ref}^s$: the small model *before* RL (SFT or base).
- $\pi_\text{RL}^s$: the small model *after* RL.

This is the only step that pays sparse-reward rollout costs.

### 2. Extract the implicit reward

For any student response $y$ to a prompt $x$, define an implicit reward from the teacher pair:

$$
\hat{r}(x, y) = \log \pi_\text{RL}^s(y \mid x) - \log \pi_\text{ref}^s(y \mid x)
$$

Same shape as a DPO reward, but built from a checkpoint pair rather than a preference model. Dense (per-token score aggregable to a response score) and cheap (two frozen forward passes).

### 3. On-policy distillation to the student

The stronger student $\pi_\theta^T$ generates its own rollouts on the same prompts. Optimize $\pi_\theta^T$ against $\hat{r}$ using a policy-gradient objective (GRPO-style, with $\hat{r}$ in place of the verifier reward). The student learns from *its own* trajectories, weighted by "would the RL-shifted small model have upweighted this?"

### 4. (Optional) Sequential composition

Multiple teacher pairs can be applied in sequence: one for math, one for code, one for format. Because the object being transferred is a shift, not an absolute policy, composition is well-defined.

## Why it matters

- **Weak-to-strong that actually strong-to-strong.** A student already stronger than the post-RL teacher can still gain, because it's inheriting a *direction*, not the teacher's absolute distribution. Distilling the teacher directly (vanilla on-policy distillation) hurts.
- **Amortizes RL cost across model sizes.** One expensive RL run on a small model produces a signal usable by many larger students. Cross-family transfer works too (teacher-family ≠ student-family).
- **Decouples exploration from update.** Rollouts (expensive, size-dependent) happen once on the small model. The student side is cheap on-policy distillation.

## Gotchas & tricks

- **Baseline matters.** Vanilla on-policy distillation of $\pi_\text{RL}^s$ can *hurt* strong students because it drags them toward the small model's absolute distribution. The delta form is what makes it safe.
- **The student needs its own rollouts.** Applying the reward off-policy (on the teacher's rollouts) reintroduces distribution mismatch and loses the benefit. Sample fresh from $\pi_\theta^T$.
- **KL anchor is still needed.** Same as any RL: add a KL to a reference (typically the student's SFT/base) so the implicit reward doesn't push the policy off-distribution.
- **Composition order can matter.** Sequential deltas are not perfectly commutative; empirically some orderings converge more stably. Prefer applying deltas whose domains don't overlap.
- **Signal quality is capped by the teacher pair.** If small-model RL didn't discover a useful direction (small model too weak, verifier too sparse), the delta carries no signal. Run headline evals on the *small* pair before spending student compute.

## Sources

- Paper: *Weak-to-Strong Generalization via Direct On-Policy Distillation* — Feng et al., SIA-Lab (Tsinghua AIR + ByteDance Seed), 2026 — arXiv:2607.05394. Project page: https://bytedtsinghua-sia.github.io/Direct-OPD/
- Related: *Direct Preference Optimization* — Rafailov et al., 2023 — the "log-ratio as reward" idea, applied to preference data instead of a checkpoint pair.
- Related: PUST — Fu et al., Shanghai AI Lab, 2026 — arXiv:2607.11505 — same-day companion; abstracts the pattern into a general "proxy → signal → transfer" framework. See [proxy-guided-signal-transfer.md](proxy-guided-signal-transfer.md).

# OPDVR — On-policy Distillation with Verifiable Reward
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RLVR gives task-level correctness but sparse feedback; on-policy distillation (OPD) gives dense token-level guidance but caps at the teacher. **OPDVR** combines them **without adding hyperparameters**: reformulate sampled-token OPD's implicit reward around trajectory correctness, then **ReLU-gate** it so correct trajectories get non-negative rewards and incorrect ones get non-positive rewards. The gated OPD becomes a proper RLVR method that composes with any policy-gradient algorithm (GRPO included). Introduced by Lin et al. 2026.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md)
**Related:** [_rl.md](_rl.md), [on-policy-distillation.md](on-policy-distillation.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Two families of dense-supervision-plus-correctness recipes existed before OPDVR:

- **Weighted combination** — sum an OPD loss and an RLVR loss with a mixing weight $\lambda$. Introduces a hyperparameter that must be tuned per-setting.
- **Heuristic switching** — do OPD until some threshold, then switch to RLVR. Introduces the threshold as another hyperparameter and creates a phase transition.

Both bolt the two objectives together. OPDVR instead **reforms OPD into RLVR at the loss level**, so a single objective carries both signals.

## How it works

### Rewrite sampled-token OPD as a policy-gradient method

Sampled-token OPD, in its standard form, minimizes a token-level KL to the teacher on tokens sampled from the student's own rollout. That KL has an implicit reward at each token: the log-ratio between teacher and student. OPDVR makes this explicit — the token-level implicit reward becomes the "reward" side of a policy-gradient method.

### ReLU-gate by trajectory correctness

Given the trajectory's binary verifiable reward $y \in \{0, 1\}$:

$$
r_t^{\text{OPDVR}} = \begin{cases}
\phantom{-}\mathrm{ReLU}(\,g_t\,) & \text{if } y = 1 \\
-\mathrm{ReLU}(-g_t\,) & \text{if } y = 0
\end{cases}
$$

where $g_t$ is the token-level implicit reward from OPD (log-ratio between teacher and student on the sampled token). The gate enforces sign alignment:

- Correct trajectories can only get non-negative rewards from the teacher.
- Incorrect trajectories can only get non-positive rewards from the teacher.

The teacher's distributional guidance is preserved (magnitudes still reflect $g_t$), but incorrect-trajectory teacher signals cannot push the policy toward the incorrect answer, and correct-trajectory teacher signals cannot penalize a right answer just because it deviated from the teacher's phrasing.

### It's a proper RLVR method

Because the gated $r_t^{\text{OPDVR}}$ is a per-token reward with correct sign alignment, standard policy gradient (or GRPO) can consume it directly. No new hyperparameter, no phase switch, no mixing weight.

## Why it matters

- **Removes a hyperparameter and a phase transition.** No $\lambda$ to tune, no threshold to pick — a real usability win over prior OPD-plus-RLVR hybrids.
- **Reveals what the two objectives share.** OPD and RLVR were framed as complementary. OPDVR shows they're the same *kind* of thing at the loss level; you can express dense supervision as sign-gated shaped rewards.
- **Same principle transfers.** Any dense-supervision + sparse-verified-reward combination can be reframed the same way: reformulate the dense signal as a per-token implicit reward, then sign-gate by the sparse verifier. Applicable to preference-model + RLVR hybrids, teacher-forced + RL, rubric + verifier combinations.

## Gotchas & tricks

- **Requires a differentiable teacher signal per token.** The reformulation depends on being able to compute the implicit reward $g_t$ per token. Trivial for KL-to-teacher; less trivial for black-box teachers (proprietary API models).
- **Correct-trajectory teacher noise.** If the teacher itself is noisy (e.g. a weaker distillation source), the correct-trajectory rewards can be near-zero, effectively degenerating to pure RLVR. Fine as a fallback, but the whole point was dense supervision — monitor $\mathbb E[\mathrm{ReLU}(g_t)]$ over training.
- **Verifier signal has to be reliable.** ReLU-gating hard-clips the wrong-sign contributions. A noisy verifier that marks a correct answer as incorrect nulls the teacher's supervision on that trajectory.
- **Group-relative composition.** When composed with GRPO, the gated per-token rewards are aggregated per response into the same scalar advantage GRPO would use — the token-level dense supervision then rides along in the gradient.

## Sources

- Paper: *On-policy Distillation with Verifiable Reward* — Lin et al., 2026 — introduces OPDVR. [arXiv:2608.24696](https://arxiv.org/abs/2608.24696). Code: [github.com/LeapLabTHU/OPDVR](https://github.com/LeapLabTHU/OPDVR).
- Related: *DeepSeekMath* (GRPO) — the policy-gradient backbone OPDVR composes with.
- Related: *On-Policy Self-Distillation in Diffusion Models* (Zhou et al., 2026) — same principle (turn scalar reward into dense supervision) applied to diffusion.

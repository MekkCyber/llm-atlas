# SPLICE: Learner-Conditioned Option Splicing
*Depth — repair all-fail GRPO groups by injecting short options into failed trajectories and training on return-weighted local targets.*

**TL;DR:** In long-horizon agentic RL (SWE-bench, browser, deep-research) most sampled trajectories fail early in training. When *every* trajectory in a GRPO group fails, the group-relative advantage is degenerate and the policy gets zero signal. **SPLICE** picks a splice point inside a failed trajectory, has a proposal model generate a small set of short options from that point, estimates each option's value using continuations from the *current* policy, and trains on a return-weighted local target. Reports **+2.8 to +9.8** points over GRPO on SWE-bench Verified and BrowseComp-Plus across two policy models, at lower proposal-model cost than full-trajectory guidance baselines.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md) · [on-policy-distillation](on-policy-distillation.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [../agents/README.md](../agents/README.md)

---

## What it is

GRPO ([grpo](grpo.md)) normalizes rewards within a group of $G$ rollouts per prompt. If all $G$ rewards are 0, the group-mean baseline equals every rollout's reward and the advantage $A_i = (r_i - \bar r)/\sigma_r$ is 0 or undefined. The policy's gradient for that prompt is also 0 — all-fail groups are *wasted compute*. For long-horizon agentic tasks this is the majority regime in early training.

SPLICE's response: don't throw the group away. Repair it locally. Instead of asking "which rollout was best overall" (there is no answer when all fail), ask "at some chosen point inside this failed rollout, which short continuation would have been best?" — and train on that.

## How it works

Given a failed trajectory $\tau = (s_0, a_0, s_1, \dots, s_T)$ with $r(\tau) = 0$:

1. **Pick a splice point $t$** inside $\tau$. The paper uses heuristics based on the trajectory's natural step boundaries (action boundaries for SWE-bench, tool-call boundaries for BrowseComp).
2. **Generate $K$ short options** from a *proposal model* starting at $s_t$: $\{ u^{(1)}, u^{(2)}, \dots, u^{(K)} \}$. These are short ($\ll T - t$ tokens), cheap to produce, and the proposal model is lighter than the policy.
3. **Evaluate each option** by having the *current* policy continue from $s_t \oplus u^{(k)}$ to a terminal state, and scoring with the environment verifier. This gives a scalar return $R^{(k)}$ per option.
4. **Compute an option-level advantage** relative to the group's own mean $\bar R$:

$$
A^{(k)} = R^{(k)} - \bar R
$$

5. **Update the policy on the option tokens**, weighted by $A^{(k)}$, with a standard PPO-clipped ratio and KL to $\pi_{\text{ref}}$.

The key design choice is that the *value estimate* comes from the current policy's own continuations (learner-conditioned), not the proposal model's — so the signal is calibrated to "which option would help *me* succeed from here," not "which option a stronger model would like."

## Why it matters

- **Recovers signal from all-fail groups.** The common early-training regime of long-horizon agents stops being a dead zone. SPLICE spends compute on the hardest prompts and returns learning signal; GRPO alone returns zero.
- **Pairs naturally with partial rollouts.** The splice point is a partial-rollout boundary. Infrastructure built for [../systems/partial-rollouts.md](../systems/partial-rollouts.md) already produces the hooks SPLICE needs.
- **Proposal model can be small.** Because the proposal is only generating short continuations (not whole trajectories), it can be a cheap smaller LM, not a frontier teacher. The paper's cost analysis puts SPLICE below full-trajectory guidance baselines.

## Gotchas & tricks

- **Splice-point choice is load-bearing.** Too early in the trajectory and every option "fails" for the same downstream reason. Too late and there's no room for the option to change the outcome. The paper's natural-boundary heuristic works for SWE-bench; other domains need re-tuning.
- **Learner-conditioned value is what makes it work.** Replacing the current policy's continuation with a teacher's continuation for the value estimate biases the update toward imitating the teacher — closer to on-policy distillation than to RL. Keep the learner in the inner loop.
- **Doesn't help well-mixed groups.** When some rollouts succeed and some fail, GRPO already has signal and SPLICE adds overhead. Route around SPLICE when $|\{i : r_i = 1\}| > 0$.
- **Reward hacking at the splice.** A short option that happens to satisfy a cheap intermediate check but derails the overall task will still get $R^{(k)} > 0$ if the terminal verifier is lax. Combine with robust verifier design (see [../safety/reward-hacking.md](../safety/reward-hacking.md)).

## Sources

- Paper: *When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL* — Liu, Huang, Wang (SJTU / SII / JD.com), 2026, arXiv 2611.34777.
- Related: [grpo](grpo.md) — the base algorithm SPLICE repairs.
- Related: [on-policy-distillation](on-policy-distillation.md) — companion recipe for long-horizon agents; different lever on the same cost problem.

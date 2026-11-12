# Matched-Compute Evaluation
*Depth — the methodology correction that re-ranks RLVR tricks when you count tokens instead of steps.*

**TL;DR:** Comparing RL post-training methods at equal **optimizer steps** systematically overrates methods that consume lots of generation tokens per step — like Monte-Carlo process rewards, which spend 91% of generated tokens on reward evaluation. The matched-compute protocol pins (tokens generated, GPU-hours) across all methods instead. Under that protocol, no dense-reward variant beats vanilla GRPO; the MC-reward method that looked strong at equal updates finishes 14.3 points behind at equal tokens.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md).
**Related:** [_rewards.md](./_rewards.md), [rl-prompt-curation.md](./rl-prompt-curation.md), [reasoning/prm.md](./reasoning/prm.md).

---

## What it is

A critique-by-methodology of the current pattern for validating RLVR improvements. Most published RLVR tricks (dense rewards, PRM-style shaping, step-level credit assignment) report wins at **equal update count**, which hides the generation-token cost of the reward itself. The matched-compute protocol treats generation tokens and wall-clock GPU-hours as first-class budgets and forces every method to spend the same amount of each.

The authors run the protocol as a **preregistered** study with 1,668 runs across six base models.

## How it works

Three steps:

1. **Token accounting.** Count every generated token — rollouts *and* reward-evaluation continuations. For Monte-Carlo rewards that sample k continuations per intermediate state, those continuations count. (In their setup: 91% of generated tokens are reward-evaluation continuations.)
2. **Match budgets, not steps.** Fix a total-tokens budget and a GPU-hour budget. Each method is allowed as many optimizer steps as fit under those budgets.
3. **Preregister the metric and the comparison.** Lock in the evaluation set, the significance test, and the budget before running — so the matched-compute plot can't be re-chosen after the fact.

An independent per-token gradient test correctly orders methods but systematically overestimates each one's downstream gain by 1–2 points; the matched-compute number is what actually predicts final evaluation.

## Why it matters

- **Several widely cited RLVR gains disappear** when compute is matched. Dense process rewards, step-level credit assignment, and MC-style shaping all tie or lose against plain GRPO at equal tokens.
- **Replicates one honest effect: prompt curation at predicted-success ≈ 1/2.** +1.7 on Qwen3-0.6B, +0.8 on Qwen2.5-1.5B at matched tokens — a small but real gain.
- **Reveals how undercounted LR warmup / initial-LR tuning is.** Doubling vanilla GRPO's initial LR comes within half a point of the curation gain — a reminder that "known baselines, better tuned" is a competitive comparator.
- **Sets a standard of practice.** New RLVR techniques should now be expected to carry a matched-compute table; a win at equal updates is no longer sufficient evidence.

## Gotchas & tricks

- **Generation tokens, not training tokens.** Backward-pass FLOPs are negligible next to rollout + reward-evaluation FLOPs for modern RLVR. Matching GPU-hours alone is nearly equivalent, but tokens are easier to report and reproduce.
- **Preregistration is the enforcement mechanism.** Without it, matched-compute comparisons still invite p-hacking (choose eval set after observing runs). The 1,668-run study exists because the registered-protocol choices were costly.
- **Doesn't argue dense rewards are useless everywhere** — only in the token-rich LLM RLVR regime where every evaluation continuation competes for the same generation budget. On classic small-policy RL where rewards are cheap, dense rewards still help.
- **Rank-vs-magnitude discrepancy.** The per-token gradient test ranks methods correctly but inflates predicted gains. Use it to triage, not to publish numbers.

## Sources

- Paper: *Are Dense Rewards Worth Their Tokens in GRPO?* — Singh, Chen, Ganguly et al., Case Western Reserve, 2026 — [arXiv:2611.13560](https://arxiv.org/abs/2611.13560).

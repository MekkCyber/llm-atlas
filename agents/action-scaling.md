# Action Scaling (test-time compute at the harness boundary)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For agents that act on an environment (terminal, browser, API), a bad action can poison every future step. **Action scaling** allocates extra test-time compute at each decision point: sample $K$ candidate actions, verify them, forward only the best one for execution. Keeps generator and harness unchanged; the verifier can be the same model doing pairwise comparisons. Mid-Harness (Kang et al., NVIDIA 2026) shows this beats scaling trajectories per unit token cost.

**Prereqs:** [_self-evolving-agents.md](./_self-evolving-agents.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md).
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md).

---

## What it is

Terminal / tool-using agents produce **actions** that mutate state. If the agent generates `rm -rf /tmp/.cache`, the harness runs it — and if that was a mistake, the next step inherits a broken env. Standard test-time scaling (more trajectories, higher temperature) pays full execution cost each time, with no mechanism to catch bad actions before they commit.

Action scaling inserts a verify-before-execute gate at every decision point.

---

## How it works

Per decision point:

1. **Sample.** Generate $K$ candidate actions from the agent (same prompt / state, higher temperature or diverse decoding).
2. **Verify.** A verifier scores each candidate. In Mid-Harness, this is done either by a *stronger* model acting as pairwise judge, or by the *same* generator model doing pairwise comparisons on itself.
3. **Execute.** The harness runs only the top-scoring candidate.

Verifier mechanisms tested:
- **Pointwise score** (least effective at generator ≈ verifier).
- **Pairwise** (strongest among in-family verifiers — flipping between the two candidates reveals preference).
- **Listwise** (expensive, modest gains).

The verifier doesn't need to be stronger than the generator — pairwise self-verification extracts most of the gain. If a *stronger* verifier is available, distillation of its judgments back into the generator keeps the improvement when the verifier is removed at inference.

---

## Why it matters

- **TerminalBench-Lite:** GPT-5.6 verifier takes TMAX-9B Pass@1 from 50.00% → 68.03% with $K=8$.
- **Pairwise self-verification is the sweet spot** for in-family deployment: no larger verifier required.
- **Cheaper than trajectory scaling.** Combining action and trajectory scaling reaches higher success at *lower* estimated token cost than more trajectories alone, because fewer wasted execution rounds.
- **No training changes required.** Mid-Harness treats generator and harness as fixed and only adds the gate.

---

## Gotchas & tricks

- **Weak verifier → no benefit.** With bad pointwise scoring, more action samples don't help (noisy ranking). Verify this before investing in large $K$.
- **$K$ scaling is sublinear.** Diminishing returns past $K \sim 8$ in reported experiments.
- **Pairwise cost grows quadratically.** For $K$ candidates, $O(K^2)$ pairwise comparisons. Use round-robin or tournament selection to cut.
- **Distillation from stronger verifier is the standard follow-up.** Serves the one-time cost of a frontier verifier, then runs cheap at inference.
- **Doesn't fix generation diversity collapse.** If the model always samples the same bad action, no verifier can help. Increase temperature or re-prompt.
- **Harness compatibility.** The verify step must happen before the action commits — needs a harness hook or a dry-run mode. Not all tool interfaces support this cleanly.

---

## Sources

- Paper: *Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents* — Kang, Hachiuma, Zhang, Radhakrishnan, Fu, Jiang, Liu, Hosseini-Asl, Dong, Wang, Lee, 2026. NVIDIA. arXiv:2609.39982.

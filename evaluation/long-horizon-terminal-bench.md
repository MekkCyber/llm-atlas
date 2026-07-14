# Long-Horizon-Terminal-Bench (LHTB)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agent benchmark that stresses **multi-hour terminal workflows** — end-to-end system administration, multi-step data engineering, debugging sessions — measured with a **dense reward-based grading** protocol that assigns partial credit at intermediate checkpoints. Frontier agents that saturate short-horizon terminal benchmarks fall off sharply on LHTB, and the dense-reward view reveals characteristic failure modes (drift, tool-loop, checkpoint regressions) invisible to binary graders.

**Prereqs:** none.
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../agents/README.md](../agents/README.md)

---

## What it is

Agent evaluation has settled into two families: **long-horizon, sparse-reward** (SWE-bench: does the patch pass tests?) and **short-horizon, dense-reward** (Terminal-Bench: did each isolated CLI puzzle succeed?). Both leave a hole in the middle: realistic terminal workflows are long (hundreds of tool calls, hours of wall-clock time) *and* have natural intermediate checkpoints (a directory got created, a service restarted, a migration applied) that a good grader should reward.

LHTB fills that hole. Tasks are realistic terminal workflows — system admin, multi-step data engineering, cross-tool debugging. Grading is done by **container-scoped scripted checks** at each checkpoint, not by an LLM judge. Every run gets a *trajectory-shaped* score, not just a binary pass/fail.

---

## How it works

### Task structure

Each task is a scripted environment (Docker container + optional external services) with:

- A **task specification** given as a natural-language goal plus any necessary setup context.
- A sequence of **checkpoint predicates** — small scripts that inspect the container's state and return a reward signal for reaching a specific intermediate milestone.
- A **final-state predicate** for the terminal success condition.

### Dense reward grading

At each checkpoint predicate satisfaction, the agent accrues partial credit. The final trajectory score is a weighted aggregate:

$$
R_{\text{traj}} = \sum_c w_c \cdot \mathbb{1}[\text{checkpoint } c \text{ satisfied}]
$$

The weighting can prioritize the terminal state (final success matters most) while still distinguishing a run that got 6/10 checkpoints from a run that got 1/10.

### Long-horizon stress

Tasks are calibrated to require hundreds of tool calls and hours of wall time — comparable to a real ops or data-engineering session. This exposes long-horizon-specific failure modes: context drift, tool loops, over-repair (undoing correct earlier work), and checkpoint regressions (satisfying then breaking a checkpoint).

### Comparison to related benchmarks

- **Terminal-Bench**: short-horizon, isolated CLI puzzles. LHTB is the long-horizon extension.
- **SWE-bench**: long-horizon but binary (test-suite pass). LHTB adds dense partial credit.
- **OS-World**: GUI-based OS tasks. LHTB is terminal-only, script-graded.

---

## Why it matters

- **Short-horizon terminal benchmarks are saturating.** Frontier agents already score high on Terminal-Bench. A harder, longer-horizon benchmark is needed to distinguish top models.
- **Dense reward is the natural signal for RLVR.** RL post-training pipelines need per-step verifier signals to scale. LHTB's checkpoint predicates *are* that signal — the benchmark and the training recipe are made for each other.
- **Exposes failure modes binary graders miss.** Trajectory-shaped scoring reveals *how* agents fail (drift, loops, regressions), not just *whether* they fail. This is a diagnostic tool as much as an evaluation.
- **Scripted, not LLM-judged.** Container-scoped checks are reproducible and cheap to run; no LLM-judge variance in the leaderboard.

---

## Gotchas & tricks

- **Checkpoint predicate design is the hard part.** Bad checkpoints reward degenerate strategies (e.g. touching a file to "satisfy" a milestone without doing the actual work). The paper spends significant effort on adversarial predicate design.
- **Wall-clock evaluation is expensive.** Multi-hour tasks × many models × repeats is a real infra investment. Expect leaderboard entries with limited-N runs early on.
- **Not directly comparable to SWE-bench scores.** The dense-reward number is on a different scale; treat LHTB as complementary rather than a replacement.
- **Container-scoped grading rules out out-of-band tricks.** Agents can't shell out to a hosted service to solve the task and paste results — the grader only sees the container's state.
- **Deterministic environment vs realistic non-determinism trade-off.** Real terminal workflows sometimes involve retryable network calls, timing sensitivity, and human-in-the-loop. LHTB fixes seeds and stubs external services for reproducibility, which is the right choice for a benchmark but means agents trained on LHTB may still miss real-world flakiness.

---

## Sources

- Paper: *Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading* — 8-institution consortium (Tencent HY LLM Frontier, U. Maryland, U. Georgia, U. Minnesota, Indiana U., Lehigh U., NUS, PolyU) — [arXiv:2607.08964](https://arxiv.org/abs/2607.08964).
- Related benchmarks: Terminal-Bench (short-horizon CLI puzzles), SWE-bench (repo-level patches, binary grading), OS-World (GUI-based OS agents).

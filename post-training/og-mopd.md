# Observation-Gated MOPD (OG-MOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-teacher on-policy distillation for tool-using agents, with one addition: when a student's rollout emits a malformed tool call, OG-MOPD **replaces it mid-trajectory with a canonical call** and resumes distillation from the *repaired* observation. The student is never distilled on broken trajectories, so it doesn't learn to imitate teachers that would have called a different tool. Introduced in **Reframe** (2026) as the second stage after [gd-grpo](./gd-grpo.md).

**Prereqs:** [./on-policy-distillation.md](./on-policy-distillation.md), [./grpo.md](./grpo.md)
**Related:** [./gd-grpo.md](./gd-grpo.md), [./mismatch-budget.md](./mismatch-budget.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Classical multi-teacher on-policy distillation (MOPD): the student rolls out trajectories; each teacher scores or produces counter-trajectories; the student's loss is a weighted KL to teacher distributions. For agents, this breaks when a student's tool call is syntactically wrong or returns an unusable observation — the subsequent teacher distributions are conditioned on a *broken* observation, and distilling against them teaches the student the wrong lesson.

OG-MOPD plugs a **repair gate** into the rollout loop: at each tool call, if the call is malformed or the observation is unusable, replace with a canonical call *for the remainder of distillation only*, and resume.

## How it works

Each student rollout is a sequence of action/observation pairs $(a_1, o_1), (a_2, o_2), \ldots$. At step $t$:

1. Student emits $a_t$. If $a_t$ is invalid (unparseable JSON, wrong tool name, obviously malformed args) *or* $o_t$ is empty / an error, trigger the gate.
2. The gate substitutes $a_t$ with a **canonical action** (a simple default call for the current state — e.g. a bounding-box crop at image center) and re-executes to get a usable $\tilde o_t$.
3. The student's distillation target at step $t$ is the teacher distribution conditioned on $\tilde o_t$, **not** the original $o_t$.
4. The student's policy gradient / KL loss at step $t$ is still with respect to the student's *original* $a_t$ — so the student still learns "don't do what I just did".

The gate operates only inside the distillation pipeline; the student's own rollout policy continues to issue $a_t$, so the student remains the one being trained.

## Why it matters

- Fixes a quiet failure mode in agentic MOPD: compounding errors from early rollout steps poison later distillation signal.
- Decouples "did the student pick the right tool call" from "what should the student imitate next". The former is punished; the latter is set by the teacher on repaired observations.
- Paired with [gd-grpo](./gd-grpo.md), the recipe needs **no task-specific SFT cold start** — a notable result for visual tool-using agents where SFT was previously necessary.

## Gotchas & tricks

- **Choice of canonical action matters.** Too-good a canonical (an oracle tool) teaches the student unrealistic trajectories; too-weak a canonical (identity / no-op) collapses to vanilla MOPD. Reframe uses a simple center crop for vision tasks.
- **Gate trigger rules must be cheap and strict.** Running a parser on every student action is OK; running a teacher on every action is not.
- **Gate rate is a telemetry signal.** If the gate fires on most steps, the student's action policy is far from the teacher's and distillation is unlikely to converge — consider more SFT or a lower LR.
- **Reward attribution.** OG-MOPD by itself is a distillation recipe, not an RL recipe; combined with GD-GRPO in Reframe, the GRPO stage first grounds tool usage, then distillation sharpens teacher imitation on the repaired observations.

## Sources

- Paper: *Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning* — Li et al. (CASIA), 2026 — [arXiv:2611.12886](https://arxiv.org/abs/2611.12886)

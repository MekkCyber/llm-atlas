# Training Control Plane
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Decouple training *execution* from training *steering*. A trainer declares the knobs and actions it exposes (LR, data mix, restart-from-checkpoint, …); humans and automated controllers submit change requests through one shared protocol; the training loop validates and applies them at safe control points. Every request and outcome is logged, making mid-run interventions **auditable** rather than ad-hoc. Introduced in Interactive Training 2 (2026); demonstrated across five NLP and RL workflows.

**Prereqs:** [../pre-training/_training-stability.md](../pre-training/_training-stability.md)
**Related:** [../pre-training/mid-training.md](../pre-training/mid-training.md), [../pre-training/_lr-schedules.md](../pre-training/_lr-schedules.md)

---

## What it is

A shared protocol that sits between a training loop and the humans / agents that want to change its behavior mid-run. Existing experiment trackers (Wandb, Aim, MLflow) *show* training state; a control plane also *steers* it — LR schedule swaps, data-mix reweights, reward-model hot-swaps, checkpoint restores — all through typed requests, all recorded.

## How it works

- **Trainer declares its surface.** Each training application registers the set of parameters and actions it will accept from the control plane (`set_lr(float)`, `swap_data_mixture(spec)`, `restart_from(checkpoint_id)`, …).
- **Clients submit typed requests.** A human via a workspace UI or an automated controller (an agent that watches loss and decides to intervene) submits requests over the same interface.
- **Safe control points.** The training loop drains a queue of pending requests at explicit checkpoints, validates them, and applies them. Applying LR mid-microbatch is not allowed.
- **Audit log.** Every request, its validator verdict, and its outcome is written to a chronological log rendered next to live metrics.

## Why it matters

Modern LLM training runs increasingly require *in-flight* interventions — annealed data mixes, LR resets after a loss spike, hot-swapping a reward model mid-RL. Today each intervention is ad-hoc code, unaudited, unrepeatable. A shared protocol makes both reproducible human intervention and a whole class of *training-controller agents* possible — and, crucially, safe enough to deploy on production runs because the audit log makes changes reviewable after the fact.

## Gotchas & tricks

- **Validation ≠ safety.** The protocol validates types and ranges, not semantics. A "legal" LR of 1e-2 during late RL is still a disaster.
- **Safe control points are load-bearing.** Applying a data-mixture swap mid-step causes silent corruption if the microbatch was pre-fetched under the old mix.
- **Agent controllers are only as good as their observations.** An automated controller reading noisy per-step loss will chase its tail; expose smoothed metrics as its input, not raw ones.
- **Audit log is the deliverable.** If the log is truncated, undated, or ambiguous, the whole "auditable" property fails.

## Sources

- Paper: *Interactive Training 2: Auditable Control Plane for Live Model Training* — Zhang, Pan, Zhou, Lu, Deng (U. of Waterloo / U. of Wisconsin-Madison), 2026 — [arXiv:2607.18314](https://arxiv.org/abs/2607.18314).

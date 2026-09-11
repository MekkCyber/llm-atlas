# On-Policy Expert Correction

*Depth — teach a weaker agent from a stronger expert by rewriting only the failing turn in the weaker agent's own rollouts, preserving its planning style.*

**TL;DR:** When a weak model runs under an agent harness that was evolved around it, its rollouts fit the harness. Naively fine-tuning the weak model on the expert's rollouts under that same harness *regresses* performance (−4 to −30 points across enterprise agent benchmarks): the expert's plan doesn't fit the scaffolding evolved for the weak model. **On-policy expert correction** flips it: run the weak model, localize the specific turn where the rollout fails, and have the expert rewrite *only that turn*. The rest of the trajectory stays native. The corrected rollouts are the SFT signal.

**Prereqs:** [rejection-sampling](rejection-sampling.md), [../agents/agent-harness.md](../agents/agent-harness.md)
**Related:** [fine-tuning/README.md](fine-tuning/README.md), [rl-prompt-curation](rl-prompt-curation.md)

---

## What it is

Agent post-training has two levers — the model weights and the [harness](../agents/agent-harness.md). Evolving the harness with a weak model lifts weak-model performance; using a stronger expert on the *same evolved harness* lifts it further, suggesting the weak model has headroom to imitate.

But imitation-via-full-trajectory backfires. The paper's diagnosis: the weak model does acquire the expert's knowledge and makes greater use of the harness scaffold, but its *fit* to the harness is impaired — it adopts the expert's planning strategy without the competence to execute it, and no longer fits the scaffolding evolved around its native planning style.

On-policy expert correction is the surgical alternative: only replace what's actually broken.

## How it works

Pipeline, automated end-to-end by a meta-level MLE agent:

```
for each task:
    rollout_weak = weak_model.run_under_harness(task)
    if success(rollout_weak): continue
    t_fail       = locate_failing_turn(rollout_weak)      # meta-agent
    turn_fixed   = expert.rewrite_turn(rollout_weak, t_fail)
    rollout_new  = splice(rollout_weak, t_fail, turn_fixed)
    training_set.add(rollout_new)

weak_model = SFT(weak_model, training_set)
```

Key ingredients:

- **Failure localization.** The meta-agent scans the weak model's rollout for the earliest turn where the trajectory becomes unrecoverable, using downstream failure signals and the harness's own hooks.
- **Turn-level rewrite.** The expert regenerates a single turn given the exact context the weak model saw. All earlier turns (and the surrounding harness state) are the weak model's own.
- **SFT on spliced rollouts.** The training signal is a coherent trajectory that fits the weak-model harness *except* for the corrected turn.

## Why it matters

- **Recovers imitation's promise without breaking harness fit.** All 7 tasks improve under Qwen3-Coder and Gemma 4, versus regressions with full-trajectory imitation.
- **Cheap teaching signal.** One expert query per failing rollout, not one per turn.
- **Slots into a co-evolution loop.** Alternate harness-evolution with on-policy correction; each lever's gains stack.

## Gotchas & tricks

- **Failure-turn localization is the ML problem.** Getting the wrong turn gives the model the expert's fix in the wrong context. Use downstream signals (test failures, verifier outputs) to anchor.
- **Splicing must be self-consistent.** The corrected turn must not reference facts the weak model's earlier turns didn't establish; validate that the resulting trajectory type-checks.
- **The weak model still needs to reach the failure turn.** If it never gets that far in its own rollout, there is nothing to correct — pair with easier warm-up tasks or partial-rollout collection.
- **Not RL.** No advantages, no policy gradient — this is targeted SFT on repaired rollouts. Compose with downstream RL as usual.

## Sources

- Paper: *Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails* — Bi, Pentyala, Mehrotra, Chaudhuri, Bhagavath, Chen, Xu, Mui, Zhu, Asur — Salesforce AI, 2026 — [arXiv:2609.09134](https://arxiv.org/abs/2609.09134).

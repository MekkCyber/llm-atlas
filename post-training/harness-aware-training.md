# Harness-Aware Training (HAT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Production agent stacks pair a compact latency-friendly model with an **evolvable harness** (skills, hooks, prompts, tools updated independently of weights). Small models trained against a single fixed harness *overfit* to it and break when the harness inevitably changes in production. Harness-Aware Training solves this by treating the harness itself as an **environment axis** at training time: sample harness configurations, run SFT+RL under each, and train the model against the *distribution* of harnesses rather than a single fixed one.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [fine-tuning/README.md](fine-tuning/README.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md), [../agents/README.md](../agents/README.md), [../agents/live-self-improvement.md](../agents/live-self-improvement.md)

---

## What it is

A training recipe for the "small model + rich scaffold" pattern that most agent products actually ship. The scaffold — the *harness* — is the collection of prompts, tool definitions, hooks, and skills wrapped around the model at runtime. In production, product teams iterate on the harness weekly; the model weights are updated much less often. Any small model trained against one harness silently loses accuracy every time the harness is edited.

HAT names this failure mode (**harness overfit**) and gives a training-time fix.

## How it works

1. **Enumerate the harness-configuration space.** Prompt variants, tool schemas, tool-availability subsets, hook orderings, skill-library snapshots — anything the scaffold ships variability in.
2. **Sample harnesses per batch.** For each training example, sample a harness configuration $h \sim \mathcal{H}$. Format the input, tool descriptions, and system prompt under $h$.
3. **Two-phase training.** SFT on demonstrations run under sampled harnesses, then an RL phase (GRPO/RLVR-family) with rollouts also under sampled harnesses. Reward is the task outcome, not per-harness — the model is graded on being *right*, not on matching any particular harness's expected trace.
4. **Eval under harness drift.** Held-out evaluation uses harnesses the model never saw at training time. This is the metric that matters — same-harness eval hides overfit.

The distribution $\mathcal{H}$ is the design choice. Wider covers more future edits but dilutes signal; narrower keeps sharper task performance but returns to overfit.

## Why it matters

Cleanly separates two forms of adaptation: **weights** adapt slowly (retrain) and **harness** adapts fast (product iteration). A HAT-trained small model absorbs the fast-adaptation cost gracefully — prompts and tools can be swapped between deploys without retraining weights. That's the difference between a small model that stays useful for six months versus one that decays every product cycle.

Reported empirically as maintaining strong product-QA performance and generalization while meeting real-time latency targets that large-model configurations miss.

## Gotchas & tricks

- **Harness-space cardinality.** If $|\mathcal{H}|$ is tiny (a few templates), sampling is essentially fixed-harness — no robustness gain. If it's huge and unrealistic, you burn capacity on configurations that will never ship.
- **Correlated features across harnesses.** Sample so distinct-looking configurations still test distinct *behaviors* — otherwise the model learns a superficial harness invariance that doesn't generalize to the next product iteration.
- **RL reward independence.** Reward on task outcome, not on matching a golden trace — traces are harness-dependent by construction.
- **Held-out harness eval is the only trustworthy signal.** Training-harness metrics look great even when overfit is total.

## Sources

- Paper: *Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report* — Sun et al., 2026 — [arXiv:2608.15763](https://arxiv.org/abs/2608.15763)

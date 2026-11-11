# InfraRSI
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agentic harness that turns high-level commands like "make this training run faster" into **verified** distributed-LLM-training optimizations. Runs a five-stage loop — *init → doctor → run → report → apply* — that combines launch-configuration tuning with verified code patches, fits execution-time and memory models online, and transfers experience across GPU architectures. Across four dense and MoE workloads, up to **2.75× speedup** and mean MFU **17.4 % → 30.0 %**.

**Prereqs:** [./README.md](./README.md), [./dualpipe.md](./dualpipe.md)
**Related:** [./partial-rollouts.md](./partial-rollouts.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

Hand-tuning distributed training (choosing TP/PP/DP, activation checkpointing, bucket sizes, kernel flags, scheduling) is the hidden tax on every large run. InfraRSI reframes that work as a closed-loop optimization an agent can execute: measure the current run, propose a change, verify the change's predicted effect against a fitted cost model, apply if verified.

The "RSI" is literal — the agent re-reads its own prior patches and the fitted parameters from previous workloads to warm-start new ones instead of cold-searching the config space each time.

## How it works

The five stages run as explicit tool calls on the training job:

- **init** — snapshot the model/data config and probe the hardware (NICs, NVLink topology, GPU arch).
- **doctor** — run a short calibration pass, collect per-op timings and memory peaks, and fit per-component cost models ($T_\text{exec}$, $M_\text{peak}$ as functions of micro-batch, TP, PP, activation ckpt). Residuals from the fit drive what to patch first.
- **run** — execute the current config for a bounded number of steps and record ground-truth timings.
- **report** — compare measured vs predicted; flag the biggest residual and propose a patch (launch flag change, PP schedule change, or a code-level patch — e.g. swap an all-reduce for a reduce-scatter + all-gather).
- **apply** — commit the patch only if, in a short re-run, the measured delta matches the predicted delta within tolerance. Otherwise roll back.

**Experience transfer.** Fitted cost-model parameters from one workload seed the init stage of the next, amortizing the doctor cost across similar runs (same arch or same model family).

## Why it matters

- Nearly doubles mean MFU (17.4 % → 30.0 %) across four dense and MoE workloads with no per-workload human tuning.
- "Verified patches" matters more than raw speedup: the common failure of auto-tuners is applying a change whose predicted win doesn't materialize at scale. InfraRSI's two-run (predict / measure) gate is the point.
- Suggests that "achieve X MFU" is becoming an automatable step. Hand-tuning the launch is likely to retreat to the frontier of new architectures / new topologies.

## Gotchas & tricks

- **Cost-model residuals, not absolute values, drive the loop.** The absolute fit can be mediocre as long as residual structure is informative. Don't over-invest in the first-pass fit.
- **Budget the verify step.** Every patch pays a ~5–10 min re-measurement cost. For >50-patch runs, this dominates — batch low-risk patches and verify together.
- **Transfer across archs is not free.** H100 → B200 transfer works for shape-only params (TP, PP); kernel-level patches rarely survive. Keep a per-arch "do not transfer" list in the experience store.
- **MoE workloads reveal more gain than dense.** The default launch configs for MoE are further from optimal (all-to-all is hard to tune by hand), so the agent recovers more there.
- **Reproducibility.** The "apply" stage mutates code — treat the applied patches as first-class training artifacts and log them in the run metadata, or your reproductions won't match.

## Sources

- Paper: *InfraRSI: Harnessing Recursive Self-Improvement for Training Infrastructure* — Tian et al. (TokenRhythm / PKU / Tsinghua / Infinigence / SJTU), 2026 — [arXiv:2611.11558](https://arxiv.org/abs/2611.11558)

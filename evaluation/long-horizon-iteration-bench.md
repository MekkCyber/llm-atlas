# Long-horizon iteration benchmark (AutoLab)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** AutoLab (Xu et al., 2026) is a closed-loop iteration benchmark for frontier agents. Each task starts from a *correct but deliberately suboptimal* baseline; the agent has a strict wall-clock budget to improve it through repeated benchmark → edit → measure → decide cycles. Across 17 SOTA models, the dominant predictor of success is not initial-attempt quality but *persistence in repeatedly benchmarking and editing*. Claude Opus 4.6 stands out; most others terminate prematurely or exhaust budget without progress.

**Prereqs:** [README.md](README.md), [../agents/README.md](../agents/README.md)
**Related:** [production-agent-bench.md](production-agent-bench.md), [livecodebench.md](livecodebench.md)

---

## What it is

Most frontier evaluations measure single-shot output quality or short-horizon agent trajectories. Real engineering and research are *iterative*: propose a change, measure its effect, refine, repeat. AutoLab benchmarks this directly by giving the agent a working but suboptimal artifact and a budget, then scoring the *gap* the agent closes through iteration.

The benchmark covers four domains:

1. **System optimization** — performance tuning of complex software stacks.
2. **Puzzle & challenge** — algorithmic problems with open-ended improvement.
3. **Model development** — modify and improve a machine-learning training run.
4. **CUDA kernel optimization** — GPU kernel performance.

36 tasks total, expert-curated.

## How it works

1. **Baseline artifact.** Each task ships with a correct (functional) but suboptimal solution. The agent inherits it as a starting point.
2. **Wall-clock budget.** Strict per-task budget (e.g. hours of compute and tool-call calendar time). Budget *exhausts* if the agent runs out.
3. **Closed-loop API.** The agent can edit the artifact, run a benchmark, observe the measurement, and decide what to do next. The benchmark is the *signal*; the agent must actually use it.
4. **Scoring.** Final score is the relative improvement over baseline within budget. Early termination or budget exhaustion with no improvement scores zero.

## Why it matters

- **Time awareness is now measurable.** Most agent benchmarks don't punish a model for giving up; AutoLab does. This produces an empirical signal on a property (*persistence*) that the field has been hand-waving.
- **Iteration capability ≠ raw reasoning.** The headline finding — initial-attempt quality doesn't predict success — shifts where agent improvements should come from: iteration discipline, not better one-shots.
- **CUDA / model-dev domains stress real frontier skills.** These are domains where humans iterate constantly; AutoLab brings them under the LLM benchmark umbrella.

## Gotchas & tricks

- **Budget calibration.** Too generous and all models converge; too tight and only the best model registers. Per-domain tuning needed.
- **Benchmark-as-reward leak.** Agents that simply re-run the benchmark to read its source code could game the system; the harness must isolate the measurement code.
- **Improvement metric brittleness.** "Relative improvement" rewards small wins on already-good baselines; absolute targets per domain reduce that.
- **Model-specific termination heuristics.** Some models terminate early because of explicit "stop when uncertain" training; the benchmark inherently penalizes this conservatism.

## Sources

- Paper: *AutoLab: Can Frontier Models Solve Long-Horizon Auto Research and Engineering Tasks?* — Xu et al., 2026 — [arXiv:2606.05080](https://arxiv.org/abs/2606.05080).
- Related: production-agent-bench (RAMP) for a parallel runtime-grounded benchmark.

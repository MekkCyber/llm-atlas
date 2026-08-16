# Meta-Harness Optimizer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A learning loop above a long-horizon coding/design agent: the agent runs its harness, a **meta-optimizer** inspects rollout traces, then rewrites the harness (prompts, tool policies, control flow) so the next run is better. Aligned with human design priors and accumulates a reusable **DesignHarness** across model configurations. Introduced by AutoDesign (2026) on paper-to-poster generation with the PosterBench benchmark.

**Prereqs:** [../agents/README.md](README.md)
**Related:** [agent-harness-evolution.md](agent-harness-evolution.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Long-horizon agentic systems (design, coding, research) run for tens of minutes with hundreds of tool calls. Their harness — the scaffolding the model runs inside — is usually hand-authored and static. AutoDesign adds a second-level optimizer that treats the *harness itself* as the thing being trained on rollout feedback, keeping the base model frozen.

Where [agent-harness-evolution](agent-harness-evolution.md) uses a **population** and **selection**, the meta-harness optimizer uses a **single line of descent** with a **learned improver**: read rollout traces, propose a targeted harness edit, execute again, repeat. The recursive edits accumulate into a reusable DesignHarness that transfers across code-agent-model configurations.

## How it works

1. **Prior-aligned harness template.** Start from a harness structured around explicit human design priors for the task (for poster generation: layout, hierarchy, typography, discipline conventions).
2. **Rollout.** Run the code agent through the harness on a task, capturing tool-call transcripts, intermediate artifacts, and self-critique.
3. **Meta-optimizer proposes an edit.** A separate optimizer LLM reads the trace and proposes a *targeted* modification to the harness — usually a prompt refinement, a tool-selection rule change, or a new sub-routine — with the modification framed as filling a specific observed failure mode.
4. **Re-run and score.** The new harness is executed against a benchmark or held-out task and the improvement is measured. Successful edits are committed; regressive edits are rolled back.
5. **Accumulate.** After enough rounds, the harness is a substantive artifact (DesignHarness) that can be plugged into other code-agent-model configurations.

Compared to population-based evolution, this is cheaper per round (one variant, one rollout) but more path-dependent — it relies on the meta-optimizer's proposals being consistently useful.

## Why it matters

- **Turns the harness into a learned artifact.** Prompt/harness engineering becomes an optimizer-driven pipeline instead of hand-tuning.
- **Transferable across configurations.** On PosterBench, a learned DesignHarness lifted average score from **54.99 to 67.39 (+12.4)** across seven code-agent-model configs.
- **Cheap enough to be practical.** A fully autonomous run: 253 tool calls, 11 editing turns, 40 minutes, under \$3 — reaching conference-quality poster output.
- **Beats a closed-source baseline.** 78.32 on PosterBench Main Track vs 70.87 for Claude Design.

## Gotchas & tricks

- **The optimizer needs the *right* view of the trace.** Raw transcripts overwhelm the optimizer; task-specific rollups (which sub-routine burned tokens, which artifact failed which criterion) help.
- **Regressive edits are easy to miss.** Because there's no explicit preserve-and-extend contract like in evolution, a "good" edit for one task can silently hurt another — run on a small preserved suite each round.
- **Prior alignment is what makes it work on niche tasks.** For paper-to-poster, the harness bakes in typography and layout priors. Without task-specific priors, the meta-optimizer wanders.
- **Optimizer strength is a moving cost.** Stronger optimizers propose better edits but cost more per round. Cheaper optimizers work if the harness template starts strong.

## Sources

- Paper: *AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design* — Luo et al., 2026 — [arXiv:2608.13560](https://arxiv.org/abs/2608.13560).
- Related work: DarwinX (population-based sibling), see [agent-harness-evolution.md](agent-harness-evolution.md).

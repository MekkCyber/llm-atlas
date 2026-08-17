# Harness optimization

*Taxonomy — improving an agent's capability by changing its harness (prompts, tools, skills, control flow) with the underlying model frozen.*

**TL;DR:** An agent's task performance depends on both model weights *and* its harness — the prompts, tool schemas, skill library, and control flow around the model. Harness optimization is the family of techniques that leaves the weights alone and searches over the harness instead. It's much cheaper per iteration than fine-tuning, transfers across base models, and is the current cheapest way to move the frontier on tool-use and long-horizon-task benchmarks.

**Related taxonomies:** [_rl](../post-training/_rl.md), [_rewards](../post-training/_rewards.md)
**Depth files covered here:** [darwinx](darwinx.md) · [agent-skills](agent-skills.md) · [procedural-memory](procedural-memory.md)

---

## The problem

Fine-tuning a big agent model is expensive, slow, and sample-inefficient for small capability gaps; it also entangles the improvement with the specific base model. A large fraction of what limits an agent is *not* what the model knows but the scaffolding around it — the wrong prompt shape, a badly named tool, a control-flow bug, a missing skill. Iterating on that scaffolding directly, with the model treated as a black-box function, is a much tighter loop.

## The shared pattern

All variants follow the same shape:

1. **A frozen model** + a **structured harness** (prompts, tools, skills, retrieval, control flow).
2. **A verifier** that scores harness variants on a benchmark or environment — no gold trajectory required, only outcome verification.
3. **An edit operator** that proposes new harness variants from evidence (own failures, teacher feedback, self-critique, past experience).
4. **A selection rule** that decides which variants survive — greedy, evolutionary, monotonic, or memory-conditioned.

What changes between variants is the *unit* being optimized (whole harness / individual skill / cached lesson) and *how* survival is enforced (single-lineage greedy vs population + preserve-and-extend).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [darwinx](darwinx.md) | Evolutionary search over a population of harnesses, with preserve-and-extend to prevent regressions | Bigger compute per round; needs an archive infra | Long-horizon agent benchmarks where local edits regress other tasks |
| [agent-skills](agent-skills.md) | Package reusable procedural knowledge as a standardized *skill* format the harness can compose | Requires a skill schema; skills can rot | Repeatable multi-step tasks with reused sub-procedures |
| [procedural-memory](procedural-memory.md) | Retrieve distilled "lessons" from past verified rollouts to guide future ones; reliability-scored | No weight update = capability ceiling is base-model; retrieval hit rate matters | Adding narrow, capability-adjacent skills to a frozen VLM without any tool call |
| Single-lineage self-editing (no depth file) | One-agent-edits-itself loop with each round replacing the previous harness | Path-dependent; local wins regress other tasks | Fast iteration on a narrow task where cross-task regression is not a concern |
| Meta-harness optimization (no depth file yet) | Learn a *meta*-optimizer that rewrites the harness in response to critique + priors | Second-order signal is noisy; small critique models can miscritique | Creative long-horizon design tasks where per-input harness customization pays off |

## How to choose

**Default: single-lineage self-editing** when the task is narrow, the verifier is cheap, and you only care about one benchmark — it's the simplest thing that works and gets you the fastest wall-clock improvement.

Move to **evolutionary search (DarwinX-style)** the moment you care about more than one benchmark or the base model is expensive: population + preserve-and-extend costs more per round but stops local wins from regressing the rest, and the archive gives you recombination for free.

Use **agent skills** as the unit of edit when the harness accumulates repeatable sub-procedures (multi-step tool calls that appear across many tasks). Skills compose better than monolithic prompts, and swap in cleanly across base models.

Use **procedural memory** when the model is truly frozen and you cannot even change the system prompt permanently — retrieval-time injection is the least invasive of these techniques and stacks on top of any of the others.

These are not mutually exclusive: real systems typically combine an evolutionary search over prompts + a skill library the search can call + a procedural memory the skills read from.

## Adjacent but distinct

- [rl-prompt-curation](../post-training/rl-prompt-curation.md) — curates *training* prompts for RL; harness optimization edits the *deployment* prompt.
- [rlvr](../post-training/rlvr.md) — verifier-driven RL that updates *weights*. Harness optimization uses the same verifier signal but updates the harness.

## Sources

- *DarwinX: Evolving Agent Harnesses Through Natural Selection* — Zhang et al., 2026, [arXiv:2608.07545](https://arxiv.org/abs/2608.07545) — the evolutionary variant.
- *SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction* — Dang et al., 2026, [arXiv:2608.10538](https://arxiv.org/abs/2608.10538) — the agent-skills variant.
- *Spatial Memory Agent* — Zhang et al., 2026, [arXiv:2608.12743](https://arxiv.org/abs/2608.12743) — the procedural-memory variant.
- *AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design* — Luo et al., 2026, [arXiv:2608.13560](https://arxiv.org/abs/2608.13560) — the meta-harness variant.

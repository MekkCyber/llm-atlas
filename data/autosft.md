# AutoSFT
*Depth — multi-agent system for automated data research in supervised fine-tuning.*

**TL;DR:** Agents given free rein over an SFT pipeline mostly focus on data edits — and then fixate on one kind of edit. AutoSFT decomposes the agent into a meta-agent, an executor, four analysis subagents, and one synthesis subagent, with a shared vocabulary of six data actions (reweight, prune, repair, synthesize, mix, filter). Each round retrains from the base model, so gains attribute to the data change rather than accumulated fine-tunes.

**Prereqs:** [_data-curation.md](_data-curation.md), [quality-filtering.md](quality-filtering.md), [post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [deduplication.md](deduplication.md)

---

## What it is

An agentic data-iteration system with specialized sub-agents rather than one generalist. The authors first show that one generalist agent, over many rounds, picks one action class and plateaus. Decomposing by action class fixes the breadth problem.

## How it works

1. **Six data actions.** Reweight by category, prune low-quality rows, repair labeled rows, synthesize new examples, mix sources, filter by model-based score.
2. **Four analysis subagents.** Each specializes in diagnosing one bottleneck axis (coverage gaps, label noise, difficulty distribution, format drift) and proposes one or two of the six actions.
3. **Synthesis subagent.** Produces new rows targeting gaps the analysis subagents named.
4. **Executor.** Carries out the chosen actions against the dataset; retrains from base each round.
5. **Meta-agent.** Routes between analysis and synthesis, decides when to stop. Keeps a bounded history to avoid context explosion.

## Why it matters

Automated data research is the near-term lever most labs can pull without more compute. AutoSFT's result — *diverse actions > stronger single action* — is a direct recipe for teams building dataset-iteration pipelines: don't give the agent a bigger action, give it more actions.

## Gotchas & tricks

- Retraining from base every round costs compute but makes gains attributable. Teams that can't afford this will see noisier signals.
- Reported wins: analysis subagents +6.5 IFEval; synthesis +3.4 IFEval / +2.9 IFBench; +3.7 MBPP+ on code.
- Subagent decomposition brings its own prompt-engineering burden; the six-action vocabulary is the stable interface.

## Sources

- Paper: *AutoSFT: An Agentic System for Automated Data Research in Supervised Fine-Tuning* — Sun et al., UIUC / Tsinghua / ModelBest / CUHK / BUPT, 2026 — [arXiv:2610.22194](https://arxiv.org/abs/2610.22194)

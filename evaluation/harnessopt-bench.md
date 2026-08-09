# HarnessOpt-Bench

*Depth — benchmark for how well an LLM can iteratively optimize another LLM's *harness* — prompts, tools, control flow, memory, orchestration code — under a fixed evaluation budget.*

**TL;DR:** Agent quality depends not only on model weights but on the *harness* around them: prompts, tools, control flow, memory, and orchestration code. HarnessOpt-Bench (Scale AI, 2026) is a benchmark for the "AutoML for agents" problem — how well can a frontier LLM iteratively improve a harness under a bounded budget? Five models × four tasks × two harness formats (shared + native). Headline finding: **optimizer-model choice matters more than the coding framework** the harness is written in.

**Prereqs:** *(basic understanding of agent harnesses — prompts, tools, orchestration)*
**Related:** [README.md](README.md), [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark that treats "improve this agent harness so it scores higher on task T" as the top-level task. The optimizer is an LLM; its actions are edits to the harness (prompts, tool schemas, control-flow code, memory strategy). The evaluation budget bounds how many candidate harnesses the optimizer can try.

Distinguishes between two coding harness scaffolds:

- **Shared harness.** A common framework used across all optimizer models — separates optimizer capability from framework-specific advantages.
- **Native harness.** Each optimizer uses its own preferred agent framework — measures the combined optimizer+framework capability.

Four tasks × five models × two harness formats gives a matrix that isolates the optimizer's contribution.

---

## How it works

**Round loop.** The optimizer proposes a harness edit; the modified harness is evaluated on the target task; the score returns; the optimizer proposes the next edit. Budget = a fixed number of evaluation rounds.

**Optimizer state.** The optimizer sees the current harness, the history of edits, and the score of each. It must decide *which* experiment is worth spending a round on.

**Metrics.** Final task performance after the budget is spent; efficiency curves (score vs. rounds); ablation across shared vs. native harnesses.

**Result.** Optimizer-model choice creates larger performance spreads than the coding framework. That is, "which LLM is driving the AutoML loop" matters more than "which agent scaffold is being edited."

## Why it matters

- **Reframes the agent-frameworks debate.** The public conversation about DSPy vs. LangChain vs. Reflexion vs. bespoke code centers on *frameworks*; HarnessOpt-Bench says the framework matters less than the optimizer that edits it.
- **A concrete benchmark for meta-agent capability.** Prior "AutoML-for-agents" work (DSPy, TextGrad, self-refinement) evaluated on ad-hoc downstream tasks; HarnessOpt-Bench standardizes.
- **Direct signal for choosing an optimizer.** Anyone building an agent-optimization pipeline can look at the leaderboard and pick the model most efficient under their round budget.

## Gotchas & tricks

- **Budget shape matters.** A model that improves fast per round but plateaus is different from one that improves slowly but keeps climbing; report both integrated area and terminal score.
- **Shared vs. native harness confound.** Native-harness numbers combine LLM capability with framework fit; comparing raw native scores across models mixes two effects. The shared-harness split is the fairer optimizer-only comparison.
- **The optimizer can Goodhart on the eval.** If the evaluation is deterministic and the optimizer sees the score, it can overfit specific test cases. Blind or held-out evaluation splits are important.
- **Cost-per-round.** The most expensive optimizer is not always the best per dollar; report cost curves alongside quality curves.
- **Task diversity is limited.** Four tasks is enough to distinguish top from bottom but not enough to characterize per-domain strengths; expect follow-up benchmarks that broaden the task pool.

## Sources

- Paper: *HarnessOpt-Bench: Evaluating LLMs at Harness Optimization* — Shanker, Maurya, Yasser, Kalmath, Chatrath, Xue — Scale AI, 2026 — [arXiv:2608.06301](https://arxiv.org/abs/2608.06301).

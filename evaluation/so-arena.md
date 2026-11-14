# SO-Arena
*Depth — a benchmark for System-One (single-shot) LLM decisions that separates capability from reliability.*

**TL;DR:** A benchmark that evaluates LLMs as "System One" decision models — one-shot answers with no chain-of-thought — across 11 domains and five capability families. The novelty is that accuracy is only one of two axes: SO-Arena also measures **reliability** (decision stability under option-set perturbations and recognition of "no valid answer") and reports them independently. In the paper's 21-model sweep, the top-accuracy model places only **7th on reliability**, so the two axes genuinely disagree.

**Prereqs:** [ifeval](ifeval.md)
**Related:** [mmlu](mmlu.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A benchmark targeted at models used as fast, single-shot decision heads: an agent's action selector, a tool-router, a classifier. The question SO-Arena is built around is not "can the model get the right answer" but "can the model get the right answer *stably*" — because any downstream system that depends on it has to budget for both.

Three design choices distinguish it from static MCQ suites like MMLU:

1. **Evidence-grounded labels.** Reference labels are manually verified against inputs the model can see, not pulled from distributional noise in a web crawl.
2. **Domain-specific difficulty strata.** Each of the 11 domains has its own difficulty rubric; aggregate scores stratify by difficulty instead of averaging.
3. **Interactive sandboxes.** The model iteratively executes actions in a sandbox for subsets of tasks, so evaluation includes multi-turn tool use, not only a single reply.

## How it works

**Capability spectrum.** For each domain × difficulty cell, measure accuracy against evidence-verified labels. Reasoning-intensive tasks are included so the same prompts can be run in System-One (no CoT) or System-Two (with CoT) modes — the delta between the two scores quantifies "reasoning value" per task.

**Reliability.** Two complementary measures:

- *Decision stability under option-set changes.* For a multiple-choice decision, the option set is perturbed (reordered, dropped, or augmented with distractors). A reliable model's choice is invariant to perturbations that preserve the correct option.
- *Missing-answer recognition.* Inject "none of the above" distractors by removing the correct option entirely. A reliable model should recognize that no valid option is available instead of hallucinating one.

Both metrics score a *policy over option sets*, not a single prediction.

## Why it matters

Agentic systems increasingly use a small LLM as a fast decision head and reserve long-CoT models for the rare hard calls. If that head is unstable under trivial surface perturbations, the whole agent is unstable. SO-Arena puts a number on that risk and frames capability and reliability as separate training targets — the paper's "top-1 accuracy ≠ top-1 reliability" finding is a concrete call to train for stability, not just accuracy.

## Gotchas & tricks

- The "System One" framing is normative — you *could* run a reasoning model on SO-Arena, and the system-one/system-two comparison column is precisely how you'd measure what reasoning buys per task.
- Perturbations have to be structure-preserving. Shuffling options is safe; semantic rewrites aren't, because they mix capability and reliability back together.
- Low reliability does not have to mean low accuracy on perturbed sets — a model can be wrong-but-consistent, which is a different failure from accuracy loss and needs its own metric.

## Sources

- Paper: *SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models* — 2026 — https://arxiv.org/abs/2611.15777

# Harness Optimization (AutoSaddler)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM agents are wrapped in a **harness** — prompt scaffolding, tool schemas, control flow, retry logic — that turns a language model into a task-completing system. Harness design has been artisanal: hand-tune a giant prompt, hand-pick tool schemas, hand-write control logic. **AutoSaddler** formulates harness improvement as **offline learning from failure traces**: collect failures in mini-batches, diagnose them, generate structured code patches to the harness, and select updates via validation. Introduced by Park et al. 2026.

**Prereqs:** None (agents-cluster fundamentals)
**Related:** [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

An LLM agent's behavior depends on much more than its weights: system prompt, tool definitions, output format, retry policy, subagent orchestration, memory layout. This "harness" is the *actual* configuration deployed users run against. Improving the harness — often — matters more than improving the model.

But harness engineering has been prompt-tinkering. AutoSaddler treats the harness as a **program** and the failure log as a **training set**, and applies offline learning to it.

## How it works

### The three components

1. **Failure-trace diagnosis.** On a mini-batch of failed trajectories, an LLM-based diagnoser localizes the failure: a bad tool-schema string, an ambiguous system-prompt instruction, missing retry logic on a specific tool error, etc. Output is a structured diagnosis pointing at specific harness components.

2. **Structured patch generation.** Given the diagnosis, propose a patch. The harness is represented as **code** (a prompt template with parameter slots, a tool schema JSON, control-flow logic) rather than as a single monolithic prompt. Patches are code diffs: modify tool schema, edit a prompt section, add a control-flow branch.

3. **Validation-based update selection.** Roll the patched harness on a validation batch. Accept only if it improves the target metric without regressing others. Roll back if not.

### The mini-batch outer loop

```
for batch of failed trajectories:
    diagnoses = diagnose(batch, current_harness)
    for diagnosis:
        patch_candidates = generate_patches(diagnosis, current_harness)
        for patch in patch_candidates:
            new_harness = apply(current_harness, patch)
            if validate(new_harness) > validate(current_harness):
                current_harness = new_harness
```

Because updates are validation-gated, the harness only ever moves in improving directions. Because patches are code, edits are **durable** (unlike prompt-embedded few-shot injections that get diluted over long contexts) and **auditable** (a human can read the diff).

## Why it matters

- **+9–10 points on hard agent benchmarks.** GAIA2, SWE-Bench Pro, Terminal-Bench 2.0 all see ~9–10 percentage-point gains over the base harness, without touching the underlying model.
- **Turns harness engineering into an ML problem.** Everything the framing does — treating configuration as code, learning from failures, using validation to gate updates — is standard ML practice applied to a space that had none.
- **Complements weight-level improvements.** Because harness optimization is orthogonal to model training, you can compose it with any post-training gain from the underlying model. As models get stronger, the harness search space grows (more capabilities to invoke), and automated search becomes more valuable, not less.

## Gotchas & tricks

- **Validation-set overfitting.** A harness optimized against a fixed validation set will eventually overfit to it — especially if patch generation is high-throughput. Rotate held-out validation sets or use cross-validation on the failure batches.
- **Reward hacking of the metric.** If the target metric measures completion but not quality, patches can push toward superficial completion. Same lesson as reward-hacking in RLHF — audit patches whose gains look surprising.
- **Diagnosis quality is the bottleneck.** A misdiagnosed failure produces a patch that fixes the wrong thing. Consider ensembling multiple diagnosers or having the patch generator justify its target component.
- **Harness code representation matters.** Harnesses represented as loose text (a giant prompt with implicit sections) give patch generators a huge, ambiguous edit surface. Structured representations (tool-schema JSON, sectioned prompt templates, control-flow AST) narrow the edit surface and make patches meaningful.
- **Don't optimize on tasks you'll evaluate on.** GAIA2 patches that "work" often just memorize GAIA2 quirks. Reserve untouched benchmarks for out-of-distribution evaluation.

## Sources

- Paper: *AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces* — Park et al., 2026. [arXiv:2608.23041](https://arxiv.org/abs/2608.23041).
- Related: *SWE-Bench* — as a harness-sensitive benchmark widely used for evaluating agent scaffolding.

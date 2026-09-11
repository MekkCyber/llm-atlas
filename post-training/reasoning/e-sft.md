# E-SFT — Endpoint SFT for Reasoning

*Depth — train SFT only on the beginning and ending segments of a reasoning trace, dropping the redundant middle.*

**TL;DR:** Reasoning traces used in SFT are highly redundant. Attention analyses and token-removal studies show the *middle* of a long chain-of-thought contributes disproportionately little to learning. **E-SFT** — endpoint SFT — trains on only the *setup* and the *conclusion* of each trace, dropping the middle. It matches or beats standard full-trace SFT on reasoning benchmarks and improves downstream RL and distillation. Less data, less compute, shorter output distribution.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md)
**Related:** [long2short](long2short.md), [../rejection-sampling.md](../rejection-sampling.md), [../rlvr.md](../rlvr.md)

---

## What it is

Standard reasoning SFT trains on `(prompt, full CoT + answer)` pairs. Because state-of-the-art reasoners emit multi-thousand-token traces, this is expensive to store, expensive to train on, and encourages the model to generate similarly long outputs at inference.

E-SFT observes that not all trace tokens carry the same signal. By attention pattern analysis and by systematically removing token ranges before SFT and measuring downstream performance, the paper finds that a long middle section of most traces is *redundant* — removing it does not hurt (and sometimes helps) SFT quality.

## How it works

The recipe:

```
for (prompt, trace) in dataset:
    head   = trace[:H]      # first H tokens: problem setup, plan
    tail   = trace[-T:]     # last T tokens: final derivation + answer
    target = head + tail    # concatenate
    SFT(model, prompt -> target)
```

`H` and `T` are hyperparameters; the paper reports values that preserve or improve accuracy while cutting many thousands of tokens. The intuition: `head` fixes the *reasoning shape* (what the model is about to do); `tail` fixes the *derivation shape* + answer. The middle is a lot of expansion that the model can re-derive from those endpoints.

E-SFT can be applied to any trace source — teacher-generated CoT, expert derivations, or filtered rollouts from a stronger reasoning model.

## Why it matters

- **Trace-data cost.** A frontier reasoner emits 8–32k tokens per problem; dropping most of the middle cuts training data volume proportionally.
- **Shorter output distributions.** Models trained on truncated traces emit shorter traces at inference, addressing the same length-tax problem as [length-penalty](length-penalty.md) and [long2short](long2short.md) — but at the SFT stage rather than RL.
- **Improves downstream RL and distillation.** The paper shows E-SFT'd checkpoints are a better warm start for both RL post-training (RLVR) and distillation into smaller students.
- **Simple.** No new architecture, no new objective — just a training-data transform.

## Gotchas & tricks

- **`H` and `T` are dataset-dependent.** Traces on IMO-hard problems can't be trimmed as aggressively as GSM8K-style traces. Sweep them per dataset.
- **Coherence matters.** If the head sets up steps whose intermediate values only appear in the removed middle, the tail may not type-check. The paper uses simple boundary heuristics; more careful boundary selection could go further.
- **Doesn't replace filtering.** Endpoint-only training on a bad trace is still bad training. Filter (rejection sampling) *then* truncate.
- **Complementary to length-penalty RL.** Truncating traces at SFT and penalizing length at RL stack.

## Sources

- Paper: *Revisiting Complete Reasoning Traces for Post-Training* — Hwang, Yun, Heo, Han — NAVER AI Lab, 2026 — [arXiv:2609.07103](https://arxiv.org/abs/2609.07103).

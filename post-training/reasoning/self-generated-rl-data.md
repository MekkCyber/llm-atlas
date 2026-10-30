# Self-Generated RL Data (LongZero)
*Depth — have the model author its own verifiable training problems, then RL against them. Zero human labels.*

**TL;DR:** A long-context reasoning self-improvement recipe: the model emits a structured `(question, answer, evidence)` triple; a cheap transform **expands the evidence** by padding with distractor documents into a long haystack; the model is then trained with [GRPO](../grpo.md) to recover its own pre-committed answer from that long context. The verifier is exact-match against the self-generated answer. LongZero (Wan & Chen, Tsinghua, 2026) uses this to improve long-context reasoning **from zero external data**.

**Prereqs:** [grpo](../grpo.md), [rlvr](../rlvr.md), [long-cot-rl](long-cot-rl.md)
**Related:** [rl-prompt-curation](../rl-prompt-curation.md), [rejection-sampling](../rejection-sampling.md), [../../data/dolma.md](../../data/dolma.md)

---

## What it is

A training loop where the model plays three roles:

1. **Problem generator** — writes a QA pair plus the evidence that proves the answer.
2. **Difficulty amplifier** — a deterministic padding step buries the evidence inside a long distractor context.
3. **Solver** — attempts to recover the pre-committed answer from the long context, scored by exact match.

The RL verifier is `generated_answer == recovered_answer`, which is pure code. No human labels, no learned reward model.

---

## How it works

### Rollout

```
# Phase 1 — self-generated task
(q, a, evidence) ~ π_θ(gen_prompt)        # model writes a QA + evidence

# Phase 2 — long-context padding
distractors = sample_distractor_docs(K)   # from a cheap doc pool
context = pad(evidence, distractors, length=L)

# Phase 3 — solve
a_hat ~ π_θ(context + q)
r = 1 if a_hat == a else 0
```

### GRPO update

Standard GRPO over `G` rollouts per prompt:

$$
A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}
$$

broadcast to all tokens of the solver response, plus KL to a frozen reference. See [grpo.md](../grpo.md).

### Difficulty knob

`L` (padded context length) and `K` (distractor count) jointly set difficulty. A curriculum schedule scales `L` from short → long as the running success rate rises. This is a cheap, implicit [prompt curation](../rl-prompt-curation.md) over the self-generated pool: easy prompts get larger `L`, hard ones don't need it.

### Why the verifier is honest

The ground-truth answer was *authored by the same model* that solves the task — so the model cannot "cheat" by writing a vague question whose answer it cannot pin down later: the generator step itself has to commit to a specific `a` string before padding. The exact-match check is on `a`, not on an open-ended judgment.

---

## Why it matters

- **Pure self-play for long context.** Long-context reasoning data is expensive to label. This recipe produces unlimited verifiable long-context tasks for the compute cost of generation.
- **Difficulty follows the model.** As the solver improves, the padding length can grow, keeping success rate in the GRPO-productive band (not too easy, not too hard).
- **Combines long-CoT RL with synthetic curriculum.** Builds on [long-cot-rl](long-cot-rl.md) by removing the dependency on a labelled task source like AIME / MATH-500. The emergent-reasoning finding carries over; the data assumption drops out.
- **Transfers to downstream long-context eval.** The paper reports lift on standard long-context reasoning benchmarks, from a base that never saw external long-context supervision.

---

## Gotchas & tricks

- **Generator-solver collapse.** If the generator and solver share weights, the model can collude: generate questions whose answers are trivially recoverable (keyword-match). The paper breaks this by forcing a structured template with disjoint `question` and `answer` fields and by requiring the answer to be short (so verifier is strict).
- **Distractor quality matters.** Distractors sampled from random web text are too easy to ignore — the model learns to attend to the one coherent block. Mix domain-matched distractors to prevent trivial attention cues.
- **Evidence placement bias.** If evidence is always at position 0 of the padded context, the model learns a positional shortcut. Randomize placement.
- **No format reward is needed if the generator schema is enforced.** The template itself provides format discipline; adding a format reward on the solver side only risks collapse to short outputs.
- **Zero-data is a slogan, not literal.** The base model's pretraining supplies the latent reasoning capability; LongZero only avoids *post-training* data. On a weak base, no amount of self-play fixes a missing capability — see the R1-Zero vs R1-Zero-Qwen-32B contrast in [long-cot-rl](long-cot-rl.md).

---

## Sources

- Paper: *LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data* — Wan & Chen, Tsinghua, 2026, [arXiv 2610.30132](https://arxiv.org/abs/2610.30132).
- Related: [grpo.md](../grpo.md) — the policy-optimization algorithm used.
- Related: [long-cot-rl.md](long-cot-rl.md) — the base recipe this extends from.
- Related: [rl-prompt-curation.md](../rl-prompt-curation.md) — related difficulty-labelling ideas on the prompt-pool side.

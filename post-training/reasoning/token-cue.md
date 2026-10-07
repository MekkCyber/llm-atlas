# Token Cues — Reasoning From Training-Data Associations
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A single fixed starting-token cue (e.g. `.\n\n Okay`) raises a base model's math/code accuracy close to its RL-trained counterpart. The 2026 paper causally traces this to **training-data associations**: cues trigger latent reasoning modes already present in the base, and RL mainly *makes those cues more likely*. Any arbitrary word can be made into an effective cue by targeted data interventions, and existing cues can be removed the same way.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../rlvr.md](../rlvr.md)
**Related:** [../_post-training.md](../_post-training.md) · [../grpo.md](../grpo.md) · [../../data/_data-curation.md](../../data/_data-curation.md)

---

## What it is

A reframing of what RL post-training does for reasoning. Rather than *teaching* the base model new reasoning behavior, RL preferentially raises probability on **tokens that gate** already-learned reasoning modes. The gate is literal: a few opening tokens of the response determine which behavioral mode the rest of the generation follows.

## How it works

- **The empirical setup.** Prepend a short, fixed cue to the model's response and measure pass@1 on math and code benchmarks.
  - Cue `.\n\n Okay`: **Olmo-3-7B MATH-500 42% → 78%.**
  - Cue ` Alright,`: **Qwen3-14B MATH-500 72% → 87%.**
  Fixing the cue recovers a large fraction of the gain between the base and its RL-trained version.
- **Causal data interventions.** Edit the pretraining data to associate an arbitrary word (e.g., "chicken") with reasoning-mode documents, and that word becomes an effective cue. The reverse edit neutralizes an existing cue. A targeted edit turns "Think duck duck goose" into a replacement for "Think step by step."
- **Hidden-state correspondence.** The hidden-state representations induced by different cues correlate with distinct document types in the training set, consistent with the mechanism being "cue → latent-mode gate."
- **Safety case study.** Different cues elicit distinct refusal/compliance behaviors, each corresponding to a different training-document type.

## Why it matters

- Reframes RL for reasoning: a portion of what looks like "RL teaches reasoning" is better described as "RL discovers and reinforces the data-conditioned tokens that already route to good modes." (The remainder — length growth, reflection — is still genuinely emergent; see [long-cot-rl.md](long-cot-rl.md).)
- Gives practitioners a **cheap, non-training knob**: starting-token prompting with carefully chosen cues can recover much of an RL run's benefit, which matters for budget-constrained adaptation.
- Suggests a new axis for **data design and safety**: curating which tokens-document associations the base learns is a lever on downstream behavior, both capability and refusal.

## Gotchas & tricks

- **Cues are model-specific.** `.\n\n Okay` works on Olmo-3; ` Alright,` works on Qwen3. The association is learned from each base's training mix.
- **Not a free lunch past the frontier.** Cues cap at the latent capability already in the base. For problems the base can't solve under any mode, cueing doesn't help.
- **Combines with RL.** Cueing complements, rather than replaces, RL — adding an RL phase on top of a good cue still improves.
- **Watch for safety side-channel.** The same mechanism lets attackers choose cues that unlock non-default (less-safe) behavioral modes; audit refusal under common cues, not only the default system prompt.

## Sources

- Paper: *Base Models Can Reason By Taking a Cue From Training Data* — 2026 — [arXiv:2610.06851](https://arxiv.org/abs/2610.06851).

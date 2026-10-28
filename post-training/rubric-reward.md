# Rubric-based rewards (with precision objective)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Rubric-based rewards score a response by checking it against a per-prompt **checklist** of required or forbidden properties. They are the main lever for RL on non-verifiable tasks (writing, planning explanations, soft reasoning) because they avoid training a reward model while still giving a graded signal. Pure rubric rewards score **recall over the checklist**, which rewards "cover everything plus noise." Fork (2026) adds a **precision objective** — each unit of the response must be both necessary and factually correct — which fixes the length- and padding-hacking failure mode without losing coverage.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md)

---

## What it is

A rubric is a prompt-specific list of items the response is graded on:

```
[ "names at least two historical examples",
  "cites a date for each example",
  "does not fabricate dates",
  "concludes with a one-sentence takeaway" ]
```

The reward is a function of how many rubric items the response satisfies. In the simplest form:

$$
r_{\text{coverage}}(x,\,y) = \frac{\#\{\text{items in rubric}(x)\;\text{satisfied by}\;y\}}{\#\text{rubric}(x)}
$$

This is **recall only**. Fork complements it with a precision term:

$$
r_{\text{precision}}(x,\,y) = \frac{\#\{\text{units in}\;y\;\text{that are necessary and correct}\}}{\#\text{units in}\;y}
$$

The composite reward is a weighted combination of recall and precision, optimized with any RL algorithm — the paper uses GRPO-style updates.

## How it works

1. **Rubric generation.** For each prompt, a stronger model (or a human) writes a rubric with 3–10 items. The rubric is stored once, reused across training.
2. **Scoring at rollout time.** A grader LLM checks each rollout against each rubric item (coverage) and segments the response into units (sentences or claims) to check each for necessity and correctness (precision).
3. **Composite reward.** Fork combines coverage and precision in a single scalar reward; the paper reports that the exact weighting is less important than *including* precision at all.
4. **RL update.** Standard GRPO or PPO on this reward.

## Why it matters

Rubric RL is widely used for non-verifiable tasks where RLVR doesn't apply but learned reward models are brittle. Coverage-only rubric rewards consistently produce longer and noisier outputs over training — the classic "cover everything plus padding" failure mode. Adding an explicit precision term is a one-line fix: it penalizes irrelevant units directly, so hacking the reward by stuffing extras no longer pays.

## Gotchas & tricks

- **The grader is the ML.** Rubric scoring is only as good as the grader LLM. Prompt the grader to score items strictly; audit on a held-out set.
- **Unit segmentation is subtle.** If "units" are too fine (per clause), honest answers get penalized; too coarse (per paragraph), hacking slips through. The paper uses sentence-level units.
- **Compose with length penalties, don't fight them.** Precision reward already discourages long padding; a separate length penalty can over-correct and clip genuinely longer, correct answers.
- **Fork reduces length *and* improves win rate** against the base model in the paper — the length drop isn't a loss of substance.

## Sources

- Paper: *Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-based RL* — anonymous (affiliations not disclosed), 2026 — [arXiv:2610.33512](https://arxiv.org/abs/2610.33512) — introduces the Fork reward with the precision term.
- Prior art for rubric-style RL: *Constitutional AI* (Anthropic), *RLAIF* literature.

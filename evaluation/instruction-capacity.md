# Instruction Capacity
*Depth — a statistical probe for how many simple, independent instructions a language model can hold concurrently.*

**TL;DR:** **Instruction Capacity** (Kutomanov, 2026) measures instruction-following not as per-instruction correctness but as a **load curve**: accuracy as a function of the *number of concurrent independent rules*. The task is controlled word-replacement — the system prompt contains N "replace word X with word Y" rules, independent by construction, and the model is scored on replacement correctness over hundreds of thousands of trials. The headline: GPT-4o mini holds **~37–38 rules at 95% correctness** and **~18–19 rules at full-density** replacement, with smooth (not cliff-shaped) degradation. Complementary to IFEval — IFEval measures "did one instruction pass"; Instruction Capacity measures "how many can pass at once."

**Prereqs:** *(none)*
**Related:** [ifeval.md](ifeval.md), [mmlu.md](mmlu.md)

---

## What it is

A microbenchmark of concurrent-instruction load:

- **Rule form**: "replace word X with word Y" (`X ↦ Y`).
- **Rules are deliberately independent**: no two rules share source or target words. Interference is pure capacity, not conflict.
- **Density**: fraction of occurrences of X in the input that must be replaced.
- **Metric**: fraction of X occurrences that are correctly replaced according to the rules.

Reporting is a **load curve**: accuracy vs number of concurrent rules, at a fixed density. The characteristic numbers are *degradation thresholds* — the N where correctness crosses a target rate.

---

## How it works

### Task construction

```
rules = [(X_i, Y_i)] for i in 1..N        # independent source/target words
density ∈ (0, 1]                           # fraction of X_i occurrences to replace
text = sampled text containing many (X_i)
prompt = system(rules) + user(text)
```

The input text is engineered to contain each X_i multiple times, so each rule gets multiple test instances per trial.

### Scoring

```
for each X_i occurrence in the input:
    expected = Y_i if the per-density rule says replace else X_i
    observed = extract from model output at that position
    correct += (expected == observed)
```

Reported per-rule-count correctness averaged over many trials. Variance is reduced by the large number of trials (642K+ for the GPT-4o mini headline numbers).

### Degradation characterization

The paper reports two canonical operating points:

| Density | Correct-replacement share | GPT-4o mini N |
| --- | --- | --- |
| Partial (0.95 share target) | 0.95 | **37–38** rules |
| Full density (1.0) | 1.0 | **18–19** rules |

The degradation is **smooth**, not cliff-shaped — accuracy drops gradually as N rises, which rules out a "working memory" cliff model.

---

## Why it matters

- **Isolates concurrent-instruction load from per-instruction difficulty.** IFEval mixes both; Instruction Capacity holds difficulty constant and varies only count.
- **Matches a real-world failure mode.** Complex system prompts (agent setups, tool schemas, role prompts) often stack many instructions. "The model forgot the third one" is a common complaint; this benchmark quantifies it.
- **Smooth degradation suggests a resource-pool model.** Not a cliff — the model has a soft capacity whose exceedance degrades all rules proportionally. Useful mechanistic prior.
- **Cheap and repeatable.** 2,958 requests / 642K trials; no human grading. A full capacity measurement runs in minutes.

---

## Gotchas & tricks

- **Independence must be enforced in the generation.** If rules share source words ("replace cat with dog" + "replace cat with bird"), the test measures conflict resolution, not capacity. The benchmark's definition requires independence.
- **Vocabulary difficulty is a confound.** Rare source words are harder; frequent source words are easier. The paper controls this by sampling from a common vocabulary.
- **Model-specific scaling.** GPT-4o mini's thresholds don't transfer to Claude or Gemini. Run the probe on your own target model.
- **Density matters.** Full-density (replace every occurrence) is a much harder regime than partial (replace a sampled subset). Report both or state which.
- **Doesn't measure composite-instruction load.** Instructions in production are rarely independent — many have conditional structure ("if X then Y"). Capacity for such instructions is likely lower than this benchmark's upper bound.
- **Not a replacement for IFEval.** Pair with IFEval for a two-axis picture: one measures *that* instructions pass, the other measures *how many*.

---

## Sources

- Paper: *Instruction Capacity: Behavior and Degradation of Language Models under Controlled Load* — Kutomanov, independent, 2026, [arXiv 2610.36095](https://arxiv.org/abs/2610.36095).
- Related: [ifeval.md](ifeval.md) — the per-instruction instruction-following complement.

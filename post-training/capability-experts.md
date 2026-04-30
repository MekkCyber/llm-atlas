# Capability Experts

*Depth — branch pretraining partway, continue on a domain-heavy mix, then use the expert to generate SFT data for the flagship.*

**TL;DR:** Pretraining produces a generalist base. To generate **high-quality domain-specific SFT data** (code, math, multilingual) for the flagship instruct model, **branch the pretraining run and continue on a domain-heavy mix** to create a "capability expert." Use the expert to answer prompts in that domain; the flagship SFT is trained on the expert's best generations (via rejection sampling). Introduced at scale in Llama 3 (Sec. 4.3.1 "Code Expert", 4.3.2 "Multilingual Expert"). A specific instance of teacher-student distillation where the teacher is a specialist fork of the same pretraining run.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md), [reward-modeling](reward-modeling.md)
**Related:** [mid-training](../pre-training/mid-training.md) · [llama-3 case study](../case-studies/llama-3.md)

---

## What it is

The challenge: SFT data for specialized capabilities (code, math, rare languages) is hard to collect at scale. Human annotators for code are expensive; for math are scarce; for Thai are both.

Llama 3's solution: **grow your own annotator** by branching a partially-trained pretraining run and continuing on a domain-heavy mix. The resulting "expert" model is far stronger in the target domain than the flagship-pretrained model. Use the expert to generate SFT data for the flagship.

Two canonical cases in Llama 3:

- **Code Expert (Sec. 4.3.1)**: branch Llama 3 pretraining, continue on **1T tokens at >85% code**, then LCFT to 16K. Use for code SFT data generation.
- **Multilingual Expert (Sec. 4.3.2)**: branch, continue on a **90% multilingual** mix. Use to annotate non-English data and for RS on multilingual SFT prompts.

---

## How it works

### The pipeline

```
Main pretraining run (trillions of tokens, general mix)
    ↓
    ├── Continue fully → flagship base model (generalist)
    └── Branch at some point, continue on domain-heavy mix → Expert
                                                                ↓
Expert rejection-samples K completions per domain prompt
    ↓
RM or rule verifier picks top-1
    ↓
Top-1s → added to flagship's SFT dataset (alongside general-mix SFT data)
```

### When to branch

Branch point depends on:
- **Shared foundation**: The expert should share enough pretraining with the flagship that its style matches; branch too late → the expert and flagship diverge too much.
- **Specialization budget**: The expert needs enough domain training to be measurably stronger than the flagship. Llama 3 Code Expert continues on 1T tokens of code-heavy mix — a substantial fraction of a full pretraining.

Llama 3 doesn't disclose the exact branch point; the paper says "we branch the main pre-training run."

### What changes in the branch

- **Data mix**: shift heavily toward the target domain (>85% code, 90% multilingual).
- **Maybe context length**: Llama 3 Code Expert gets LCFT (long-context fine-tune) to 16K at the end.
- **Maybe LR**: continue with a lower LR or a cosine decay-to-zero.

### Using the expert

1. **Collect prompts** in the target domain (code problems, multilingual queries).
2. **Sample K rollouts per prompt** from the expert.
3. **Filter**:
   - For code: rule-based execution (unit tests) + LLM-as-judge for style. Keep only perfect-2 samples. See [rejection-sampling](rejection-sampling.md).
   - For multilingual: language-match check + RM top-1.
   - Responses-revise-rather-than-drop: if strict filtering over-drops hard prompts, have a model revise existing responses instead of discarding them.
4. **Add top-1s** to the flagship's SFT mix.

### Llama 3 Code Expert specifics (Sec. 4.3.1)

Three generation pipelines the Code Expert feeds:

1. **Execution feedback pipeline (~1M dialogs)**:
   - Generate problem → solve → static analysis (parser/linter) → unit-test generation → containerized execution → iterative self-correction if any check fails.
   - ~20% of initial solutions wrong but self-correct via the feedback loop.
   - Pure execution signal — no learned-RM needed.

2. **Programming-language translation (~? dialogs)**:
   - Translate common-language (Python/C++) code to rarer (TypeScript/PHP/Swift/Rust).
   - Validate via parse/compile/execute.
   - Improves MultiPL-E / HumanEval-Mul scores.

3. **Backtranslation (~1.2M dialogs)**:
   - For documentation/explanation where execution isn't a signal: generate docs from code → backtranslate docs back to code → use the expert as self-verification judge → keep top-scoring samples.

Total: **>2.7M synthetic code SFT examples**.

### Llama 3 Multilingual Expert specifics (Sec. 4.3.2)

- 90% multilingual mix during branch.
- Used to produce rejection-sampled responses for 7 target languages (German, French, Italian, Portuguese, Hindi, Spanish, Thai).
- RS temperature: 0.2–1.0 early, fixed 0.6 in final round.
- Language-match check before RM selection (prevent script mismatches).
- Multilingual SFT mix: 2.4% human, 44.2% other-NLP-task data, 18.8% RS, 34.6% translated reasoning.

---

## Why it matters

- **Scales domain-specific data without humans.** Code and multilingual data generation would require unfeasible human labeling otherwise. Expert generation is cheap (one forward pass per rollout).
- **Better than the flagship at the target domain.** The expert is purpose-trained for one thing; within that domain, its outputs are higher-quality than the flagship's would be. Using its outputs as SFT lifts the flagship.
- **Cleanly separates concerns.** The flagship trains on general data + expert-sourced domain data. The expert trains on domain-heavy. Each is optimized for its job.
- **Generalizes to any specialized capability.** If you need math, branch a math expert. If you need low-resource language X, branch a multilingual expert weighted toward X. Reusable pattern.

---

## Gotchas & tricks

- **Expert != flagship.** The expert is not deployed; it's only used to generate SFT data. Its domain-heavy training may damage general capability, which is fine because the flagship doesn't inherit those damages.
- **Branch point matters.** Too early: the expert isn't strong enough. Too late: it's essentially the flagship. Llama 3 doesn't disclose; common practice is to branch after ~80% of pretraining.
- **Domain data quality matters.** Expert's ceiling is the quality of its domain-heavy mix. Curate carefully; contamination with benchmark test sets is especially bad here.
- **Style drift.** Expert outputs can have a different style from the flagship's generalist output (too formal for code, too terse for dialog). Mix expert-generated data with general SFT data at typed rates to control this.
- **Per-domain filter quality.** Code has execution verification (strong). Math has rule verifiers (medium). Multilingual has language-match checks (weak) + RM (hackable). Filter strength determines how much RS data you can trust.
- **Tool use doesn't use experts for RS.** Llama 3 notes (Sec. 4.3.5) that tool-use SFT doesn't benefit from RS — so no tool-use expert. Domain-dependent.
- **Revising-over-dropping for hard prompts.** If strict filters over-drop challenging prompts, have an expert model **revise** an existing response instead of discarding. Keeps prompt coverage.
- **Don't use experts for DPO.** DPO needs preferences matched to the *current policy's distribution*. Expert-generated responses don't match the flagship's distribution, so using them for DPO would be off-policy.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 4.3.1 (Code Expert), 4.3.2 (Multilingual Expert).
- Paper: *Code Llama* — Rozière et al., 2023, arXiv 2308.12950 — a specialist derived from Llama; one-expert precursor.
- Related: [rejection-sampling](rejection-sampling.md) for the data-filtering step.
- Related: [reward-modeling](reward-modeling.md) for the ranking signal.

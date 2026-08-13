# Decoding-Level Taboo
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A zero-prompt diagnostic stress test for LLM robustness that intervenes **directly in logit space at decoding time**: at each word boundary, mask the model's top-probability candidate token and force it to pick something else. The model must then *circumlocute* — say the same thing with different words. Measures how much of a model's benchmark performance depends on walking the narrow, highly optimized nominal generation corridor. Off-path robustness improves with **parameter scale** and **post-training instruction alignment**. Introduced in *Decoding-Level Taboo* (Kamijo et al., 2026).

**Prereqs:** *(none)*
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Standard LLM evaluation measures **nominal-path performance** — the score you get when the model is free to sample its highest-probability continuation. In real deployment, this path is rarely available:

- **System prompts** and role instructions perturb the distribution.
- **Safety guardrails** filter or rewrite tokens.
- **Structural constraints** (JSON mode, grammars, tool schemas) force particular tokens at particular positions.

Every one of these pushes the model off its optimized generation corridor. Nominal-path evals cannot see the gap; adversarial-prompt evals mostly measure prompt sensitivity, not decoding robustness.

Decoding-Level Taboo isolates the decoding-robustness dimension by intervening at exactly the layer where the perturbation happens in production: **the logits**.

## How it works

1. Generate the response token by token.
2. At each **word boundary** (whitespace or subword boundary heuristic), inspect the logit distribution.
3. **Mask the primary candidate token** — set the top-probability token to $-\infty$ before sampling.
4. Sample the next token from the remaining distribution.
5. Repeat until the response is complete.
6. Grade the resulting response using the same rubric as the un-tabooed benchmark.

The model is forced to *circumlocute*: it has to produce the same content using different words, because its favorite word is systematically taken away.

**Zero prompt engineering.** Everything happens at the logit layer. Any model with logit access can be tested; no adversarial prompts to write or maintain.

## Why it matters

- **Portable, reproducible robustness eval.** No prompt library, no adversary model — just a logit hook.
- **Correlates the right way.** Off-path robustness improves with **parameter scale** and with **post-training instruction alignment**. Bigger and better-aligned models circumlocute more successfully; brittle models produce nonsense once their favorite token is gone.
- **Diagnostic + audit tool.** Beyond a benchmark number, Taboo doubles as (a) a generator of **diverse synthetic datasets** by producing many different phrasings of the same answer, and (b) an **audit** of runtime safety guardrails — do they still work when the model is decoding off-path?
- **Fills a niche between nominal evals and adversarial prompts.** Complementary to both: nominal evals measure the ceiling, adversarial evals measure prompt-attack robustness, Taboo measures decoding-perturbation robustness.

## Gotchas & tricks

- **Word-boundary heuristic matters.** Masking every token (not just at word boundaries) hurts too aggressively — the model can't even complete a multi-token word. Word-boundary masking is the setting the paper defaults to.
- **Grade with the original rubric, not a robustness-tuned rubric.** The gap between nominal and taboo scores is the signal; changing the rubric confounds it.
- **Report both nominal and taboo scores.** A model that scores 80% nominal → 75% taboo is different from one that scores 65% → 60%, even though both drop 5 points.
- **Small models can degrade catastrophically.** Below some scale, taboo scores collapse. Report the scale-vs-robustness curve, not a single number.
- **Logit access required.** Closed models without logit APIs can't be tested this way; use it on open-weight models or on providers exposing logit_bias.

## Sources

- Paper: *Decoding-Level Taboo: A Diagnostic Stress Test for LLM Robustness* — Kamijo, Rottenstreich, Conde, Martínez, Reviriego, arXiv 2608.09900, 2026.

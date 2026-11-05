# Model-Aware Adversarial Problem Synthesis
*Depth — Stage 1 data construction for Axion 1.0: generate reasoning problems from the base model's own successful solution strategies, then adversarially strengthen them.*

**TL;DR:** A self-generating reasoning-SFT dataset pipeline that pins problem difficulty to the **frontier of what the base model can almost do**. The base model solves a seed problem; the pipeline extracts the solution's key strategy; a generator produces variants that *require that strategy* but are harder than the seed; a filter keeps only variants the base model fails on with meaningful probability. Trained as the Stage 1 data for [Axion 1.0](../case-studies/axion-1-0.md), it supplies the single-stage SFT that lifts base → +11 points on HLE-130 (122B).

**Prereqs:** [_post-training.md](_post-training.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md) · [../case-studies/axion-1-0.md](../case-studies/axion-1-0.md)

---

## What it is

Reasoning SFT data construction usually either (a) scrapes expert-solved problems from the internet (noisy, finite), (b) prompts a stronger teacher to generate problems (costly, teacher-bound), or (c) hand-authors problems (unscalable). Each of these decouples problem difficulty from what the student *actually needs* to grow.

Model-aware adversarial problem synthesis closes that loop. Problems are generated *conditional on* the base model's own success trajectories — so they inherit the strategies the model already knows — and then adversarially strengthened until they sit right at the threshold where the model fails. The resulting SFT signal is dense exactly where the model has headroom.

---

## How it works

Four steps per seed:

1. **Seed solve.** Give the base model $M$ a seed problem $p_0$ from a pool (math, science, coding). Keep only seeds where $M$ succeeds with high probability (correctness verified by a rule-based checker).

2. **Strategy extraction.** From $M$'s successful solution $s_0$, extract the **key strategy** — the pivotal reasoning move(s). The paper does this via a prompted extraction pass (prompt ask: *"what is the key technique used in this solution?"*).

3. **Variant generation.** Prompt a generator (same model or a different one) to produce new problems $p_1, \ldots, p_k$ such that:
   - Each $p_i$ requires the extracted strategy to solve correctly.
   - Each $p_i$ is harder than $p_0$ (longer, more adversarial numbers, additional constraints, or distractors).

4. **Adversarial filtering.** For each variant $p_i$:
   - Run $M$ multiple times; estimate success rate $\alpha_i$.
   - Keep $p_i$ only if $\alpha_i$ falls in a target band (e.g. $\alpha_i \in [0.1, 0.5]$) — hard enough that the model fails sometimes, not impossible.
   - Collect correct solutions (from multiple samples or from a rule-based verifier) as the SFT target.

Discard seeds whose variants all end up outside the target band — either they're too easy (strategy generalizes trivially) or too hard (strategy doesn't transfer).

### Why this works

- **Strategy-conditioned variants** inherit the model's working knowledge, so the SFT target isn't "learn something new" but "sharpen something you already can almost do."
- **Adversarial filtering** targets the frontier — the specific failure modes of $M$ on problems requiring its own known strategies.
- **No stronger teacher required.** The generator can be $M$ itself; difficulty comes from adversarial numbers and constraints, not from stronger problem authorship.

---

## Why it matters

- **Eliminates the teacher bottleneck.** Most reasoning-SFT pipelines bottleneck on teacher-generated problems. Here the student is its own problem generator.
- **Dense signal at the frontier.** SFT loss on correct solutions of near-threshold problems is far more informative than loss on easy or impossible ones.
- **Implicit preference signal.** Variants that $M$ succeeds on give positive examples; those it fails on indicate where the signal is sharpest. Axion's paper argues this replaces a DPO phase in the broader recipe.
- **Composable with existing SFT stacks.** The output is $(prompt, response)$ pairs — ingest unchanged.

---

## Gotchas & tricks

- **Strategy extraction is noisy.** Prompted extraction often returns paraphrases of the full solution rather than the pivotal move. Filter extracted strategies for length and for presence of named techniques.
- **The target difficulty band matters.** Too narrow → few variants kept, pipeline stalls. Too wide → mix of easy and impossible, signal diluted. Band $[0.1, 0.5]$ is the paper's default; adjust to the base model's ability.
- **Verifier quality dominates.** Rule-based correctness checks work for math and code; general reasoning needs a stronger verifier (judge LLM, test cases). A weak verifier mislabels variants and poisons the SFT target.
- **Diversity over seed pools.** Running the pipeline on one seed pool (e.g. AMC problems only) specializes the model to that pool. Mix seed sources to avoid topical collapse.
- **Variant generator creativity vs fidelity.** Too creative and variants don't actually require the extracted strategy (the model solves them with an easier shortcut). Prompt the generator to *preserve* the strategy explicitly.
- **Not a pretraining substitute.** The pipeline shines on reasoning-shaped tasks where "strategy" is well-defined. For open-ended generation (chat, creative writing), the frame breaks down.

---

## Sources

- Paper: *Axion 1.0: Open Agent Foundation Models, Simplified* — 2026, arXiv 2611.03233 — primary source. Stage 1 data construction, §3.
- Related: [rejection-sampling](rejection-sampling.md) — adversarial filtering is a difficulty-targeted rejection-sampling step.
- Related: [../case-studies/axion-1-0.md](../case-studies/axion-1-0.md) — full training recipe this feeds into.

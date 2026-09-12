# Multilingual Data Mixing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning language models default back to English mid-CoT even when prompted in low-resource languages. Data-mixture design — not per-language reasoning supervision — is the lever that fixes this. "Building Multilingual Bridges" trains Tiny Aya L2-Thinker at 3.35B parameters and hits **>93% in-language reasoning across 60 languages** by picking the right multilingual pretraining/SFT mixture, *without* CoT data in the target languages. Reasoning transfers across typologically distant languages when the mixture is right.

**Prereqs:** [_data-curation](_data-curation.md), [dolma](dolma.md)
**Related:** [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

The default recipe for multilingual reasoning is "collect CoT data in every target language" — expensive, and impossible for languages without native mathematical or scientific writers online. Multilingual data mixing reframes: keep CoT data English-heavy, but design the *pretraining* mixture (and, in a lesser role, the SFT mixture) so the *behavior* transfers cross-lingually. The claim: reasoning is language-agnostic behavior; language grounding is a mixing problem.

## How it works

Two levers:

1. **Pretraining language mix.** Balance the language distribution of the pretraining corpus so that lower-resource languages are up-sampled beyond their web frequency (per-language temperature scaling). Under-sampling leaves the language brittle to fine-tuning drift; over-sampling wastes capacity on rare tokens.
2. **SFT / thinking-mix.** During post-training, mix a small share of *in-language non-reasoning* data with the English CoT data. The non-reasoning target-language data anchors the output distribution to the target language; the English CoT data teaches the reasoning pattern. Neither alone is enough.

The paper's recipe (paraphrasing):

- Pretraining language weights follow $p_l \propto n_l^\tau$ with $\tau \in (0, 1)$ to up-sample low-resource languages.
- SFT: majority English-CoT + a curated slice of target-language instruction data (no CoT), stratified across the 60 target languages.
- Result: at inference, the model *reasons* in the prompt's language even though it never saw a CoT trace in that language during training.

## Why it matters

- **CoT scaling without per-language supervision.** The community has been stuck at ~a dozen languages with reasoning traces; this route generalizes to 60+.
- **Language transfer is a mixture property.** A cleaner way to think about what makes multilingual reasoning work — data engineers can *design* it rather than *collect* it.
- **Small-model unlock.** 3.35B model reaches >93% in-language reasoning; this pattern doesn't require frontier scale.

## Gotchas & tricks

- **Temperature $\tau$ tuning.** Too low ($\tau = 1$) = English dominates and low-resource languages stay brittle. Too high ($\tau = 0$) = uniform distribution wastes capacity. Empirically $\tau \in [0.3, 0.5]$.
- **Language-family clumping.** Transfer within a family (Slavic → Slavic) beats transfer across families (Slavic → Bantu). Curate mixture per-family, not per-language, when you have limited data.
- **Instruction-tuning bleed.** SFT can *undo* the pretraining mixture if it's too English-heavy; keep a per-language quota in the SFT set.
- **Evaluation trap.** In-language *response* rate is easy to hit ("say the answer in Swahili"); in-language *reasoning* rate (does the CoT itself stay in Swahili?) is the real metric.
- **Reasoning-in-English drift.** Even with the right mixture, RL post-training (long-CoT-RL) can pull the model back toward English CoT; keep the target-language SFT anchor throughout post-training.

## Sources

- Paper: *Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning* — Mofakhami, Sahu, Salamanca, D'souza, Berard, Euyang, Fadaee, Kreutzer (Cohere / Mila), 2026 — [arXiv:2609.10445](https://arxiv.org/abs/2609.10445)
- Related: Aya model family — the multilingual lineage this extends.
- Related: earlier work on cross-lingual instruction tuning (mT0, BactrianX) — precursors that stopped short of reasoning transfer.

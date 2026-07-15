# Theory-Grounded Moral Reasoning (MET / MET-D)
*Depth — a two-step moral-reasoning scaffold that first selects theory-based moral "grounds," then reasons over them in the user's native language; self-distills into weights without external supervision.*

**TL;DR:** For culturally- and linguistically-diverse moral decision-making, standard English-centric CoT scaffolds miss cultural specificity. **MET** first has the model select from expert-curated moral **grounds** drawn from psychology and philosophy (situation- and culture-specific), then reasons over those grounds in the user's native language. **MET-D** internalizes MET into weights via **self-distillation** — no external teacher model, no human labels. Averages +4 macro-F1 over base models across three model families; boosts native-language reasoning by +62.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [low-resource-language-jailbreak.md](low-resource-language-jailbreak.md), [../post-training/dpo.md](../post-training/dpo.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Three failure modes in current LLM moral-reasoning stacks:

1. **Benchmarks translate.** Multilingual moral-decision benchmarks are built by machine-translating English items — losing culture-specific norms.
2. **Scaffolds are English-centric.** Inference-time CoT prompts are static, English-centric, and detached from actual moral theory.
3. **Training needs strong teachers.** SFT recipes for moral decisions typically rely on a stronger model or human annotators — expensive, and biased toward the annotator's culture.

MET addresses all three with (a) a benchmark contribution and (b) a training-time recipe:

- **MCLASH** — a multilingual moral-decision benchmark built with cultural adaptation, not translation.
- **MET** (inference-time) — two-step prompting: **select grounds → reason over them in native language**.
- **MET-D** (training-time) — self-distillation of MET's second step; the model produces its own targets from MET-scaffolded reasoning, then SFTs on them.

## How it works

### Step 1: Ground selection

A curated pool of moral **grounds** — expert-authored, theory-based reasoning primitives from moral psychology (care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, sanctity/degradation, liberty/oppression) and philosophy (deontological, consequentialist, virtue). Grounds are tagged by culture and situation type.

Given a moral scenario, the model **selects the situation- and culture-specific grounds** relevant to the case. This is a retrieval-like step over a small, curated set.

### Step 2: Native-language reasoning over grounds

Given the selected grounds, the model reasons over them **in the user's native language** to produce a decision. The scaffold is compact: grounds provide the moral content, native language provides cultural fit.

### MET-D: self-distillation

For each training scenario:

1. Run MET (steps 1–2) on the base model to produce a scaffolded response.
2. Extract the reasoning (step 2 output) as a training target.
3. SFT the model on `(scenario, target)` — the model learns to internalize the MET reasoning without the scaffold at inference time.

No external teacher, no human labels. The scaffold is the teacher; distillation is into the model itself.

## Why it matters

- **Cultural fit without larger models.** MET-D lifts small models (Qwen3-4B, Qwen3-8B, Gemma3-4B) on culturally-grounded morality tasks by internalizing the scaffold, without requiring a stronger teacher.
- **General pattern: distill your prompting into weights.** MET-D is one concrete instance of "scaffolded reasoning → self-distill → drop scaffold." Applies beyond morality — any task where a structured scaffold beats free-form CoT.
- **Native-language reasoning gains are massive.** +62 average points on native-language reasoning suggests base models default to English CoT even when prompted in-language; MET-D breaks this.
- **Theory-grounded > free-form CoT.** For value-loaded tasks where "reasoning steps" are contested, giving the model explicit theoretical primitives beats letting it invent moral content.

## Gotchas & tricks

- **Grounds pool quality bounds performance.** The scaffold is only as good as the expert-curated pool. Adapting to new domains (medical ethics, professional ethics) requires expert re-curation.
- **Culture tags are coarse.** "Malay culture" is a shorthand for a huge diverse population; treat MCLASH results as directional, not definitive.
- **Self-distillation can amplify biases.** If the base model has biased ground selection, MET-D bakes that bias in. Audit the SFT targets before training.
- **Not a safety fix on its own.** Better moral reasoning ≠ better refusal or better resistance to jailbreaks. Compose with standard safety training.
- **Distinct from constitutional AI.** CAI uses an LLM judge over principles; MET uses theory-grounded scaffolding as the reasoning substrate. Overlapping motivations, different mechanisms.

## Sources

- Paper: *MET: Theory-Grounded and Culture-Aware Multilingual Moral Reasoning* — Lee et al., 2026 — arXiv:2607.11736.
- Related benchmarks: MCLASH (introduced in this paper) and MMoralExceptQA (referenced for evaluation).
- Related methodology: self-distillation lineage from *Self-Instruct* (Wang et al., 2023) and rejection-sampling SFT ([rejection-sampling.md](../post-training/rejection-sampling.md)).

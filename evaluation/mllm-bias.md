# MLLM Bias Evaluation (Attribute-Isolated)

*Depth — measuring multimodal-LLM social bias by varying one visual attribute at a time on a fixed identity, isolating appearance effects from identity effects.*

**TL;DR:** Most MLLM bias benchmarks compare *different people* across demographic groups, which conflates "what cue is the model responding to" with "which group is the person in." **Attribute-isolated bias evaluation** holds identity fixed and generates ~50 single-attribute variations of each base face (age, body type, fashion style, etc.), then measures how each attribute alone shifts model judgments. Introduced as StylisticBias (Kolli et al., 2026): 500 base faces × ~50 variations × 6 MLLMs × 25 binary judgment scenarios. Result: **~15 attributes drive ~80% of social bias** — the bias surface is concentrated, not diffuse.

**Prereqs:** [_data-curation](../data/_data-curation.md)
**Related:** [../safety/_jailbreaks.md](../safety/_jailbreaks.md), [../safety/competing-objectives.md](../safety/competing-objectives.md)

---

## What it is

A controlled-perturbation evaluation methodology for MLLMs:

1. Generate a base set of photorealistic faces with diverse identities.
2. For each base face, generate ~50 *single-attribute variations* — same identity, one attribute changed at a time. Attributes span body type, age, fashion style, accessories, hairstyle, etc.
3. For each (image, judgment scenario) pair, query the MLLM and record the binary judgment.
4. Compute per-attribute *sensitivity*: how much does swapping attribute A (with everything else held constant) shift the judgment?

The methodology produces a **per-attribute, per-model fingerprint** of bias — not "this model is biased" but "this model is 4× more sensitive to body type than the next."

## How it works

Construction:

```
For each of 500 generated base faces (controlled for identity diversity):
    For each of ~50 attributes:
        Generate a variant with that attribute changed, identity preserved
        → ~25,000 images total

For each of 25 binary judgment scenarios (socioeconomic, style, trustworthiness, ...):
    For each of 6 MLLMs:
        Run the model on every image, record binary output
```

Sensitivity per attribute = average shift in judgment probability between base face and the attribute-varied face, aggregated over identities and scenarios.

## Why it matters

Three things this methodology unlocks that group-comparison benchmarks don't:

- **Causal isolation.** A measured shift is attributable to the attribute swap, not to confounded identity differences.
- **Attribute ranking.** The bias surface is concentrated — ~15 attributes (fashion style, body type, age) drive ~80% of variation. Means targeted interventions could move bias metrics more than broad-spectrum SFT.
- **Per-model fingerprinting.** Different MLLMs have different sensitivity profiles. This is useful for model selection in deployments where some attribute sensitivities matter more than others.

Sensitivity is strongest in judgments that are semantically aligned with appearance (socioeconomic, style-related) — which means the failure mode is structural rather than random noise.

## Gotchas & tricks

- **Identity preservation under attribute swap is generator-dependent.** If the image generator changes the face when adding a beard, you've conflated the attribute again. Audit the variant set.
- **Photorealism quality affects results.** Models can refuse or hedge on stylized images; the benchmark uses photorealistic base faces specifically to keep MLLM behavior comparable.
- **Binary judgment scenarios are coarse.** A pairwise-preference variant would be more informative but expensive to scale; the binary setup is the throughput compromise.
- **The "concentrated bias surface" finding is dataset-specific.** Other attribute sets (different cultures, different image styles) might surface different dominant attributes. Treat the result as a methodology demonstration first, a universal claim second.

## Sources

- Paper: *StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs* — Kolli, Cavelius, Nikeghbal, Dalal, Diesner, 2026 — https://arxiv.org/abs/2606.20527

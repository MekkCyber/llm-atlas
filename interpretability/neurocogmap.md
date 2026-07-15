# NeuroCogMap
*Depth — parcellate an LLM's internal features into cognitive-neuroscience-style functional regions, then link each parcel to interpretable functions and specific failure modes.*

**TL;DR:** Borrowing the "functional parcellation" methodology from human cognitive neuroscience, NeuroCogMap organizes an LLM's internal features into **functional parcels**, each linked to (a) an interpretable function, (b) a cognitive capability, and (c) a position in a cognitive hierarchy. Parcels are partly conserved across model families, predict output behavior, and map hallucination / bias / refusal-failure / sycophancy to distinct disruptions in specific systems.

**Prereqs:** [README.md](README.md)
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Two dominant units of "internal structure" in interp today: **circuits** (small, hand-traced subgraphs implementing one behavior) and **SAE features** (sparse over-complete decompositions of activations into human-readable directions). Both are useful but atomic — they zoom into single mechanisms.

NeuroCogMap sits one level up. It borrows **parcellation** — the neuroscience move of dividing cortex into functional regions each linked to a cognitive function — and applies it to LLM internals. The output is a **map of the model's cognitive organization**, not a list of individual features.

Concretely, each parcel is:

- A group of internal features that behave similarly across contexts,
- Labeled with an interpretable cognitive function (working memory, refusal control, retrieval, decision, etc.),
- Placed in a hierarchy that mirrors classical cognitive taxonomies.

## How it works

### Parcellation

Start from internal representations (residual-stream activations, MLP neuron activations, or SAE features — the paper's framing is method-agnostic at the input level). Cluster them into parcels using functional criteria: features that co-activate across tasks and whose ablation affects similar cognitive functions land in the same parcel.

### Function assignment

For each parcel, run a battery of probes measuring which cognitive function it participates in — attention control, semantic memory, source monitoring, refusal, etc. The paper uses standard cognitive-psychology task batteries repurposed for LLMs.

### Hierarchy

Parcels are ordered from perceptual / lexical (low) to decision / control (high). The hierarchy is inferred, not imposed — comes from how information flows between parcels during forward passes.

### Failure-mode mapping

Once parcels are labeled, characterize failure modes:

- **Hallucination** — disruption in retrieval + source-monitoring parcels.
- **Bias** — disruption in decision-control parcels influenced by lexical parcels.
- **Refusal failure** — disruption in specific control parcels (adjacent to but distinct from the safety-training refusal circuits).
- **Sycophancy** — disruption in belief-update parcels driven by user-context parcels.

Each disruption gives an **internal signature** — an activation pattern in the affected parcels — that enables detection and targeted intervention.

### Cross-model conservation

Parcels are partly conserved across model families: the same functional organization recurs when parcellation is run independently on different models. Suggests some functional structure is *task-shaped* rather than *training-run-shaped*.

### Bridge to biology

Parcel-level features **predict human cortical responses** during naturalistic language comprehension, with the strongest correspondence in higher-order association cortex. Not a claim of biological realism — but evidence that the parcellation captures something task-general enough to align with human data.

## Why it matters

- Provides a **system-level** unit of interpretability, complementing circuit- and feature-level work.
- Gives targeted-intervention handles for major failure modes (hallucination, bias, refusal failure, sycophancy) at the parcel granularity rather than requiring per-example patching.
- Cross-model conservation is a lead on the "universal features" question — some structure is shared, some is model-specific.
- Cognitive-hierarchy framing brings LLM interp closer to a shared vocabulary with cog-neuroscience, enabling cross-field methodology transfer.

## Gotchas & tricks

- **Parcels are not circuits.** A parcel is a *coarse* functional unit; the underlying mechanism may still be one or many circuits. Don't conflate levels.
- **Function labels are inferred from probes.** The label reflects the probe battery, not a ground-truth mechanism. A parcel called "working memory" is what the probes said, not what the model "is doing."
- **Cross-model conservation ≠ identity.** Parcels align in role, not in weights. Comparing parcels across models means comparing functional signatures, not architectures.
- **Correspondence with human cortex is prediction, not equivalence.** Better prediction of cortical BOLD signals is a *validity* argument, not a claim of biological realism.
- **Interventions are parcel-scale.** Steering a whole parcel is a blunter instrument than steering a single feature; expect side effects on adjacent functions.

## Sources

- Paper: *NeuroCogMap Reveals Cognitive Organization of Large Language Models* — Sun et al., 2026 — arXiv:2607.00397.
- Adjacent methodology: circuit analysis (Anthropic transformer-circuits thread), sparse autoencoders (Bricken et al. 2023, Templeton et al. 2024).
- Adjacent framing: mechanistic interpretability's "features / circuits / model" hierarchy — NeuroCogMap adds a *system* layer above features and circuits.

# Self-Patching (and the Knowing–Using Gap)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When you fine-tune an LLM on new facts, it *memorizes* them within a few steps but struggles to *use* them for downstream reasoning — a real accuracy and temporal gap the authors name the **Knowing–Using Gap**. Self-patching is a mechanistic-interpretability probe that identifies (layer, position) pairs where re-injecting an internal representation flips a failed generalization case into a success. Aggregating those pairs into a simple heuristic recovers **58–75% of the oracle-patch headroom**.

**Prereqs:** none (basic knowledge of transformer layers + activation patching).
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

Fine-tuning an LLM to inject new knowledge (facts, entities, relations) is a widespread practical need — RAG-adjacent enterprise adaptation, continual learning, knowledge editing. Empirically, models memorize training facts quickly (probe accuracy near 100% after a few epochs) but fail to *deploy* those facts in downstream reasoning tasks that require them (multi-hop QA, chained inference).

The paper formalizes this as the **Knowing–Using Gap**, with two measurable properties:

- An **accuracy gap**: probe / memorization accuracy is much higher than downstream-use accuracy on the same facts.
- A **temporal lag**: memorization saturates well before generalization begins to lift, if at all.

**Self-patching** is the probe that both diagnoses the gap and suggests a fix.

---

## How it works

### The knowledge-circuit misalignment hypothesis

The paper's hypothesis: memorized representations *exist* in the network — you can find them with linear probes — but they live in layers or positions that don't feed into the reasoning-heavy computation path. Fine-tuning writes the fact into the wrong "wire" of the circuit.

### The self-patching intervention

For a failed generalization case, iterate over (source layer $\ell_s$, target layer $\ell_t$, token position $p$) triples. At each candidate, copy the activation from the source location on a **memorization** prompt and paste it into the target location on the **generalization** prompt:

```
h_target[ell_t, p] = h_source[ell_s, p]
```

Re-run the forward pass on the generalization prompt with this patched activation. If the model now succeeds, that triple identifies a location where the memorized representation, when routed to the right place, is sufficient for generalization.

Aggregate over many failure cases: locations that consistently rescue generalization are the **routing points** the fine-tuning failed to write into.

### From probe to heuristic

The paper turns the oracle-patch experiment into a practical fix: use the aggregated routing points to select a small subset of activations to intervene on at inference time, with no ground-truth label needed. The resulting heuristic recovers **58–75%** of the oracle-headroom on cross-domain generalization failures — evidence that the diagnosis is causally correct, not just correlational.

---

## Why it matters

- **Mechanistic diagnosis of a widely-observed failure.** Knowledge-injection fine-tuning is known to underperform; self-patching provides evidence that the failure is *routing*, not capacity or data.
- **Actionable at both training and inference time.** The heuristic can be applied at inference (patch selected activations) or used to design new fine-tuning objectives that reward writing to the right layers.
- **A clean interp win with practical value.** Mech-interp papers often struggle to convert circuit-level findings into engineering wins. Self-patching's heuristic is a rare exception — a specific intervention with a measured effect on a benchmark.
- **Cross-domain robustness.** The finding replicates across domains, supporting the "structural routing failure" reading over a "training-data specific quirk" reading.

---

## Gotchas & tricks

- **Self-patching is expensive at oracle scale.** Sweeping over all (source, target, position) triples for many failure cases is quadratic in layer count. The paper caches per-layer activations to amortize the cost.
- **Heuristic accuracy < oracle accuracy.** 58–75% oracle-headroom recovery is a big chunk but not everything — the routing signal generalizes only imperfectly across queries.
- **Distinct from activation steering.** Activation steering adds a direction vector to a chosen layer; self-patching **copies** an actual activation from one location to another. The former imposes structure; the latter re-routes existing structure.
- **Requires the failure to be a routing problem in the first place.** If the model never memorized the fact at all (probe accuracy also low), self-patching won't help — it presupposes an internal representation to relocate.
- **Interacts with LoRA / PEFT.** With adapters at some layers only, the routing surface is constrained and the failure modes may look different. Full-rank fine-tuning was the paper's primary setting.

---

## Sources

- Paper: *Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning* — Dai, Rao, Wang, Wang, Liu, Xiong — HKUST(GZ) / HKUST — [arXiv:2607.08393](https://arxiv.org/abs/2607.08393).
- Code / data: [https://anonymous.4open.science/r/Mem2Gen-71FF](https://anonymous.4open.science/r/Mem2Gen-71FF).
- Lineage: activation patching / causal tracing (Meng et al., 2022, ROME) — the intervention primitive self-patching specializes.

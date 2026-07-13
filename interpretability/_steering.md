# Model Steering

*Taxonomy — inference-time interventions that control an LLM's behavior by editing its internal state, without fine-tuning weights.*

**TL;DR:** All steering techniques share a shape: find *where* a behavior lives inside the model (a direction, a neuron, a feature), then *edit that location* during the forward pass to amplify or suppress the behavior. Four families exist along two axes — **localized vs. distributed** and **hand-picked vs. learned**. **Activation steering** (directional injection) is the modern default because it handles distributed features and needs only a handful of contrastive examples; **SAE feature editing** is the more principled cousin when a trained sparse autoencoder is available.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [activation-steering](activation-steering.md)

---

## The problem

Once a model is deployed you often can't retrain it, but you may still need to modify its behavior — nudge it toward a dialect, suppress a class of refusals, amplify honesty, discourage a style. Prompting is the first tool; when it fails (jailbroken, overridden, drifting), you need something structural.

Steering assumes behaviors are represented internally in ways you can find and *edit* — as sparse neurons, as directions in activation space, as SAE features, or as targeted weight changes. The right lever depends on where the behavior lives and how much data you have.

## The shared pattern

Every steering method has three moves:

1. **Locate** — find the internal correlate of the target behavior. Neuron indices, an activation direction, a learned feature.
2. **Estimate strength** — how much to intervene. A scalar $\alpha$ for directional methods, a multiplier per neuron for ablation.
3. **Apply at inference** — inject into the forward pass at a chosen layer (or set of layers). No weight updates.

They differ on whether the target is a **single unit** (localized) or a **direction/subspace** (distributed), and on whether the target is **hand-picked** (feature you already understand) or **discovered** (via contrast, probing, or SAE training).

## Variants

| Technique | Locality | Discovery | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Neuron ablation / amplification (no depth file yet) | Localized | Discovered via probing or contrast | Only touches behaviors that concentrate on individual units | Behaviors known to be sparse (some dialect features, specific concepts) |
| [Activation steering](activation-steering.md) | Distributed (direction) | Discovered from contrastive activations | Steering strength $\alpha$ is fragile; can degrade fluency at large $\alpha$ | Most behaviors — the modern default |
| SAE feature editing (no depth file yet) | Distributed (learned features) | Requires a trained sparse autoencoder | Requires SAE per model + layer; features may not match target behavior | Interpretability-grounded control when an SAE exists |
| Weight editing / ROME / MEMIT (no depth file yet) | Localized (weight subset) | Hand-picked target fact | Modifies weights permanently; can produce unintended ripple | Injecting or removing specific factual associations |
| Inference-Time Intervention (ITI, no depth file yet) | Distributed | Learned attention-head direction | Specific to attention heads; strong on truthfulness | Truthfulness / honesty interventions |
| Representation engineering (no depth file yet) | Distributed (multi-direction) | Contrastive, top-down | Broader framework; overlaps activation steering in practice | Systematic multi-behavior control |

## How to choose

**Default: activation steering.** Cheap to extract (dozens of contrastive examples), works on distributed behaviors, no infrastructure beyond a forward-pass hook. Start here.

**Behavior is known to concentrate on a small neuron set** — pair activation steering with neuron amplification/suppression on the localized part. The 2026 Arabic-dialect paper shows this combination captures both localized and distributed halves of the behavior; neither alone was sufficient.

**You have an SAE for the target model and layer** — SAE feature editing gives semantically clean control ("amplify the 'French' feature") that raw directions can't match. Cost is the SAE training.

**You want to inject or remove a factual association** — reach for weight editing (ROME/MEMIT). Different tool than steering; different failure modes (fact ripple).

**Prompting works reliably** — don't reach for steering. Steering is what you use *when* prompts don't hold up under adversarial or distribution-shifted conditions.

## Adjacent but distinct

- **Fine-tuning** — updates weights on a dataset. Different economics: expensive, capable. Steering trades capability for zero training cost.
- **Prompting** — everyone's first tool. Steering starts where prompting stops.
- **CoT monitoring** ([../safety/cot-monitoring.md](./../safety/cot-monitoring.md)) — observing the reasoning trace, not editing internals. Complementary to steering.
- **Circuit-level interpretability** — finding *which subgraph* implements a behavior, then possibly ablating it. Localization plus intervention overlap heavily with the steering taxonomy but the emphasis is on understanding.

## Sources

- Paper: *Steering Language Models With Activation Engineering (CAA)* — Rimsky et al., 2023.
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023.
- Paper: *Inference-Time Intervention (ITI)* — Li et al., 2023.
- Paper: *Locating and Editing Factual Associations in GPT (ROME)* — Meng et al., 2022.
- Paper: *Sparse Feature Circuits* — Marks et al., 2024 — SAE-based steering / circuits.
- Paper: *Can Dialects Be Steered Like Languages?* — Elozeiri et al., 2026, [arXiv 2607.03936](https://arxiv.org/abs/2607.03936) — combines neuron and directional steering.

---

## Conventions

- **Filename:** `_steering.md` (leading underscore — taxonomy).
- **Folder placement:** `interpretability/` root — steering is filed here because the same directions that control the model are evidence about how it represents behavior. Some techniques (weight editing) blur into `safety/` and `post-training/`.

# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Modify a model's behavior at inference time by **adding (or projecting toward) a concept direction in hidden state**. Compute a steering vector — often a difference-of-means between positive and negative concept-bearing prompts — and inject it at chosen residual-stream layers during the forward pass. No weight updates. The 2026 angle-norm analysis (Aparin & Gaintseva) shows concepts are carried mostly in *angle*, but interventions that change *norm* are what destabilize generation — the two effects must be parameterized separately to get predictable steering.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [sae.md](sae.md) · [../safety/refusal-suppression.md](../safety/refusal-suppression.md)

---

## What it is

Take a frozen LM. Pick a target concept (refusal, sycophancy, a persona, a topic). Collect $N$ pairs of prompts that differ only in expressing the concept or not. At a chosen layer $\ell$ and token position $t$, compute the difference of mean hidden states:
$$
v_{\text{steer}} = \mathrm{mean}(h_\ell^{(+)}) - \mathrm{mean}(h_\ell^{(-)})
$$
At generation time, for each forward pass through layer $\ell$, add $\alpha \cdot v_{\text{steer}}$ to the residual stream. Positive $\alpha$ amplifies the concept; negative $\alpha$ suppresses it.

This is **representation engineering** in the narrow sense: a non-finetuning intervention that exploits the linear-probe-ability of concepts in hidden space.

## How it works

Decompose any additive intervention into an **angular** component (rotating the hidden state's direction) and a **radial** component (changing its norm). The Aparin–Gaintseva controlled study across 7 LLMs shows:

- Concepts are encoded primarily in **angular structure** — the hidden state's direction relative to a learned concept direction is what classifiers and steering interventions actually use.
- Hidden-state **norm** carries little concept information but **strongly affects downstream stability**: norm changes propagate through LayerNorm and modulate every later layer's effective output.

That gives a cleaner parameterization. Instead of one scalar $\alpha$ that mixes both effects, specify:
- $\theta$ — the angular rotation toward the concept direction (in radians, or as cosine similarity target),
- $\rho$ — the norm change (multiplicative factor or additive in log-norm).

Spherical steering (project onto the sphere of the original norm, then rotate) makes $\theta$ a pure angle. Additive steering with a normalized $v_{\text{steer}}$ mixes both. The norm choice is now a *deliberate* knob, not an accident.

## Why it matters

- **No fine-tuning.** Activation steering is the cheapest way to bias a frozen model. Useful when weights are not modifiable (closed APIs, deployed models).
- **Auditable.** A steering vector is a single hidden-state-sized object you can inspect, ablate, and compose.
- **The angle–norm framing resolves contradictions.** Prior work disagreed on whether norm carries concept information; the answer is "concept signal lives in angle, but stability lives in norm, so don't ignore norm just because it's not concept-relevant."

## Gotchas & tricks

- **Layer choice matters.** Steering at very early or very late layers behaves differently — mid-layers tend to give the cleanest concept handles. Sweep $\ell$.
- **Norm runaway.** Naive additive steering grows the residual-stream norm step by step, which silently breaks later attention. Either project to the original norm (spherical) or scale the addition by current norm.
- **Difference-of-means is one estimator.** PCA, contrastive activation addition, and learned probes all give steering vectors. The angle-norm decomposition applies to all of them.
- **Concept directions are not always linear.** Sometimes concepts are conditional on context — a single global steering vector then fails on out-of-distribution prompts.
- **Generalization is narrow.** A steering vector learned on one prompt family often fails on syntactically distant prompts that should trigger the same concept.

## Sources

- Paper: *A Geometric Account of Activation Steering through Angle–Norm Decomposition* — Aparin & Gaintseva, Huawei Noah's Ark / QMUL, 2026 — arXiv 2606.06735 — the angle / norm decomposition.
- Paper: *Steering Language Models with Activation Engineering* — Turner et al., 2023 — the difference-of-means activation-addition recipe (early reference).
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023 — broader frame.

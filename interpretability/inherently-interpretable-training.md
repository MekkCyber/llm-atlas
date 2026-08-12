# Inherently Interpretable Training (Steerling)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Bake **interpretability as a training-time constraint** alongside the standard language-modeling loss, instead of excavating a trained model post-hoc with SAEs or probes. The resulting model has natively disentangled representations that support concept attribution, training-data retrieval, and corrective steering with no additional interpretability training. Steerling-8B stays competitive with peer models while using 2–16× less compute.

**Prereqs:** [README.md](README.md)
**Related:** [sae.md](sae.md), [activation-steering.md](activation-steering.md)

---

## What it is

The dominant interpretability recipe today: train a model normally (LM loss), then bolt on Sparse Autoencoders (SAEs), probes, or activation steering to reverse-engineer the resulting representations. This is expensive (SAE training is a large project of its own) and lossy (SAE reconstructions never fully match the underlying activations).

The inverse framing: add an **interpretability co-objective to the LM training loss itself** so that the model's own representations satisfy the disentanglement properties SAEs try to recover after the fact. Concepts become first-class basis vectors in the model's activations, not post-hoc reconstructions of them.

## How it works

Exact loss composition varies, but the shape is:

```
L_total = L_LM  +  λ · L_interp
```

`L_interp` is an interpretability-inducing regularizer on intermediate activations. Instantiations in the Steerling family:

- **Concept-basis disentanglement.** Encourage activations to align with a discrete concept basis learned jointly with the LM — the basis vectors act like SAE features, but they're the model's actual computation, not a reconstruction of it.
- **Attribution consistency.** Enforce that the model's own gradient-based attributions align with a small set of concepts per output — makes the concept basis the *natural* explanation for a completion.
- **Retrieval-friendly representations.** Structure representations so that "which training examples influenced this output" is a nearest-neighbor query, not an influence-function computation.

At inference, the trained model supports:
- **Concept attribution** — read off which concept dimensions fired for a given completion.
- **Training-data retrieval** — nearest-neighbor lookup into an index of training examples.
- **Corrective steering** — clamp / scale specific concept dimensions to steer generations, no additional intervention training needed.

## Why it matters

- **Interpretability by construction, not by excavation.** Sidesteps the SAE reconstruction gap and the compute cost of separate SAE training runs.
- **Scales positively.** The paper argues the interpretability constraint doesn't hurt scaling laws — Steerling-8B is competitive with peer models trained on 2–16× more compute.
- **Steering is native.** Post-hoc steering vectors need calibration per model, per layer, per concept. Inherently interpretable models expose steering as a first-class API.
- **Deployment implications.** Native attribution and retrieval let a deployed model explain its own outputs and cite the training examples it drew from — with no additional runtime cost.

## Gotchas & tricks

- **λ tuning is load-bearing.** Too small → constraint doesn't bite, model reverts to normal representations. Too large → LM loss suffers. Ablate carefully.
- **Concept basis must be pre-specified or co-learned.** Fully pre-specified bases (from a taxonomy) limit expressiveness; co-learned bases (like SAE features) risk drift. Steerling's exact choice here is one of the paper's key design decisions.
- **Not a full substitute for SAEs.** Post-hoc SAEs still expose features the training-time basis missed. The two are complementary: interpretability-by-construction gives you the primary concept axes; SAEs excavate the residual.
- **Retraining cost.** Existing frontier models can't be retrofitted with a training-time constraint without retraining. This is a "next model" technique, not a fix for deployed models.
- **Concept-basis choice is a governance question.** Whoever picks the basis picks what the model can be steered along. Different from SAE-post-hoc, where the basis emerges from the trained model.

## Sources

- Paper: *Scaling Inherently Interpretable Language Models* — Madsen, Abdelsalam Ismail, Nguyen, Plant, Chaudhary, Monson, Azim, Guo, Adebayo (Guide Labs), 2026.
- Related: [sae.md](sae.md) for the post-hoc alternative this paper aims to displace or complement.
- Related: [activation-steering.md](activation-steering.md) for the steering primitive this paper makes native.

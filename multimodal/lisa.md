# LISA — Likelihood Score Alignment

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A drop-in regularization loss for **dual-branch controllable generation** (ControlNet-style): a frozen main diffusion / flow network plus a trainable side branch encoding the conditioning. LISA re-interprets that architecture through the score-based lens — main net = prior unconditional score, side net = implicit likelihood score — and **explicitly aligns** the side branch's intermediate feature to an approximated likelihood-score target. Speeds convergence and improves quality with zero inference-time cost.

**Prereqs:** [../architectures/_flow-matching.md](../architectures/_flow-matching.md)
**Related:** [danceopd](danceopd.md), [visual-tokenization](visual-tokenization.md)

---

## What it is

Modern controllable image and video generation typically wires a small **side network** that encodes the conditioning (depth, edge, pose, sketch) and *injects* its intermediate features into the frozen main generator. This dual-branch design is the workhorse behind ControlNet, T2I-Adapter, and most production conditional flows.

The unsolved part: the side branch is trained only through the standard diffusion/flow MSE on the main output. The side network's *features* are pushed to whatever shape happens to improve that loss; there is no signal that says "these features should encode the conditional likelihood."

LISA adds that missing signal.

## How it works

Score decomposition of conditional generation:

$$
\nabla \log p(x_t \mid c) = \underbrace{\nabla \log p(x_t)}_{\text{prior, main net}} + \underbrace{\nabla \log p(c \mid x_t)}_{\text{likelihood, side net}}.
$$

In a frozen-main + trainable-side architecture, the main net already approximates $\nabla \log p(x_t)$. By Bayes, the side net's *contribution* should approximate $\nabla \log p(c \mid x_t)$ — the likelihood score. LISA enforces this:

1. **Hook a side-network feature.** Choose a designated layer $\ell$ inside the side branch; extract its activations.
2. **Project to score space.** A lightweight decoder $D_\phi$ maps the hooked feature to a tensor in the same shape as the diffusion/flow score.
3. **Construct a likelihood-score target.** Approximate $\nabla \log p(c \mid x_t)$ — in practice, by combining the joint score (from the main + side path) with the prior score (main alone), exploiting the decomposition above. The target requires no extra ground-truth labels.
4. **Regularize.** Add $\mathcal{L}_{\text{LISA}} = \|D_\phi(\text{feat}_\ell) - \text{target}\|^2$ to the standard diffusion/flow loss.
5. **Train jointly.** Side branch + decoder optimize against the combined objective.

At inference the decoder is dropped and the main + side architecture runs as before — **zero extra cost**.

## Why it matters

- **Theoretical anchor for a years-old heuristic.** Dual-branch conditional generation has been deployed for years without a clean reason why; LISA gives one (score decomposition) and a way to act on it.
- **Faster convergence.** Across multiple image/video tasks, architectures, and diffusion + flow backbones, LISA shortens training time to match-or-beat baseline quality.
- **Better disentanglement.** Forcing the side feature to track the likelihood-score nudges representations to be more conditional-axis-disentangled, easing downstream interpretability / editing.
- **No inference cost, no architecture change.** Plug-and-play across the dual-branch controllable family. The training-only nature makes it a strong default ingredient.

## Gotchas & tricks

- **Choice of hook layer matters.** Too early → features are still input-shaped; too late → features are already entangled with the main output. The paper picks a middle layer; this is a tunable.
- **Target construction needs a frozen main net** — applies cleanly to the ControlNet recipe; if you also fine-tune the main net, the "prior score" reference drifts.
- **Decoder is lightweight on purpose.** A bigger decoder over-fits to the joint score and stops giving useful regularization.
- **Composes with other side-branch tricks** (zero-conv init, adapter scaling, cross-block injection). LISA is orthogonal to those — it only constrains the feature *content*, not the injection path.
- **Generalizes to non-image modalities.** The score-decomposition argument doesn't depend on image data; the technique transfers to video / audio / 3D dual-branch generators.

## Sources

- Paper: *LISA: Likelihood Score Alignment for Visual-condition Controllable Generation* — Wang, Chen, Liu, He, Liu, Wang, Chen, 2026 — [arXiv:2606.27192](https://arxiv.org/abs/2606.27192) — HKUST / Huawei Research.
- Background: *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)* — Zhang, Rao, Agrawala, 2023 — the canonical dual-branch recipe.
- Background: *Classifier-Free Guidance* — Ho & Salimans, 2022 — the closest existing way to surface the likelihood term implicitly.

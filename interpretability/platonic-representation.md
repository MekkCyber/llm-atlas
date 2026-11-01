# Platonic Representation Theorem (JEPA Alignment)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Joint-Embedding Predictive Architectures (JEPAs) learn the spectral embedding of their positive-pair graph. If two JEPAs trained on different modalities observe the same world and their positive pairs relate the same world states, their learned representations are the same up to an orthogonal rotation — the **Platonic Representation Theorem**. The paper distinguishes *good views* (two observations of one scene) from *bad views* (corruptions of one observation); good-view JEPAs land on the Platonic manifold, bad-view ones do not, and next-token prediction behaves like a bad-view JEPA.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

The Platonic Representation Hypothesis (Huh et al. 2024) said scaled-up models of different modalities converge to a shared representational geometry. It was intuitive but informal. This paper gives it a theorem and makes the sufficient condition actionable: by choosing *good-view* positive pairs during training, you force the JEPA into the Platonic manifold; otherwise the representation wastes dimensions on modality-specific nuisance.

## How it works

**JEPA as spectral embedding.** A JEPA trains an encoder $f_\theta$ and predictor $g_\phi$ to minimize $\|f_\theta(x) - g_\phi(f_\theta(x'))\|^2$ on positive pairs $(x, x')$. The paper proves that at convergence $f_\theta$ is (up to rotation) the spectral embedding of the graph whose nodes are inputs and whose edges are positive pairs.

**Good views vs. bad views.**
- *Good view:* $x$ and $x'$ are two *observations* of the same underlying state (e.g. an image and a caption of the same scene). The positive-pair graph's structure encodes the world's state manifold.
- *Bad view:* $x'$ is a *corruption* of $x$ (random mask, random crop, noise). The graph encodes corruption structure as much as scene structure.

**Platonic theorem.** If JEPAs on modalities $A$ and $B$ both use good views and both observe the same world, their spectral embeddings coincide up to an orthogonal $Q$: $f_A(x_A) = Q\, f_B(x_B)$.

**Next-token prediction as bad-view JEPA.** Next-token training treats $x_{<t}$ as a "corruption" of $x$ (prefix observation); the induced positive-pair graph is corruption-shaped. Alignment to other modalities is thus bottlenecked.

**Experimental payoff.**
- A 109M-parameter good-view text JEPA out-aligns LMs up to 72B on the paper's cross-modal alignment benchmark.
- Alignment follows a scaling law with model size and training.
- Stitching a frozen vision encoder and a frozen good-view text JEPA via a single orthogonal rotation (one small learned matrix) produces fluent captions without additional training.

## Why it matters

The theorem turns "pick a self-supervised objective" from an aesthetics question into a geometry question with a known sufficient condition. It also offers a mechanistic story for why text-image alignment is unexpectedly expensive under next-token prediction, and why contrastive / caption-paired signals work: they are the good-view structure the theorem requires. For multimodal practice, the frozen-rotation-captioner result is a striking existence proof that *representation* is the bottleneck, not sequence modeling.

## Gotchas & tricks

- **Good views require paired observations.** Augmentation-only positives (random crops of one image) are bad views.
- **Captions-as-good-views works for text.** The theorem's conditions are met when each (image, caption) pair is treated as two observations of the same scene.
- **"Up to rotation" is literal.** Alignment downstream usually requires fitting one orthogonal matrix per modality pair — one-shot, no end-to-end tuning needed.
- **Scaling matters even for good-view JEPAs** — a 109M model dominates 72B LMs on alignment but still improves with scale.
- **Not the same as CLIP.** CLIP is contrastive; a JEPA is predictive. CLIP under this frame is a good-view *contrastive* instantiation; the paper's claims generalize beyond that.

## Sources

- Paper: *Proving and Improving the Platonic Representation Hypothesis with JEPAs* — (authors not shown on HF page), 2026 — [arXiv:2610.39302](https://arxiv.org/abs/2610.39302).
- Reference: *The Platonic Representation Hypothesis* — Huh et al., 2024 — the original (informal) hypothesis.
- Reference: *I-JEPA* and *V-JEPA* — the JEPA family of self-supervised objectives.

# Sparse Autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **sparse autoencoder** is a shallow network trained to reconstruct a transformer's internal activations from a small number of active features out of a large learned dictionary. Each dictionary row ("feature") is a direction in activation space; the hope is that these directions are human-interpretable and more monosemantic than individual neurons. SAEs are now a standard tool for mechanistic interpretability on LLMs: trained once over a stream of model activations, then used to decompose activations into interpretable features, steer behavior, or build safety-relevant readouts.

**Prereqs:** [../interpretability/README.md](../interpretability/README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../safety/_jailbreaks.md](../safety/_jailbreaks.md), [../safety/mismatched-generalization.md](../safety/mismatched-generalization.md)

---

## What it is

An SAE is an autoencoder of the form

$$
z = \mathrm{act}(W_e x + b_e), \qquad \hat{x} = W_d z + b_d,
$$

trained on activations $x$ sampled from a specific residual-stream site in a frozen transformer. The hidden dimension of $z$ is much larger than $\dim(x)$ (e.g. 32× or more), and training adds a sparsity penalty so only a small number of entries of $z$ are nonzero per input. The rows of $W_d$ are the **features**; the active entries of $z$ for a given $x$ are the input's **feature activations**.

## How it works

- **Objective.** Reconstruction loss $\|x - \hat{x}\|^2$ plus a sparsity penalty. Early SAEs used L1 on $z$; TopK-SAEs and JumpReLU-SAEs use a hard activation to enforce exact top-$k$ sparsity.
- **Where it attaches.** Typically to the residual stream after a block, sometimes to MLP outputs. Each site needs its own SAE.
- **Training data.** Activations collected over a large, diverse text corpus; SAEs are trained offline, the base model is frozen.
- **Reading features.** For each dictionary row, collect the top activating inputs and the row's logit effect; label features by the pattern.
- **Using features.** Three main uses: (1) interpretation (what does a model attend to for X?), (2) steering (clamp a feature on/off at inference), (3) safety readouts (classify prompts/responses from a small feature set).

## Why it matters

Direct neuron analysis fails because single neurons are polysemantic (each fires for unrelated concepts). SAEs provide a *redundant, overcomplete* basis in which features are hoped to be monosemantic. Anthropic's "Scaling Monosemanticity" (2024) showed this hope is partially realized at scale and that steerable features exist. SAEs are the common substrate for a growing set of safety and interpretability techniques — jailbreak analysis, feature-level steering, auditing, deception detection — and are the base layer for techniques like the Representation Redistribution Score ([rss.md](./rss.md)).

## Gotchas & tricks

- **Dead features and reconstruction bias.** Many dictionary rows end up never activating; others under-reconstruct. TopK/JumpReLU variants help but don't eliminate the issue.
- **Not causally grounded by default.** A feature that *correlates* with a concept need not *cause* model behavior around it. Ablation + patching tests are the standard follow-up.
- **Site matters.** Features at layer 10's residual stream are not the same as layer 25's; multi-site analysis is often needed to track a concept.
- **Interpretation is semi-automated at best.** LLM auto-labeling of features is noisy; human spot-checks remain necessary.

## Sources

- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham et al., 2023 — early SAE results on LMs.
- Paper: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — SAEs at frontier-model scale, steerable features.
- Paper: *Scaling and Evaluating Sparse Autoencoders* — Gao et al., OpenAI, 2024 — TopK-SAE training recipe.
- Paper: *Representation Redistribution in Sparse Autoencoders* — Ma, UW–Madison, 2026 — [arXiv:2610.21240](https://arxiv.org/abs/2610.21240) — relational analysis of SAE features under paraphrase.

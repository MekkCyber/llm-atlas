# Hypernetwork-Generated LoRA
*Depth — generate LoRA adapters from context with a learned hypernetwork instead of training them per task.*

**TL;DR:** A LoRA adapter is normally trained per task or per repo by gradient descent. A hypernetwork-LoRA replaces that training with a single forward pass: a hypernetwork reads the target context (a repo, a domain, a user) and *emits the LoRA `A` and `B` matrices* directly. The base model is unchanged; the conditioning lives entirely in the generated low-rank update. **Code2LoRA** (Hotsko et al., 2026) instantiates this for code LMs and adds a recurrent variant that rolls the adapter forward as a repo evolves commit-by-commit.

**Prereqs:** [post-training/fine-tuning/README.md](../fine-tuning/README.md), [post-training/_post-training.md](../_post-training.md)
**Related:** [post-training/rejection-sampling.md](../rejection-sampling.md), [pre-training/mid-training.md](../../pre-training/mid-training.md)

---

## What it is

LoRA expresses a weight update as $\Delta W = B A$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$. Standard LoRA finds $A, B$ by gradient descent on a task. The hypernetwork variant instead defines a function

$$
H_\phi: \mathrm{context} \mapsto \{ (A_\ell, B_\ell) \}_{\ell \in \mathrm{adapted\ layers}}
$$

and trains $\phi$ end-to-end so that the base model + generated adapter does well on a *distribution* of contexts. At deployment, the adapter is the output of one hypernetwork forward pass — no per-context optimisation, no inference-time prompt token cost.

## How it works

1. **Encode the context.** Code2LoRA summarises a repository (files, signatures, READMEs) into a fixed-size embedding using a small encoder. Other instantiations could use a dataset summary, a user profile, or a retrieved cluster.
2. **Decode adapter weights.** The hypernetwork maps the encoding to per-layer $(A_\ell, B_\ell)$ pairs. Output dimensionality is `2 × n_layers × r × d`; weight tying across layers is a common trick to keep the head manageable.
3. **Train on context-task pairs.** $H_\phi$ is trained by sampling a context, generating its adapter, applying it to the base, and minimising the task loss on examples from that context. Gradients flow back through the base model into $\phi$.
4. **Evolving variant (Code2LoRA-Evo).** A GRU state tracks the repo through its commit history; new commits update the state, which re-emits an adapter. This amortises adapter generation across time instead of regenerating from scratch.

## Why it matters

- **Inference-time context cost goes to zero.** Repo knowledge that would otherwise sit in a 100K-token prompt becomes a fixed-size weight delta.
- **Generalisation to unseen contexts.** Because the hypernetwork is trained over a *distribution* of repos/users/tasks, it produces useful adapters on new ones without retraining.
- **A clean answer to context drift.** The recurrent variant is one of the few principled stories for keeping a model current with a living codebase without nightly fine-tunes.
- **Composable.** Hypernetwork outputs are still standard LoRA adapters: you can merge them, swap them, or stack multiple (one per repo, one per user).

## Gotchas & tricks

- **Capacity vs. rank.** A hypernetwork can only express the adapter family its decoder spans. If $r$ is too small, the generated adapters can't capture per-context variation; too large and you overfit.
- **Context encoder is the bottleneck.** Garbage summary in, garbage adapter out. Code2LoRA spends real engineering on the repo summariser.
- **Training cost is non-trivial.** Backprop through the base model on each step is expensive — this trades inference cost for training cost.
- **Cold-start contexts.** A repo with little signal (empty, brand-new) produces a near-identity adapter; that's the right behaviour, but verify it on small contexts before shipping.
- **Catastrophic forgetting of base.** Same risk as any LoRA: the adapter can degrade general capabilities for the target context. Code2LoRA mitigates by keeping the adapter low-rank.

## Sources

- Paper: *Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution* — Hotsko, Li, Deng, Nie (University of Waterloo), 2026 — [arXiv:2606.06492](https://arxiv.org/abs/2606.06492).
- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021 — the underlying adapter parameterisation.
- Paper: *HyperNetworks* — Ha et al., 2016 — the original hypernetwork idea.

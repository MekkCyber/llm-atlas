# Hypernetwork-Generated LoRA
*Depth — generate per-target LoRA adapters from a hypernetwork instead of training them.*

**TL;DR:** Per-domain specialization usually means either *(a)* shoving the domain into the prompt via retrieval, paying a per-token tax forever, or *(b)* fine-tuning a LoRA per domain, paying a training tax per target. Hypernetwork-LoRA collapses both: a hypernetwork sees the *target description* (a codebase, a user profile, a deployment context) and **emits LoRA weights in one forward pass**. Inference uses those weights without any retrieval prompt overhead. Code2LoRA (Hotsko et al., 2026) demonstrates the recipe for code: ingest a repository, emit a repo-specific LoRA adapter, beat both RAG and per-repo fine-tuning on RepoPeftBench.

**Prereqs:** [README](README.md)
**Related:** [../grpo.md](../grpo.md), [../rejection-sampling.md](../rejection-sampling.md)

---

## What it is

A meta-learner: hypernetwork $H_\phi$ maps a **target descriptor** $c$ to LoRA weights $(\Delta A, \Delta B)$ that are then applied to a base LLM $\theta$:

$$
(\Delta A, \Delta B) = H_\phi(c) \qquad \theta_{\text{adapted}} = \theta + \Delta B \Delta A
$$

The base LLM weights stay frozen; the hypernetwork is trained once; per-target adapters are produced in a single hypernetwork forward pass, no per-target optimization.

## How it works

```
Training:
  loop over (target_descriptor c, target_task t):
      ΔA, ΔB = H_φ(c)
      loss   = base_LLM_loss(t, applying ΔA, ΔB)
      ∇φ → update H_φ
  base LLM frozen throughout

Inference:
  given new target c*:
    ΔA*, ΔB* = H_φ(c*)
    apply to base LLM → specialized model
    serve with zero retrieved-context overhead
```

Variants from Code2LoRA:

- **Static.** One pass: hypernetwork sees a snapshot of the target (e.g. the repo at HEAD), emits one LoRA. Good for stable targets.
- **Evolutionary.** Conditions on a diff history (repo commits), emitting adapters that track *target evolution*. Important for actively-developed codebases where a frozen adapter would rot.

Two design knobs:

- **Descriptor encoder.** The hypernetwork is really descriptor-encoder + weight-projector. The encoder reads the target (repo, profile, context) into a compact embedding. The projector maps that embedding to the LoRA shape.
- **Rank choice.** As with normal LoRA, choosing rank $r$ trades specialization capacity vs. hypernetwork output dimension. Code2LoRA uses standard small ranks (8–32).

## Why it matters

- **Amortizes specialization.** Per-target fine-tuning doesn't scale to a Python ecosystem's worth of repos; per-target retrieval inflates the prompt forever. A learned hypernetwork pays once.
- **Zero serving-time prompt overhead.** Adapter weights replace retrieved-context tokens. For agentic coding systems where prompt budget is precious, this is a real win.
- **Target evolution as a first-class signal.** Conditioning on diff history (Code2LoRA-Evo) lets adapters track moving targets — the kind of thing a frozen LoRA can't do.

## Gotchas & tricks

- **Hypernetwork capacity is the ceiling.** A small hypernetwork outputs limited-rank adapters; expressivity per target is bounded. Scale hypernetwork together with diversity of targets.
- **Descriptor design matters.** "What does the hypernetwork see about the target?" is the dominant lever. For code: file structure + key APIs is usually enough; raw token dump rarely is.
- **Mode collapse across targets.** If many targets look similar, the hypernetwork can ignore $c$ and emit a generic adapter. Regularize with a contrast term (different $c$'s should yield meaningfully different adapters).
- **Composition with retrieval is allowed.** Hypernet-LoRA isn't a strict replacement for RAG — combining a hypernetwork adapter with light RAG for fresh facts is a reasonable serving stack.
- **Versioning.** Adapters generated from older hypernetwork checkpoints need re-generation when the hypernetwork is retrained, since the base manifold shifts.

## Sources

- Paper: *Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution* — Hotsko, Li, Deng, Nie, U. Waterloo, 2026 — [arXiv:2606.06492](https://arxiv.org/abs/2606.06492) — introduces the Static and Evo variants and the RepoPeftBench evaluation.
- Related: classical hypernetworks (Ha et al., 2016) — the spiritual ancestor; the LLM-LoRA instantiation is what's new here.

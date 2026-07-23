# Spectral inheritance in RLVR
*Depth — the phenomenology that RLVR reuses base-model singular values, and ISO, an optimizer built on it.*

**TL;DR:** RLVR post-training substantially preserves the base model's weight *spectra* (singular values) and acquires new behavior mainly through changes in the associated input/output *singular frames* (singular vectors). ISO turns this into an optimizer stack that concentrates updates in the frame-rotation subspace — an RLVR-native, LoRA-adjacent parameterization.

**Prereqs:** [rlvr.md](./rlvr.md), [grpo.md](./grpo.md)
**Related:** [ppo.md](./ppo.md)

---

## What it is

"Spectral inheritance" is a claim about *what changes during RLVR*: not much of the singular value distribution, and a lot of the singular vector geometry. In matrix terms, if `W_base = U Σ V^T`, then after RLVR fine-tuning `W_post ≈ U' Σ V'^T` — the values `Σ` come along largely for the ride, while `U, V` rotate.

If the claim holds, it explains a lot: why RLVR preserves so much base-model capability, why LoRA-style low-rank additions work at all, and why full-rank updates spend most of their budget on things the model already "knew how to do."

## How it works

**Phenomenology (measurement).** Perform SVD on weight-matrix deltas between base and RLVR-fine-tuned checkpoints. Show that (i) `Σ_post − Σ_base` is small in a controlled sense, (ii) the singular *vectors* rotate materially, (iii) the induced behavior change is well-captured by the frame rotation alone.

**ISO (algorithm).** An RLVR-native optimizer that:
1. SVD-decomposes each targeted weight matrix.
2. Effectively freezes or lightly touches the singular values.
3. Concentrates the update on the singular vectors — the frame rotation is what carries the new behavior.

This is a physically-motivated parameterization: the optimizer's degrees of freedom match the geometry the empirical measurement says the update actually needs.

## Why it matters

- **Explanatory.** RLVR "preserves base capabilities" isn't magic — it's structural. The values that encode most of a matrix's magnitude are inherited.
- **Compression / lightweight adapters.** A frame-only adapter is smaller than LoRA at equal expressivity along the observed axis of variation.
- **Optimizer design.** ISO is a concrete instance of matching update geometry to observed structure. That template outlives any single paper.

## Gotchas & tricks

- "Spectral inheritance" is a claim about *typical* RLVR behavior at moderate KL to base. It might break down under aggressive RL where large behavior shifts require large magnitude shifts.
- SVD on huge weight matrices is not free — ISO amortizes it via update batching, but the compute story is a real engineering item.
- Empirically shown on the RLVR setting; whether SFT or DPO exhibit the same inheritance is an open question.

## Sources

- Paper: *ISO: An RLVR-Native Optimization Stack* — Zhu et al., 2026 — [arXiv:2607.19331](https://arxiv.org/abs/2607.19331)
- Prior analysis (referenced by ISO): Zhu et al. 2025.

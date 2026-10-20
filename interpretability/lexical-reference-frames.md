# Lexical Reference Frames (LexFrame)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Every hidden state inside a transformer has a **native reference frame** — the context it was computed from, plus the layers downstream that will read it. Fixed lenses (logit, tuned) force a single frame on all states and fail at mid-depth. **LexFrame[k]** instead reads a hidden state through the model's **own remaining layers**, with $k$ extra rounds of attention between positions, giving a parameter-free, context-aware token-space readout.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** _(none yet — interpretability folder is being seeded)_

---

## What it is

Linear probes, the logit lens, and tuned lenses all project a hidden state at layer $\ell$ to vocabulary space with a *fixed* mapping. This works late in the stack, where the model's own residual stream is already in output frame, and fails in the middle, where the state is in some intermediate representation the output head cannot read.

LexFrame asks: *why not use the model's own downstream layers as the lens?* Freeze a hidden state at layer $\ell$, run it forward through the remaining layers of the actual model (optionally with $k$ extra rounds of self-attention over positions), then unembed. The lens is **parameter-free** (no probe to train, no tuner to fit) and **context-aware** (attention at the lens step sees the whole sequence, not just this one state).

## How it works

**LexFrame[k].** Given a hidden state $h^{(\ell)}_t$ at layer $\ell$, position $t$:
1. Insert it into a frozen sequence of hidden states for the full prompt at layer $\ell$ (the "scaffold").
2. Run $k$ rounds of the model's own layer-$\ell$ attention + FFN over the full scaffold, so the state at $t$ can attend to its neighbors as the model would downstream.
3. Continue through layers $\ell+1, \ldots, L$ of the model to produce a vocabulary distribution.

**k = 0** is the logit lens with the full remaining stack. **k = 1** is the paper's sweet spot: one round of attention between positions is enough to recover readable semantics mid-depth; more rounds help marginally and cost compute.

**Multi-use.** The same lens supports attribution (ablate a state at $\ell$ and read the lens output), steering (modify $h^{(\ell)}_t$, re-lens, verify change), and early exit (if LexFrame[1] at layer $\ell$ already matches the output, exit).

## Why it matters

- **Readable mid-depth lensing.** On SmolLM3-3B and OLMo3-7B (including training checkpoints), LexFrame[1] recovers readable concepts at layers where logit lens and tuned lens are illegible.
- **Zero-parameter, portable.** Nothing to train, so results transfer across checkpoints, scales, and families without re-fitting a probe.
- **Native to the model's semantics.** The lens reads a state through the model's own downstream reader, which is the "ground truth" of what that state means to this model.

## Gotchas & tricks

- **Compute cost.** Each lens call runs part of the model forward; batch many positions at once rather than one-by-one.
- **Scaffold matters.** The surrounding sequence changes attention patterns at the lens step; keep the real prompt as the scaffold (not a padding token).
- **k > 1 is rarely worth it.** The marginal gain from more rounds drops fast; k = 1 is the recommended default.
- **Not a substitute for mech-interp circuits.** LexFrame tells you *what a state looks like to the model;* it does not identify *which heads/neurons wrote it.*

## Sources

- Paper: *Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics* — Anonymous, 2026 — [arXiv:2610.22169](https://arxiv.org/abs/2610.22169)
- Prior art: Logit Lens (nostalgebraist, 2020); Tuned Lens (Belrose et al., 2023).

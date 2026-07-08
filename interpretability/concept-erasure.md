# Concept Erasure

*Depth — remove a target concept from a representation while preserving everything else it encodes.*

**TL;DR:** Given a hidden representation $h$ and a target concept (e.g., "gender," "sentiment," "author identity"), edit $h \to h'$ so a downstream probe can no longer recover the concept, but every *other* concept it encoded is preserved. Load-bearing primitive for interpretability audits, fairness/debiasing, safety-side dangerous-concept removal, and controlled generation. Recent methods (LEACE, MANCE) work as one-shot linear projections or manifold-constrained iterative updates; the axis of comparison is the **leakage-vs-surgicality tradeoff** — how completely the concept is scrubbed vs how much collateral damage is done.

**Prereqs:** [README.md](README.md)
**Related:** [safety/README.md](../safety/README.md)

---

## What it is

The setup: pick a hidden layer of a trained model, extract representations $h \in \mathbb{R}^d$ on a probe dataset, and label each $h$ with the target concept $c$. An erasure operator $f: \mathbb{R}^d \to \mathbb{R}^d$ satisfies:

- **Erasure guarantee**: any probe $g(f(h))$ cannot predict $c$ (measured under a chosen probe class — linear, MLP, nonlinear).
- **Surgicality**: for any *other* concept $c'$ correlated with $c$ but distinct, a probe on $c'$ still succeeds on $f(h)$ as well as it did on $h$.

Perfect surgicality is impossible when $c$ and $c'$ are perfectly correlated in the data — the erasure has to sacrifice something. The methods compete on the tradeoff curve.

## How it works

Two families of methods:

- **Linear closed-form.** Fit a linear predictor of $c$ from $h$, then project $h$ onto the null space of that predictor. Fast, deterministic, guaranteed against *linear* probes but not nonlinear ones. Canonical: LEACE (Belrose et al., 2023), which finds the minimum-Wasserstein-shift linear projection that removes $c$ under all linear probes.
- **Iterative nonlinear.** Repeatedly train a classifier of $c$, apply small gradient steps to $h$ that reduce classifier accuracy. Handles nonlinear probes but can overshoot the manifold of natural representations and damage unrelated features.

MANCE (2026) adds a **manifold constraint** to the iterative family: estimate the manifold of natural representations from a bank of clean inputs, then project each erasure update to that manifold's tangent before applying it. The Manifold Constraint Hypothesis (MCH) is the empirical claim that the collateral damage of iterative erasure comes from stepping *off* the natural manifold — projecting back onto it recovers the surgicality of closed-form methods while keeping the erasure strength of iterative ones. MANCE++ prepends a closed-form pass before the manifold-constrained iteration.

## Why it matters

- **Interpretability probes with real causal grounding.** Erasure lets you ask "if this concept were removed, does downstream behavior change?" — a stronger claim than correlation-only probing.
- **Fairness debiasing.** Remove protected-attribute information from a representation before it feeds a decision head. LEACE-style pipelines are the workhorse here.
- **Safety-side concept scrubbing.** Erase representations tied to dangerous knowledge (weapons synthesis, malware). Related to activation-editing but with a stronger claim: full nonlinear erasure across a probe class.
- **Controllable generation.** In diffusion / VLM settings, erasing a concept from cross-attention keys is a cheap way to enforce refusal.

## Gotchas & tricks

- **Probe class matters.** Linear-only guarantees don't stop MLP probes from recovering the concept. Always evaluate against a stronger probe class than the one used for training.
- **Correlation ≠ target.** If the training data doesn't disentangle $c$ from $c'$, erasing $c$ inevitably damages $c'$. Diversify probe data.
- **Manifold estimation is fragile.** Manifold-constrained methods depend on the natural-representation bank being *natural*. Contaminated banks lead to on-support erasure that misses the target.
- **Composes with itself.** Erase A, then B, then C — but errors compound. Batched multi-concept erasure objectives outperform sequential ones when concepts are entangled.
- **Watch the leakage benchmark.** Papers often report leakage on the training split; a strong methodology reports leakage on held-out generations at the token level.

## Sources

- Paper: *MANCE: Manifold Aware Concept Erasure* — Avitan, Goldberg, Elazar, 2026 — manifold-constrained iterative erasure; state-of-the-art nonlinear leakage.
- Paper: *LEACE: Perfect Linear Concept Erasure in Closed Form* — Belrose et al., 2023 — canonical closed-form linear-probe erasure.
- Paper: *Null It Out: Guarding Protected Attributes by Iterative Null-space Projection (INLP)* — Ravfogel et al., 2020 — iterative linear predecessor.

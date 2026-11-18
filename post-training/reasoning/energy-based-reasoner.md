# Energy-Based Reasoner (EBR)
*Depth — recurrent/looped reasoning recast as gradient descent on a learned energy function over latent reasoning states.*

**TL;DR:** An Energy-Based Reasoner (EBR) replaces the opaque loop of a recurrent reasoner with an explicit energy landscape: a network **E(z, x)** scores candidate latent reasoning states, and each reasoning iteration is a gradient-descent step **z ← z − η ∇_z E**. The energy's value doubles as a quality score, its gradient magnitude as a dynamic stopping criterion. A dedicated **exploration-based training** procedure shapes the landscape so it has useful basins even on unseen problems. Reported: **99.9% on Sudoku-Extreme with 345× less inference compute** than the next-best model, plus state-of-the-art coverage of distinct valid solutions on four algorithmic benchmarks.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [README.md](README.md)
**Related:** [length-penalty.md](length-penalty.md), [mcts.md](mcts.md), [../../inference/loopfill.md](../../inference/loopfill.md)

---

## What it is

Looping has been used as an inductive bias for reasoning for years (universal transformers, Huginn, Ouro). The persistent problems are: *what does a loop iteration mean?*, *when should it stop?*, and *why does it generalize?*. EBR answers all three by grounding the loop in an explicit energy function over latent states, borrowing the EBM formalism and training it with a procedure designed to prevent landscape collapse.

## How it works

Three pieces:

1. **Energy network.** A model `E(z, x; θ)` takes a latent reasoning state `z` and input `x` and returns a scalar energy. Low energy = good partial solution.

2. **Looped update (inference).** Starting from an initialization `z_0`, iterate `z_{k+1} = z_k − η_k ∇_z E(z_k, x)`. The stopping rule is `‖∇_z E‖ < τ` — stop when further descent would barely move. The final `z_K` is decoded into the answer.

3. **Exploration-based training.** Standard EBM training (contrastive divergence, score matching) collapses in high dimensions. EBR's training adds perturbed exploration: periodically re-initialize `z` from a mixture of random, replay-buffered, and nearby-perturbed states, and apply the gradient step to each. This widens the useful basin and keeps the landscape informative outside the training distribution.

Compared with long-CoT RL, which spends *tokens* on reasoning, EBR spends *gradient steps on a continuous latent* — the loop is in the model's hidden space, not in text.

## Why it matters

- **Principled halting.** No learned "halt" head and no fixed loop count — the gradient norm is the stop signal, which falls out of the formulation.
- **Compute wins.** On Sudoku-Extreme the authors report 99.9% accuracy at **345× less inference compute** than the prior best. On four algorithmic reasoning benchmarks EBR matches or beats prior recurrent reasoners.
- **Solution diversity.** Because the energy landscape has multiple low-energy basins, EBR finds a broader set of *distinct valid* solutions — relevant for tasks with many admissible answers.
- **Compatible with speculative decoding.** Pairs naturally with LoopFill-style dual-axis speculation: shallow-loop predictions are exactly the predictions after a few gradient steps.

## Gotchas & tricks

- Energy-landscape training is notoriously unstable. The exploration-based training recipe is the load-bearing piece — don't drop it.
- Pick a bounded step size schedule (`η_k` decreasing) or you get oscillation around basins, not convergence.
- The gradient norm stop criterion depends on the energy's absolute scale — use a per-problem calibrated threshold, not a global one.
- Latent-space reasoning loses the auditability of CoT. For safety-critical deployments this is a real cost; CoT monitoring doesn't transfer here.
- On problems where the answer is not well-structured as "a basin to descend into" (e.g., open-ended generation) the formulation doesn't obviously help.

## Sources

- Paper: *Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model* — Gladstone, Li, Ji, Du, UIUC / Duke / Harvard, 2026 — [arXiv:2611.08977](https://arxiv.org/abs/2611.08977).
- Context: Ouro, Huginn — recurrent/looped reasoners that EBR is positioned against.
- Related: *LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers* — Zhu et al., 2026 — [arXiv:2611.19649](https://arxiv.org/abs/2611.19649).

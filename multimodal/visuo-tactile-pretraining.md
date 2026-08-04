# Visuo-Tactile Pretraining for VLA / World-Action Models
*Depth — jointly pretraining a foundation model over vision, tactile signals, and actions at scale.*

**TL;DR:** Vision-language-action (VLA) foundation models have historically stopped at vision + language, treating tactile as domain-specific noise. Visuo-tactile pretraining tokenizes tactile signals as a first-class modality — the N₀ family uses **NeoForce**, a unified force-based tactile representation — and jointly trains a world / action model that predicts future vision, tactile, and action simultaneously. Contact events become explicit staging signals for long-horizon manipulation.

**Prereqs:** [asymmetric-mot.md](../architectures/asymmetric-mot.md)
**Related:** [text-conditioning-scaling.md](./text-conditioning-scaling.md)

---

## What it is

A pretraining recipe that puts tactile perception on equal footing with vision inside a large multimodal foundation model. Two concrete instantiations from the N₀ family in this line of work: **N₀-TWAM** (Tactile-native World-Action Model — predicts future vision + contact) and **N₀-VTLA** (Vision-Tactile-Language-Action — adds a language pathway and offline RL policy improvement).

## How it works

**1. Unified tactile representation (NeoForce).** Raw tactile sensors output arrays that differ across embodiments. NeoForce normalizes them into a physically grounded force-based representation shared across sensors, so the model doesn't need to relearn tactile semantics per embodiment.

**2. Joint visuo-tactile pretraining.** Train over demonstrations that pair video + tactile + action trajectories. Losses:
- Next-frame video prediction (world modeling).
- Next-tactile prediction (contact prediction).
- Next-action prediction (behavior cloning).

**3. Staged tactile-pathway integration (VTLA variant).** Bolt the tactile pathway on top of an existing visual backbone in stages, so the model doesn't destabilize when tactile is added.

**4. Contact events as task stages.** During execution, detect discrete tactile contact events (grasp made, contact lost) and use them as gate conditions that advance a long-horizon task through its stages.

**5. Downstream training (VTLA variant).** Advantage-conditioned offline RL on stored deployment trajectories improves the policy without full retraining.

## Why it matters

- Tactile signals unlock contact-rich manipulation that vision-only VLAs handle poorly. The scaling curves that vision enjoyed (more data → better predictions) now demonstrably apply to tactile as well.
- Contact events as staging signals give long-horizon policies natural checkpoints, alleviating some of the credit-assignment burden on the policy itself.
- The NeoForce representation is a portable interface — the tactile equivalent of a shared image tokenizer.

## Gotchas & tricks

- Tactile data at scale is expensive; the "data at scale" bar requires 6+ embodiments and hundreds of tasks in practice.
- Real-time inference requires the asymmetric expert widths of [asymmetric-mot.md](../architectures/asymmetric-mot.md) — a naive wide-across-modalities MoT is too slow to close the control loop.
- Cross-embodiment transfer is real but not automatic; the physics of contact differ enough that per-embodiment fine-tuning still helps.

## Sources

- Paper: *N₀-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation* — NeoteAI, 2026 — [arXiv:2607.23783](https://arxiv.org/abs/2607.23783)
- Paper: *N₀-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens* — NeoteAI · Fudan, 2026 — [arXiv:2607.23782](https://arxiv.org/abs/2607.23782)

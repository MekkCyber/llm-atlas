# Self-Editing LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A sufficiently large LLM, asked to reason about its own weights, can emit closed-form edits that **improve itself without gradients, rewards or external tools**. He et al. (2026) show a 27B model gains +6.6 accuracy points on a held-out, pre-registered test set from its own edit; given a corrupted copy of itself, it diagnoses the broken matrix and patches it, and the repair transfers from a text-math probe to image benchmarks. The ability emerges at scale (2B does nothing, 4B/9B regress, 27B improves).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md), [activation-steering.md](activation-steering.md)
**Related:** [circuit-compilation.md](circuit-compilation.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

A non-gradient post-training mode: the model treats its own weight matrices as reasoning objects and proposes rank-structured edits to them. Edits are closed-form (a tensor delta the compiler applies), not learned by SGD. The model's natural-language reasoning is the search signal; no teacher, reward model or verifier is in the loop.

## How it works

1. **Exposure.** The model is given access to a description of its own architecture and a handle to specific weight matrices (names, shapes, a way to read/write them through a tool interface).
2. **Reasoning prompt.** Ask: *"Identify which matrix is miscalibrated for problem X, and propose an edit that fixes it."* The model emits a textual plan plus a structured edit (low-rank delta, bias shift, etc.).
3. **Apply.** The edit is applied as-is (no fine-tuning). Evaluation runs on held-out problems.
4. **Repair variant.** Corrupt one block of a *copy* of the model (zero a matrix, add noise). The original model is asked to locate and repair the damage. He et al. find the repair signal found on text math *also* restores accuracy on image benchmarks of the corrupted copy.

## Why it matters

- **Collapses model-vs-agent boundary.** The model is simultaneously the artefact being edited and the agent editing it.
- **Emergence signal.** Capability is sharply scale-dependent — 2B inert, 4B/9B harmful, 27B helpful — making this a candidate capability for scaling-law / safety-case tracking.
- **Converges with two adjacent lines.** Paper #3 (RL post-training in DiT = activation steering) says RL's effect concentrates in a low-rank shift; paper #4 (weight-space programming) trains vision networks to be edit-ready. Together they sketch a non-gradient post-training stack.

## Gotchas & tricks

- **Scale cliff.** Smaller models emit edits that *harm* themselves. Don't extrapolate downward.
- **Pre-registration matters.** The +6.6 pt gain was on a paired, pre-registered 490-problem test — treat unregistered self-improvement claims sceptically, variance is high.
- **Transfer is suggestive, not understood.** Math-prompt-induced repairs restoring image accuracy point to something deeper than task-specific patching, but the mechanism is open.
- **Safety case.** Self-editing is a load-bearing capability for autonomous-AI oversight: edits must be auditable before application, not merely observed after.

## Sources

- Paper: *Language Models Can Rewrite Their Own Weights* — He, Yang, Mao, Zhu, Cheng, Liang, Shen, Li, Guo, Liu (UIUC / CMU / Rutgers / Princeton / Stanford), 2026 — primary source; 27B self-edit experiments, repair-transfer finding, emergence curve.
- Paper: *Training Vision Models for LLM-Driven Weight-Space Programming* — 2026 — complementary: train the target to be edit-ready.
- Paper: *A CPU Inside an LLM* — 2026 — complementary: compile exact circuits into pretrained weights.

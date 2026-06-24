# HydraHead
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hybrid attention at the **head level**, not the layer level. HydraHead uses interpretability-based probing to identify which attention heads are *retrieval-critical* for long context, keeps those on full attention, and downgrades all the others to linear attention. Achieves Qwen3.5-2B-class long-context performance with only **15B training tokens** and >69% improvement on NIAH at 512k context vs. uniform linear attention.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md), [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Linear-attention hybrids so far have been *layer-granular*: keep N layers fully-attentioned, replace the rest with linear attention. This is conservative — it assumes every head in a "full" layer needs full attention, and every head in a "linear" layer can do without.

HydraHead observes that **heads inside the same layer specialise** — some are needle-in-a-haystack retrieval heads doing the heavy lifting on long context, others are local pattern matchers that linear attention handles fine. Going head-granular lets you keep the small fraction of heads that really need quadratic attention and put the rest on the cheap kernel.

## How it works

1. **Probe to identify retrieval-critical heads.** Use a controlled long-context probe (e.g., NIAH-style) on a small pretraining checkpoint. Measure each head's contribution; rank heads by impact.
2. **Assign attention kernels.** The top heads (the "Hydra" heads) keep full softmax attention. The rest are replaced with a linear-attention kernel (the paper uses a state-space-style linearised form).
3. **Train.** The hybrid model is trained from scratch or warm-started from a small dense checkpoint, with the kernel assignment frozen.

The retrieval-critical heads continue to receive the full per-head budget; the linear-attention heads run in the cheap state-space-style kernel. The mix happens *within a single attention layer*.

## Why it matters

- **Much cheaper long context for the same quality.** Quadratic attention is paid only on the heads that need it — typically a small fraction of total heads.
- **Evidence-based, not ablation-based, head selection.** The probing step turns "which heads matter?" into a measurement instead of a guess.
- **Sample-efficient training.** Reaching Qwen3.5-2B-class quality on 15B tokens demonstrates that head-level hybridisation is a real lever, not just an inference-time optimisation.

## Gotchas & tricks

- **Probe quality dominates.** A weak probe will misidentify heads and downgrade ones the model later relied on. The paper uses NIAH-style probes; other long-context tasks might pick a different head set.
- **Kernel choice for the linear heads is non-trivial.** Different state-space-style kernels have different generalisation properties; the paper's specific kernel is a hyperparameter.
- **The head split should be frozen after probe-time selection** — letting the assignment shift during training breaks the gradient story.
- **Doesn't compose freely with GQA-style head sharing** — sharing KV across heads conflicts with running different kernels per head. The paper sticks to MHA.

## Sources

- Paper: *HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization* — anonymous, 2026 — [arXiv:2606.20097](https://arxiv.org/abs/2606.20097).
- Background: linear-attention / state-space hybrids (Mamba, RWKV, RetNet) — the kernel family HydraHead inherits its cheap-attention slot from.

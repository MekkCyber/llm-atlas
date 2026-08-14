# Neuron Plasticity Tuning (MLLM Language Preservation)
*Depth — per-neuron update constraints during multimodal SFT to keep language capability from regressing.*

**TL;DR:** Multimodal expansion of a pretrained LLM routinely tanks language benchmarks — grafting a vision encoder + running SFT overwrites neurons that carried linguistic competence. NeuPAT (2026) measures each neuron's **plasticity during multimodal training** via a small probe, then applies neuron-wise update-magnitude constraints — protect language-critical neurons, permit large updates on the ones that adapt well to vision. Recovers **94.5%** of the language regression from vanilla SFT while keeping comparable multimodal performance.

**Prereqs:** [../multimodal/README.md](README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../post-training/README.md](../post-training/README.md)

---

## What it is

A fine-tuning modifier that treats plasticity as a **per-neuron budget** rather than a global learning rate, layer freeze, or LoRA rank. The observation motivating it: neurons in a pretrained LLM are *heterogeneously* plastic under downstream tuning — some are highly language-specific (updating them destroys pretraining), others are relatively free (updating them absorbs multimodal information without collateral damage). Vanilla SFT ignores that split; NeuPAT enforces it.

## How it works

Three stages:

1. **Probe.** A small probing run over language-only and multimodal-only data measures each neuron's contribution to language capability vs. its adaptation to multimodal input. This yields a per-neuron *language-sensitivity* score.
2. **Allocate.** Convert the sensitivity scores into update-magnitude constraints — high-sensitivity neurons get a tight cap, low-sensitivity neurons get essentially free updates. Architecture-agnostic; sits on top of any optimizer.
3. **Train.** Run standard multimodal instruction tuning with the per-neuron constraints applied. The constraint layer is lightweight and adds negligible overhead per step.

## Why it matters

- Multimodal expansion's language-capability tax is one of the least-discussed but most consequential drawbacks of building VLMs by grafting a vision encoder onto a strong LLM.
- Existing defenses — layer freezing, LoRA-only tuning, replay of language data — are all *coarse*. Per-neuron plasticity is much sharper and orthogonal to those knobs.
- Complements catastrophic-forgetting literature on continual learning: the same framing (which parameters are safe to move?) but scoped to the specific SFT-for-multimodal use case.
- Architecture-agnostic — plugs into diverse LLM families without model surgery.

## Gotchas & tricks

- **Probe budget is not free.** The probing stage costs a small but real fraction of the SFT budget; skimping on it produces noisy sensitivity scores and defeats the point.
- **Language benchmarks vs. multimodal benchmarks decorrelate.** Optimizing only for multimodal metrics hides regressions on language benchmarks — always evaluate both.
- **Neuron-level ≠ head-level.** The heterogeneity is at the neuron granularity; averaging to attention heads or layers loses most of the signal.
- **Composes with LoRA and QLoRA.** The plasticity budget can be applied to LoRA update magnitudes just as well as full-parameter updates.
- **Not a substitute for good data mixtures.** Even with plasticity constraints, an SFT set with no language-only examples will drift; keep a small language-replay share.

## Sources

- Paper: *NeuPAT: Neuron-aware Plasticity Allocation Tuning for Language-Preserving MLLMs* — Jin, Zhang, Wang, Liu, Guo, 2026 — [arXiv:2608.08107](https://arxiv.org/abs/2608.08107) — the per-neuron plasticity probe and constraint framework.

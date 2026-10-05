# HeteroFold — Prefill-Free Cross-Family KV Cache Transfer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A method for passing a KV cache between LLMs of **different families** (different tokenizers, depths, KV layouts) *without* re-prefilling on the receiver side. HeteroFold aligns the two models' structures, maps the sender cache into the receiver's KV space, and calibrates it to preserve the receiver's output distribution — both models stay frozen.

**Prereqs:** [../architectures/mla.md](../architectures/mla.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../agents/graphforge.md](../agents/graphforge.md), [../agents/belief-states.md](../agents/belief-states.md)

---

## What it is

Yun et al. (USC / UC Berkeley, 2026) attack the main cost in heterogeneous multi-agent LLM systems: every time one agent hands context to another from a different model family, the receiver has to **prefill** the shared context from scratch (text in, KVs out). HeteroFold instead ships the sender's **KV cache** directly into the receiver, bypassing prefill, while handling the fact that the two models speak different "KV languages."

## How it works

Three stages, all with both models frozen:

1. **Align model structures.** Learn a layer-to-layer alignment between sender and receiver (different depths, different KV head counts, different hidden dims). The alignment is lightweight — a per-layer linear mapping, trained offline for each (sender, receiver) pair.
2. **Map the sender cache into receiver space.** Apply the learned mapping to project the sender's cached keys and values into the receiver's representation space. Handles tokenizer differences by first re-mapping token boundaries.
3. **Calibrate** to preserve receiver behavior. A small calibration pass adjusts the mapped KVs so that the receiver's next-token distribution over a short prompt continuation matches what it would have produced from a native prefill.

At inference, the sender passes its KV cache; the receiver loads it (post-mapping + calibration) and generates immediately — no prefill forward pass over the shared context.

## Why it matters

- At 32K context, Llama-3.1-8B → Ministral-3-14B transfer runs **10.7× faster** than native prefill and 1.18–1.47× faster than prior prefill-free baselines (Dense Latent, KV Ridge).
- Best cache-transfer accuracy across six transfer directions on all four long-context benchmarks and most short-context settings.
- On the multi-agent benchmark, matches text-based communication quality while eliminating the prefill tax that makes long-context multi-agent systems painful today.
- Changes the economics of orchestration frameworks that currently prefer a single model family just to avoid cross-family prefill.

## Gotchas & tricks

- The per-pair alignment is a one-time offline cost; adding a new model into the pool means training new alignments against the existing ones.
- Calibration quality depends on the length and representativeness of the probe sequence used — too short and the mapped cache drifts on later tokens.
- Attention variants with heavier KV compression (MLA, GQA) change the shape of what's being transferred; the paper's experiments cover standard MHA and GQA — MLA receivers are a mentioned open direction.
- The method keeps both models frozen; for scenarios where you can fine-tune, learned "cross-model projection" layers may do better.

## Sources

- Paper: *Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs* — Yun, Lim, Yoo, Yoo, Annavaram, Karimireddy; USC / UC Berkeley, 2026 — [arXiv:2609.32259](https://arxiv.org/abs/2609.32259).
- Context: *Dense Latent*, *KV Ridge* — prior prefill-free KV transfer baselines.
- Context: *vLLM / PagedAttention* — Kwon et al., 2023 — the KV cache management substrate HeteroFold's mapped caches plug into.

# Latent Chain-of-Thought
*Depth — perform intermediate reasoning in compact continuous states instead of (or alongside) verbalized text tokens.*

**TL;DR:** Explicit CoT forces every intermediate thought through the discrete token bottleneck of the LM head — costly, and forces semantic, uncertain, or partially-formed updates to be prematurely verbalized. Latent CoT keeps intermediate states *continuous*, only committing to text when needed. The line started with Coconut (2024); 2026's **NF-CoT** is the first variant that preserves all four properties that made explicit CoT useful — left-to-right generation, probabilistic sampling, KV-cache compatibility, and tractable likelihoods — by modeling continuous thoughts with a normalizing flow inside the LM backbone.

**Prereqs:** [_rl](../_rl.md), [long-cot-rl](long-cot-rl.md)
**Related:** [length-penalty](length-penalty.md), [long2short](long2short.md)

---

## What it is

Explicit CoT pipeline:

```
prompt → t_1 → t_2 → … → t_n → answer
```

Each `t_i` is a discrete token sampled from the LM head. Each step pays:

- Sequential decoding (no batching across thought steps).
- Information loss: hidden state (4096-dim continuous) → vocab logits → token.
- Forced verbalization of half-formed thoughts.

Latent CoT replaces some / all of the `t_i` with **continuous thought positions** `z_i ∈ R^d` that stay in the residual stream and never round-trip through the LM head. Reasoning proceeds in latent space; the final answer (and any "speak aloud" intermediate text) is emitted via the standard LM head.

## How it works

| Method | Continuous-state mechanism | KV cache | Tractable likelihood | Probabilistic sampling |
| --- | --- | --- | --- | --- |
| **Coconut** (2024) | Inject prior hidden state directly as the next embedding | ✓ | ✗ (deterministic) | ✗ |
| **CoCoMix / SoftCoT** | Mix of soft-prompt tokens during training | ✓ | partial | partial |
| **NF-CoT** (2026) | TARFlow-style normalizing flow as a second decoder head | ✓ | **✓ exact** | ✓ |

NF-CoT keeps two heads on the same causal stream:

- **LM head** — produces text tokens at text positions.
- **NF head** — at designated continuous-thought positions, samples `z_i` from a tractable density modeled by a normalizing flow (TARFlow-style).

Continuous thoughts are *distilled* from explicit CoT traces during training, grounding the flow's target distribution. Inference: alternate (or interleave) continuous and discrete positions; the shared causal mask + KV cache keeps everything autoregressive and cacheable.

## Why it matters

- **Higher reasoning bandwidth per token.** A continuous `z` carries more information than a single vocab token.
- **Lower compute per reasoning step.** Bypassing the LM head saves a vocab-size matmul per latent step.
- **Trainable end-to-end.** With tractable likelihoods (NF-CoT), policy-gradient methods like GRPO/RLVR apply *in latent space* — opening RL on latent reasoning, which Coconut couldn't support cleanly.
- **Empirical wins.** NF-CoT improves pass rate on code-generation benchmarks over both explicit CoT and prior latent baselines, with substantially fewer intermediate-reasoning tokens.

## Gotchas & tricks

- **Interpretability cost.** Latent thoughts aren't human-readable. For safety-critical use (CoT monitoring), keep at least some explicit thought tokens.
- **Distillation source still needed.** All current variants train against explicit-CoT teacher traces to ground the latent distribution. Pure-latent training from scratch hasn't been demonstrated.
- **Mode collapse on the flow.** Without enough teacher diversity, the NF can collapse to a low-variance mode and behave like Coconut (no probabilistic sampling). Variance regularization helps.
- **Tokenization mismatch.** Designating which positions are "continuous" requires either special control tokens or a learned scheduler — both add a tiny amount of structural overhead.

## Sources

- Paper: *Latent Reasoning with Normalizing Flows* (NF-CoT) — Fu et al., 2026 — [arXiv:2606.06447](https://arxiv.org/abs/2606.06447) — primary source for the NF variant.
- Paper: *Coconut: Training Large Language Models to Reason in a Continuous Latent Space* — Hao et al., 2024 — first widely-discussed latent-CoT proposal.
- Paper: *TARFlow* — Zhai et al., 2024 — backbone normalizing-flow architecture used in NF-CoT.

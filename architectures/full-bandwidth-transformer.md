# Full-bandwidth transformer

*Depth — widening the vertical feedback channel between decoding steps by re-injecting the previous top-layer hidden state alongside the sampled token embedding.*

**TL;DR:** Autoregressive transformers compute on two axes: horizontal (across generated tokens via attention) and vertical (through depth). The vertical channel between successive decoding steps is currently just **one sampled token wide** — the top-layer hidden state is thrown away after sampling. The **full-bandwidth transformer** (Wang et al., 2026) fuses the previous top hidden state with the next input embedding through a gated linear unit (GLU) and feeds the sum into the bottom of the stack, widening that channel from 1 token to a full hidden vector. The change is parameter-light, training-friendly, and orthogonal to attention design.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [sliding-recurrent-memory](sliding-recurrent-memory.md), [mla](mla.md)

---

## What it is

In a standard autoregressive transformer, decoding step $t+1$ receives:

- The sampled token $x_{t+1}$ (an embedding, ~$d$ dimensions).
- Everything else via the KV cache at each attention layer.

The **top-layer hidden state** $h_t^{(L)}$ that produced the logits for $x_{t+1}$ is discarded — the model gets to see only the one sampled index at the bottom of the next step, even though $h_t^{(L)}$ contained the full posterior over what should come next.

Full-bandwidth transformer keeps that hidden state around. At step $t+1$, the bottom-layer input is:

$$
\tilde{x}_{t+1} = \text{GLU}(x_{t+1} \Vert h_t^{(L)})
$$

where $\Vert$ is concatenation and GLU is a gated-linear-unit fusion. Nothing else changes.

## How it works

- **One extra module per model** (not per layer) — the GLU that fuses embedding + previous top-hidden-state. Parameter count grows by ≈ $O(d^2)$, which is a fraction of a single MLP block.
- **Training.** During teacher forcing, hidden states from the *previous* time step's forward pass are cached and re-injected on the next step. Because the extra dependency is only 1 step back, training remains almost fully parallel — you compute the whole sequence in one shot, then run a pass that injects $h_{t-1}^{(L)}$ at each step. The paper describes a two-pass or staggered variant to keep this efficient.
- **Inference.** Cost per step is one GLU + one add, on top of the standard transformer forward — negligible.
- **Attention is untouched.** Full-bandwidth is orthogonal to full/MQA/GQA/MLA/hybrid choices; you can stack it on any attention variant.

## Why it matters

- **Widens the recurrent state without abandoning parallel training.** All prior "widen the vertical channel" work (RNNs, RWKV, state-space) traded parallelism for it; full-bandwidth keeps the parallel-training story of the transformer intact.
- **Cheap to add.** No new attention pattern, no per-layer changes, no restructuring of the KV cache. Trivial to bolt onto an existing pretrained model as a fine-tuning objective.
- **Consistent gains at matched size.** The paper reports perplexity and downstream improvements at multiple scales, at effectively zero inference-time cost.

## Gotchas & tricks

- **Two-pass staggering during training.** The naive dependency chain "hidden at $t-1$ enters input at $t$" serializes decoding through depth. Practical implementations use a staggered / delayed feedback so that the network still trains in one parallel pass.
- **Cache management at inference.** You now need to keep the last top-layer hidden state (a $d$-vector), which is negligible relative to the KV cache but must not be dropped between calls.
- **Interaction with quantization.** The feedback path carries a dense signal — aggressive quantization of the top layer's activations can distort what the bottom of the stack sees on the next step. Calibrate the top layer's outputs alongside the KV cache.
- **Not a substitute for long-context recall.** Widening the *step-to-step* channel doesn't help retrieve information from 10k tokens back; that's still the KV cache's job.

## Sources

- Paper: *Full-bandwidth Transformer* — Wang, Cai, Zhan, Dong, Fan, de Rosa, Pearce, Langford (Microsoft Research), 2026, [arXiv:2608.08888](https://arxiv.org/abs/2608.08888)

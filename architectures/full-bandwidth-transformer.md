# Full-bandwidth Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A one-line architectural change that widens the *vertical* channel between decoding steps. Standard transformers throw the top-layer hidden state away and only feed the sampled token embedding back to the bottom of the stack. Full-bandwidth transformers fuse the previous top-layer hidden state with the next token embedding via a **gated linear unit**, giving non-verbalized computation a renewed depth budget across steps. Same params, same KV cache, same LM loss.

**Prereqs:** [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [multi-head-attention](multi-head-attention.md), [mtp](../pre-training/mtp.md), [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

An autoregressive transformer computes on two axes: horizontally across tokens (attention), vertically through layers. The horizontal channel is wide — every past token is visible through attention. The vertical channel between decoding steps is one scalar wide: only the id of the sampled token comes back. The top-layer hidden state — which is where the model's actual continuous computation lives — is discarded after each step.

The full-bandwidth transformer routes that top-layer hidden state back in. At step $t+1$, the input becomes a gated fusion of the sampled token embedding $e_{t+1}$ and the previous top hidden $h_t^L$. Everything else about the architecture — attention, FFN, KV cache, cross-entropy objective — is unchanged.

## How it works

At each decoding step, given sampled token embedding $e_{t+1}$ and previous top-layer hidden $h_t^L$:

$$
x_{t+1} = \mathrm{GLU}(e_{t+1}, h_t^L) = (W_e e_{t+1}) \odot \sigma(W_g h_t^L) + b
$$

The GLU gate lets the network decide, per unit, how much latent context to admit. $x_{t+1}$ is then fed to layer 1 as if it were an ordinary token embedding.

The problem is training. Latent feedback breaks parallel teacher forcing — the "previous" hidden state doesn't exist yet under a single forward pass. The paper's answer is a **scheduled multi-pass objective**:

- Standard teacher-forcing loss dominates most of pretraining.
- Late in pretraining, introduce latent feedback: a second pass that uses hiddens from a first pass as the previous-step latent.
- Mix in a small fraction of deeper multi-pass steps (feeding pass $k$'s hiddens into pass $k+1$) for training stability.

At inference the added cost is negligible: one GLU per token, and $h_t^L$ was already computed for the sampling head.

## Why it matters

- **Structural compute efficiency.** 1B-param full-bandwidth transformers trained to 400B tokens **match or approach standard transformers trained with ~1.5× more tokens** across validation loss, 5-shot LM eval, math and code generation, and instruction tuning.
- **Shorter reasoning traces.** At equal or better accuracy, the model produces shorter chains — because "silent" continuous computation can now propagate across steps without being verbalized.
- **Composable.** Same objective, same KV cache, same block. Slots into existing training stacks with a two-line architectural change plus a schedule.

## Gotchas & tricks

- **Introduce feedback late, not from step 0.** Early in pretraining the top hidden is uninformative; teaching the input side to consume it starves the bottleneck. The schedule matters.
- **Deep multi-pass fraction is small.** A little goes a long way for stability; too much wastes compute and destabilizes the LM loss.
- **KV cache is unchanged** — the only new state is one hidden vector per step, and it's transient (only $h_t^L$ needs to survive to step $t+1$).
- **GLU gate, not raw addition.** Simple residual addition of $h_t^L$ to $e_{t+1}$ underperforms — the gate is the difference between "feeds latent signal usefully" and "corrupts the input distribution".

## Sources

- Paper: *Full-bandwidth transformer* — Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, John Langford (Microsoft Research), 2026 — [arXiv:2608.08888](https://arxiv.org/abs/2608.08888).

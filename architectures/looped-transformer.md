# Recurrent Looped Transformer (RLT)

*Depth — split a Transformer into a parallel causal encoder and a recurrent decoder that loops depth-wise over the sequence, so compute path length grows with sequence length at fixed per-token cost.*

**TL;DR:** A standard Transformer has a fixed number of layers L and therefore a fixed depth of computation per token, independent of sequence length. State-tracking tasks (parity, S₅ permutation composition) need depth to grow with length, which Transformers cannot do without stacking layers linearly. RLT splits its layers into a **parallel causal encoder** (processes the full sequence in one shot) and a **recurrent decoder** (feeds its hidden state back on every token). The decoder adds one effective "depth step" per token, so computation depth = sequence length at **O(L) per-token cost**.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md)

---

## What it is

Two components in series:

- **Encoder** $E$: a standard causal Transformer stack, run once over the input sequence in parallel. Produces per-position encodings.
- **Decoder** $D$: a small Transformer-style block with weight sharing across time. For each output position $t$:
  - takes the encoder output at $t$ and the previous decoder hidden state $h_{t-1}$,
  - produces $h_t$ and the next-token distribution.

The decoder's recurrent hidden state is what makes the effective depth grow with sequence length.

## How it works

The decoder update:

$$
h_t = D(h_{t-1},\; E(x)_t)
$$

Over a sequence of length $N$, the state $h_N$ depends on all $N$ decoder applications, each applying the same block weights. This is a weight-shared loop over depth-time: $N$ decoder steps = depth $N$ at inference, but only one set of weights to train.

Three design knobs:

- **Encoder depth** — standard L; stays fixed.
- **Decoder loop granularity** — one hidden-state update per token (highest expressivity), or one update per k-token chunk (parallelism).
- **Feedback dimension** — $h$ can be a single vector or a bank of memory slots.

### Why depth matters for state-tracking

Parity (compute $\sum x_i \bmod 2$) and permutation composition in S₅ are **circuit depth $\Theta(\log N)$** problems in the uniform-$\mathrm{TC}^0$ sense. A fixed-depth Transformer cannot implement them for arbitrary $N$. RLT's decoder loop can.

## Why it matters

- **Length generalisation on state-tracking.** Reports perfect generalisation on parity and S₅ beyond training length, where 8-layer Transformers stay at chance.
- **Modular arithmetic.** Beats the Transformer baseline at longer sequence lengths.
- **A clean architectural lever.** Separates "parallel semantics" (encoder) from "serial state update" (decoder) — related in spirit to Universal Transformer, Mamba, and the newer loop-based reasoning architectures.

## Gotchas & tricks

- **Chunked feedback is a knob, not a free lunch.** Updating $h$ once per 4-token chunk recovers parallelism; the paper reports some accuracy loss on permutation tracking.
- **Training stability.** The weight-shared decoder loop looks like an RNN at training time — long sequences can hit gradient-signal issues; standard RNN tricks (truncated BPTT, careful init) apply.
- **Not a general Transformer replacement.** On tasks that don't need superconstant depth, the extra decoder cost is wasted.
- **Interacts with speculative decoding.** The recurrent decoder breaks parallel draft-verify in classical spec-dec; chunked feedback restores it.

## Sources

- *Recurrent Looped Transformer* — Zhang, Feng, Qin — 2026 — [arXiv:2610.07591](https://arxiv.org/abs/2610.07591)

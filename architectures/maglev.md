# Maglev
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A recurrent Transformer with **fixed-size memory** that generalizes sliding-window attention while staying parallelizable at train time. Two coupled models: a **prefiller** $Q$ with full attention produces memory targets, and a **decoder** $P$ with sliding-window attention plus recurrent K/V injection produces decoder memories. A memory-consistency loss aligns decoder memories to prefiller targets so inference can run $P$ alone. Introduced by Liu & Liu (UT Austin, 2026).

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](mla.md), [../fundamentals/dca.md](../fundamentals/dca.md)

---

## What it is

Long-context Transformers face a train/serve tradeoff:

- **Full attention:** parallelizable to train, but serving cost grows with sequence length.
- **Sliding window:** cheap and fixed-cost to serve, but loses long-range information beyond the window.
- **Latent recurrent transformers:** fixed-cost to serve *and* long-range, but hard to train — the recurrence blocks parallelism and gradients through many time steps are unstable.

Maglev decouples training from inference. At train time it runs *two* models simultaneously; at serve time it uses only the cheap one. The recurrence is trained via a teacher signal instead of backprop-through-time.

## How it works

Two networks, tied end-to-end:

- **Prefiller $Q$** uses **full attention** over the input sequence. At each position $t$ it produces a memory target vector $m'_t$ — essentially "what the ideal fixed-size summary of the past would look like at position $t$."
- **Decoder $P$** uses **sliding-window attention** plus **recurrent K/V injection**: at each step, the previous position's produced decoder memory $m_{t-1}$ is injected into the K/V state, and the model produces the current decoder memory $m_t$ and the next-token prediction.

Both models share the input sequence. Training minimizes a joint loss:

$$
\mathcal{L} = \mathcal{L}_{\text{LM}}(P) + \lambda \cdot \mathcal{L}_{\text{consistency}}(m_t, m'_t)
$$

where $\mathcal{L}_{\text{consistency}}$ (e.g. MSE or cosine) pulls the decoder's memory toward the prefiller's target. Because $Q$ uses full attention, it's parallel over the sequence and provides a strong teacher signal to the recurrent decoder without requiring backprop through the recurrence.

At inference, $Q$ is discarded — only $P$ runs, with sliding-window attention + recurrent K/V. Serving cost is constant per token; long-range information lives compressed in $m_t$.

**Parameter sharing** between $P$ and $Q$ is optional: sharing most layers reduces the training-time parameter footprint while preserving most of the accuracy gain.

## Why it matters

- **Cheap serving.** Fixed-size memory means constant cost per decode step regardless of context length.
- **Trainable.** Sidesteps the BPTT-through-long-recurrence pathology by teaching the recurrent decoder from a parallelizable prefiller.
- **Consistent gains.** Improves validation loss and downstream pretraining benchmarks over sliding-window and latent recurrent Transformer baselines.
- **Slots next to Mamba-family SSMs.** Fixed-state efficient long-context designs are a live area; Maglev's teacher-student recipe is a cleanly transferable idea to other recurrent-memory architectures.

## Gotchas & tricks

- **Prefiller compute is real.** During training, running $Q$ (full attention) alongside $P$ doubles training FLOPs unless parameters are shared. The paper reports parameter sharing preserves most of the gain.
- **Consistency loss weight $\lambda$ matters.** Too small and the decoder memory drifts; too large and the LM loss suffers. The paper's setting is a moderate value found by ablation.
- **Window size sets the "fallback range."** If the recurrent memory ever fails to encode a critical past detail, the sliding window is the only backup. Smaller windows amplify the memory's obligation to be accurate.
- **Not a magic bullet against long-range dependencies past the window.** Fixed-size memory has finite bandwidth; tasks that require verbatim recall of many past facts still favor full-attention or retrieval-augmented setups.
- **Prefiller must be strong enough.** A weak $Q$ produces weak memory targets and drags $P$ down. Use at least a strong-baseline full-attention model as the prefiller.

## Sources

- Paper: *Maglev: Sliding Recurrent Memory* — Bo Liu, Qiang Liu, UT Austin, 2026 — [arXiv:2608.02870](https://arxiv.org/abs/2608.02870).
- Related: [mla.md](mla.md) — different family (KV-cache compression) but same underlying goal of cheaper attention at long context.

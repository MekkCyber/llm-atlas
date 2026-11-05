# Gated-Embedding Training Stall
*Depth — a silent failure mode in untied gated sequence models where near-zero input embeddings cause FP32 global-norm clipping to zero out whole-batch gradients.*

**TL;DR:** ~**52.6%** of surveyed **untied** gated and hybrid language-model checkpoints contain near-zero input-embedding rows; **0%** of weight-tied checkpoints do. Mechanism: AdamW weight decay shrinks input-embedding rows for rarely-appearing tokens. In gated sequence models (Mamba, hybrids), those near-zero rows **amplify gradients** enough to overflow FP32 global-norm clipping, which **zeroes out the entire batch's gradient**. Training plateaus silently with no loss-curve signal. Two fixes: **GroupClip** (post-hoc: decouple the clipping group for input embeddings) and **GateFloor** (pretraining: add a positive offset to the gate so the amplification factor can't collapse to zero).

**Prereqs:** [_training-stability.md](_training-stability.md)
**Related:** [../fundamentals/README.md](../fundamentals/README.md)

---

## What it is

In gated sequence models (SSM-family: Mamba, hybrids like Griffin / RWKV / Mamba-2), the per-token gate $g_t$ multiplies the embedding $e_{x_t}$ at the entry to the recurrence. The gradient flowing back to $e_{x_t}$ is scaled by $1 / g_t$ (or an equivalent amplification factor from the gate design).

AdamW weight decay shrinks embedding rows $e_v$ for tokens $v$ that rarely or never appear in the training batch (because there's no counter-pressure from a token-presence gradient). After enough steps, those rows become near-zero.

When a rare token *does* appear, three things happen in one step:

1. The forward pass computes $g_t \cdot e_{x_t}$ with a near-zero $e_{x_t}$.
2. The backward pass computes a gradient scaled by the amplification factor — **huge in magnitude**.
3. The **FP32 global-norm clip** — which pools the gradient norms of all parameters into one scalar — sees this huge value, scales the global norm way down, and zeros out effectively *all* gradients in that batch.

Training continues, the loss curve looks fine on average, but roughly one in every few hundred batches contributes no gradient. Final checkpoints are mysteriously undertrained in a way that slice-level ablations can find but standard training telemetry misses.

**Why weight-tied models are immune.** With tied embeddings, the output head's gradients provide steady counter-pressure to every embedding row — a token's row gets updated whenever the model outputs that token, not only when it sees it as input. Untied models lack this pressure, so rare-token input rows decay unchecked.

---

## How it works

The paper diagnoses the problem by surveying public gated / hybrid checkpoints and measuring the $L_2$ norm distribution of input-embedding rows. The near-zero cluster is cleanly separated and correlates with the untied-vs-tied distinction.

### Fix 1: GroupClip (post-hoc, for released checkpoints)

Replace the single FP32 global-norm clip with **per-group** clipping. Put input embeddings in their own group; the amplified gradient from a stalled row only triggers a clip on the embedding group, not the whole model.

```
# Instead of:
clip_global(grads)           # single scalar
# Do:
clip_group(grads_embed, c_embed)
clip_group(grads_rest, c_rest)
```

Patched models recover their training signal and resume loss improvement on continued training.

### Fix 2: GateFloor (pretraining)

Add a positive constant $\epsilon$ to the gate (or to its pre-activation, so the gate can't become zero):

$$
g_t = \sigma(\text{pre}_t) + \epsilon
$$

The amplification factor $1 / g_t$ is now bounded. Even if an input-embedding row decays, the downstream gradient never explodes; the global-norm clip is never triggered by a stalled row; training proceeds normally.

$\epsilon \in [10^{-3}, 10^{-2}]$ is sufficient in the paper's reproductions.

---

## Why it matters

- **Silent.** Loss curves, grad norms (aggregated), and eval loss all look healthy. The failure is per-batch dropout of gradient signal that aggregates to a small but real undertrained state.
- **Common.** More than half of surveyed public gated checkpoints exhibit the pattern. Many are "released but slightly broken."
- **Cheap to fix.** Both GroupClip and GateFloor are one-line changes in a mature training stack. There's no reason to run new gated-sequence training without one of them.
- **General principle.** The pathology is a pattern (component A weakly regularized → amplification in component B → global aggregator clips → whole-batch loss of signal). Analogous interactions exist elsewhere (e.g. attention logit explosion → softmax saturation → similar global-clip triggers).

---

## Gotchas & tricks

- **Weight-tied models don't need this.** If you're using tied embeddings, GateFloor is still a mild stability win but not necessary.
- **GroupClip grouping matters.** Putting input embeddings alone in one group is the specific fix; putting them in a group with the output head defeats the purpose.
- **GateFloor's $\epsilon$ interacts with model behavior.** Too large and the gate can't close; the model loses the "forget" capacity that motivates the gate. $\epsilon = 10^{-3}$ is the paper's safe default.
- **Watch per-row embedding norms during training.** Easy telemetry addition: log the percentile distribution of $\|e_v\|_2$ every $N$ steps. A long left tail is the warning sign.
- **BF16 masks the problem differently.** FP16 overflow triggers the clip; BF16's wider exponent range delays but doesn't eliminate the stall for sufficiently near-zero rows.

---

## Sources

- Paper: *Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models* — Liu, Sun, Wang, Song, Zhao, Liu (WUSTL / Nankai, inferred from emails), 2026, arXiv 2611.04449 — primary source.

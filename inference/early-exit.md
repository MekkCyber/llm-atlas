# Early Exit (Adaptive Depth)
*Depth — self-distilled nested sub-models plus a learned per-input depth router.*

**TL;DR:** For many decisions, the information needed sits in **middle layers**, not the final layer. **Early exit** attaches a decision head to every prefix of the stack, trains all heads jointly with self-distillation from the full-depth head, and routes each input to the shallowest head that's confident enough. NestOne (2026) is the clean modern recipe: shared weights across depths, lightweight learned gate, no extra models at inference. Across seven decision benchmarks it matches or beats independently trained per-depth models at each depth while cutting inference cost.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Adaptive-depth inference for a single trained stack. One $N$-layer transformer, $N$ decision heads (one hanging off each layer's output), and a router that picks which head to use per input.

Contrast:

- **Distillation.** Train a small model separately. You get one model at one cost; you pay for every call.
- **Mixture-of-depths.** Allocate compute per-*token* within one pass. Sub-token granularity, different mechanism.
- **Early exit.** Allocate compute per-*input* across the shared stack. Coarser granularity, cheaper gate.

## How it works

### Training: shared-weight nested sub-models

Treat layers $1\!:\!k$ of the full stack as a "sub-model at depth $k$," with its own decision head $h_k$ bolted on top of layer $k$. The weights of layers $1\!:\!k$ are literally the first $k$ layers of the full stack — nothing is copied.

Train with two losses jointly:

1. **Supervised loss on the full-depth output** (the usual task loss on $h_N$).
2. **Self-distillation to every prefix head:** for each training example, the full-depth distribution $p_N$ is the target for every $p_k$, $k < N$:

$$
L_{\text{distill}} = \sum_{k=1}^{N-1} \mathrm{KL}(p_N \,\|\, p_k)
$$

The gradients from $L_{\text{distill}}$ flow through every prefix, nudging each $h_k$ toward the full-depth output. Because weights are shared, this does not inflate parameter count.

### Inference: a lightweight depth router

A small router $g(x) \to k \in \{1, \dots, N\}$ decides which depth to exit at. Trained to pick the smallest $k$ such that $h_k$'s predicted confidence exceeds a threshold. At inference, run layers $1\!:\!g(x)$ and emit $h_{g(x)}(\cdot)$.

For classification-style decisions, "confidence" is the max softmax probability. For free-form outputs, it's typically a learned scalar from $h_k$'s hidden state.

## Why it matters

- One training run gives a family of models at different depths, instead of training each one separately.
- The depth router is a per-input knob: easy inputs exit shallow, hard inputs run full depth. Static distillation can't do that.
- Real inference-cost savings in agent systems where the decision model is called on every observation. Saving half the depth on 70% of calls is a routine deployment win.

## Gotchas & tricks

- Self-distillation needs the full-depth model to be the teacher *throughout* training, not only at the end. Teacher-student lag causes shallow heads to drift.
- The router is cheap but crucial. A bad router over-escalates to full depth and the system collapses to standard inference.
- Calibration matters: the confidence threshold has to be set per-deployment on a validation set; a threshold tuned on one task doesn't transfer.
- Early-exit at mid-depth composes with speculative decoding — the early-exit model can serve as the draft model.

## Sources

- Paper: *From Large Language Models to Efficient System One Models via Adaptive Computation (NestOne)* — 2026 — https://arxiv.org/abs/2611.16321
- Related: *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference* — 2020 — earlier early-exit for encoder models.

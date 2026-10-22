# Progress-Aligned Recurrence (PAR)
*Depth — a recurrent-update reparameterization that lets state-tracking models generalize to lengths far beyond training.*

**TL;DR:** State-tracking tasks (parity, modular arithmetic, Dyck languages) require a recurrent dynamics whose update step is *time-dilation invariant*: the state after $n$ tokens should match the state after $kn$ tokens of a $k×$-stretched input. Standard RNN/SSM updates aren't invariant that way, so they fail to generalize when test sequences are longer than training ones. PAR reparameterizes the recurrence to be driven by a *progress* scalar aligned with the task-relevant pace rather than the raw time index, letting a model trained at length 50 solve the same task at length 5 000+.

**Prereqs:** [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md), [_positional-encoding](../fundamentals/_positional-encoding.md)
**Related:** [mla](mla.md), [multi-head-attention](multi-head-attention.md)

---

## What it is

A drop-in replacement for recurrent updates in RNN/SSM-style architectures that competes with attention on tasks requiring state tracking. The design target is *length generalization* without architecture-specific hacks (chunking, buffered attention, length-aware positional encodings).

## How it works

The standard recurrent update $h_{t+1} = f(h_t, x_t)$ advances state by one step per input token. PAR introduces a per-step scalar *progress* $p_t$ and reparameterizes the dynamics so that the update commutes with time dilation:

$$ h_{t+1} = f_{\text{PAR}}(h_t, x_t, p_t), \quad f_{\text{PAR}}(h, x, \alpha p) = f_{\text{PAR}}^{\alpha}(h, x, p). $$

In practice, $p_t$ is a learned or input-conditional scalar, and $f_{\text{PAR}}$ is implemented so its *generator* is affine in $p_t$ (think continuous-time ODE solved by a step of size $p_t$). Changing from one token per step to one token per $k$ steps rescales $p_t$ but preserves the final state — the property that classic state-tracking tasks need.

## Why it matters

Length generalization has been the Achilles heel of recurrent architectures competing with transformers on language. Attention-based models dodge the problem partially via RoPE/ALiBi tricks, but even those fail cleanly on state-tracking tasks. PAR is one of the first mechanistic fixes that works *without* retraining per length and *without* relying on positional-encoding gymnastics; it gives recurrent models a principled answer to "why don't you work at 10× train length?"

## Gotchas & tricks

- The progress scalar must be driven by *task-relevant* evidence; a progress proxy fixed to a timer gives up most of the gain.
- Reparameterization requires the recurrent generator to be well-defined as an ODE; stacking PAR on top of a non-affine update (e.g., GRU with multiplicative gates) is not drop-in.
- The invariance holds *in expectation* over the training distribution — pathological inputs that fool the progress predictor can still cause state drift at extreme lengths.
- PAR targets state-tracking; it does not by itself help with retrieval-style long-context tasks where attention span is the bottleneck.
- Exact-hyperparameter details are limited to what the paper publishes; practitioners should tune the progress scalar's dynamic range to the task's expected length ratio.
- Numerical stability of the generator at large $p_t$ is the dominant failure mode at extrapolation lengths; discretization step size needs to be scaled with $p_t$.
- Progress-scalar prediction can be learned end-to-end from the task loss or supervised from an oracle progress signal when one is available — oracle supervision tends to generalize further.
- Combines cleanly with vanilla attention stacks: the recurrent core handles state tracking, attention handles retrieval and long-range dependencies.

## Sources

- Paper: *PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking* — Pan Wang, Yihao Hu, Guangyuan Dong et al., Ant International + NUS, 2026 — [arXiv:2610.27628](https://arxiv.org/abs/2610.27628).

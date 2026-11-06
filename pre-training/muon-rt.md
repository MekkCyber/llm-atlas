# Muon-RT — Momentum-Free Muon via Teacher-Student Trajectory Reuse
*Depth — a Muon variant that reuses the student-teacher displacement as momentum, saving optimizer state in self-distillation.*

**TL;DR:** Muon's momentum buffer for hidden-weight matrices is a large chunk of optimizer state. In **self-distillation fine-tuning (SDFT)**, you already keep a slow-moving teacher copy of the weights, and the vector `θ_student − θ_teacher` is itself an exponentially weighted average of recent updates. **Muon-RT replaces the explicit momentum buffer with that displacement**, eliminating it for the matrix-shaped weights while keeping Muon's update direction. Result on Qwen2.5-7B-Instruct: **+2.6 avg-accuracy over AdamW with 42% less persistent memory**, **+5.4 over momentum Muon with 27% less persistent memory**.

**Prereqs:** [_lr-schedules.md](./_lr-schedules.md), [_training-stability.md](./_training-stability.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [fp8-training.md](./fp8-training.md)

---

## What it is

Muon (Keller Jordan et al., 2024) is an optimizer for the 2-D hidden weight matrices of a Transformer that replaces the elementwise Adam-style update with a **Newton-Schulz orthogonalization** of the momentum buffer: the update direction is the orthogonalized version of the running momentum $M_t$. That buffer is the same size as the weight matrix — doubling (vs. no momentum) or matching (vs. AdamW's `(m, v)`) the persistent state.

Muon-RT observes that under self-distillation, you already maintain a teacher copy $\theta^{\text{T}}$ (typically an EMA of recent students). The displacement

$$
D_t \;=\; \theta_t - \theta^{\text{T}}_t
$$

is, up to a constant, an exponentially weighted history of past updates — the same quantity Muon's $M_t$ captures explicitly. Muon-RT drops $M_t$ and feeds $D_t$ to the Newton-Schulz step instead.

## How it works

**Per matrix weight** (hidden Transformer weights; keep AdamW for embeddings, norms, biases):

1. Compute gradient $g_t = \nabla \mathcal{L}(\theta_t)$.
2. Form displacement $D_t = \theta_t - \theta^{\text{T}}_t$ (instead of maintaining $M_t$).
3. Combine current gradient with displacement: $\tilde{D}_t = \alpha g_t + \beta D_t$.
4. Newton-Schulz: $U_t = \text{NS}(\tilde{D}_t)$ — orthogonalize the update direction.
5. Step: $\theta_{t+1} = \theta_t - \eta U_t$.
6. Update teacher by EMA: $\theta^{\text{T}}_{t+1} = (1 - \tau)\theta^{\text{T}}_t + \tau \theta_{t+1}$.

The teacher EMA rate $\tau$ plays the role of Muon's momentum coefficient — fast decay ≈ small momentum.

## Why it matters

Optimizer state is a first-order concern at 7B+ scale on consumer-to-midrange GPUs. Muon is replacing AdamW in several open fine-tuning stacks because of its strong performance per flop. Muon-RT is a "free lunch when you are already doing SDFT" — no extra operation, no new buffer, the teacher is state you were keeping anyway. Fits naturally into the **RL → self-distill → ship** pipeline because the self-distillation stage is where fine-tuning sits.

## Gotchas & tricks

- Only applies when you *actually* maintain a self-distillation teacher. Plain SFT without a teacher has no displacement to reuse.
- Teacher EMA rate $\tau$ controls the *effective* momentum. Rushing the teacher toward the student (large $\tau$) effectively removes momentum; a slow teacher ($\tau \ll 1$) recovers near-Muon dynamics.
- Keep AdamW for 1-D parameters (embeddings, biases, norms) — Muon's benefit is the matrix orthogonalization.
- Interacts with LR schedule: at high LR the student can race ahead of the teacher and $D_t$ grows unbounded; clip or renormalize $D_t$.

## Sources

- Paper: *MUON-RT: Reusing the Teacher's Trajectory for Momentum-Free Muon for LLM Self-Distillation* — Das et al., 2026 — https://arxiv.org/abs/2610.41104
- Prior: Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*, 2024 — https://kellerjordan.github.io/posts/muon/

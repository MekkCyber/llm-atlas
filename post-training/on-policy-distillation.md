# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distillation applied to the *student's own* rollouts. The student generates a response; a stronger, frozen teacher scores it token-by-token; the student minimizes a reverse-KL / cross-entropy against the teacher on exactly those tokens. Combines the dense supervision of SFT/imitation with the on-policy sampling of RL — the student always trains on its own distribution, but the labels come from a smarter model.

**Prereqs:** [fine-tuning](./fine-tuning/README.md), [_rl](./_rl.md), [grpo](./grpo.md)
**Related:** [rejection-sampling](./rejection-sampling.md), [rlvr](./rlvr.md), [long-cot-rl](./reasoning/long-cot-rl.md)

---

## What it is

Three shapes of using a teacher to train a student differ in *who samples the training data*:

| Method | Who samples | Signal shape |
| --- | --- | --- |
| **Vanilla distillation (SFT-on-teacher)** | Teacher | Dense token-level cross-entropy on teacher trajectories |
| **RL / RLVR** | Student | Sparse outcome reward on student trajectories |
| **On-Policy Distillation** | Student | **Dense** token-level supervision (from teacher) on **student** trajectories |

OPD sits in the middle: on-policy like RL, dense like SFT. The teacher never generates the training text — it only *scores* what the student produced, one token at a time.

## How it works

1. Student policy $\pi_\theta$ samples a full response $o$ for prompt $q$.
2. Frozen teacher $\pi_T$ runs a forward pass over $(q, o)$, producing teacher logits at every position.
3. Loss: token-level reverse-KL (or cross-entropy) between $\pi_\theta(\cdot \mid q, o_{<t})$ and $\pi_T(\cdot \mid q, o_{<t})$, summed over the response.
4. Backprop into $\pi_\theta$. No critic, no advantage estimator, no rollouts beyond the one used for sampling.

The teacher is typically a larger, stronger model. When it's the same model with **privileged information** (a reference solution, a plan, environment feedback), the setup is called **On-Policy Self-Distillation** — the teacher is no stronger, only better informed.

## Why it matters

- **Sample efficiency.** Reported accuracy comparable to RL at a fraction of the generated tokens — the dense per-token signal is much stronger than a single outcome reward.
- **No critic, no reward model.** Simpler than PPO/GRPO to set up when a teacher exists.
- **Composes with RL.** Prompt-level gating (see TGOPD, 2609.02998) routes verifiable prompts to OPD and hard prompts to GRPO, using OPD as the cheap dense signal wherever the teacher is trustworthy.

## Gotchas & tricks

- **Reverse-KL is mode-seeking.** A confidently-wrong teacher on a specific prompt pushes the student toward that wrong mode. Distributional proxies (entropy, likelihood) do not reliably catch this — TGOPD (2609.02998) gates at the prompt level using verifier probes of the teacher.
- **Self-distillation collapses.** OPSD (self-distillation with privileged info) is known to progressively narrow the set of reasoning paths the model can produce; the "One Symptom, Three Levers" review (2608.25936) organizes fixes around token weighting, choice of privileged information, and guidance-decay dynamics.
- **Teacher GPU cycles are wasted otherwise.** During student sampling the teacher is idle. Prompt-level verification probes (TGOPD) reclaim that idle capacity — teacher GPU utilization from 9.8% to ~79% in one reported 4B configuration.
- **Not the same as sequence-level KD.** Sequence-level KD trains on teacher-sampled sequences (off-policy from the student). OPD trains on student-sampled sequences (on-policy). That difference is the whole point.

## Sources

- Paper: *Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation* (TGOPD) — 2026 — [arXiv:2609.02998](https://arxiv.org/abs/2609.02998).
- Paper: *One Symptom, Three Levers: A Critical Review of On-Policy Self-Distillation* — 2026 — [arXiv:2608.25936](https://arxiv.org/abs/2608.25936).
- Paper: *FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience* — 2026 — [arXiv:2609.03241](https://arxiv.org/abs/2609.03241) (an OPSD-adjacent method that composes self-guidance with verifier gating).

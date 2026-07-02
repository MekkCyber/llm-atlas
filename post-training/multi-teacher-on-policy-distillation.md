# Multi-Teacher On-Policy Distillation (MOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** MOPD trains an LLM to integrate multiple capabilities (math, code, tool-use, reasoning) by first running **per-domain specialized RL** to produce a set of domain teachers, then **on-policy distilling all teachers into one student on the student's own rollouts** with dense token-level supervision. Removes the cross-domain interference that plagues Mix-RL and the exposure bias that plagues Off-Policy Finetune. Deployed in the post-training of the industrial-scale frontier model MiMo-V2-Flash.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [_rl](_rl.md)
**Related:** [_distillation](_distillation.md) · [grpo](grpo.md) · [rlvr](rlvr.md) · [../pre-training/model-souping](../pre-training/model-souping.md)

---

## What it is

Frontier post-training pipelines want to combine capabilities that have been developed independently: math specialists trained with RL on verifiable math, code specialists trained with RL on unit tests, reasoning specialists trained with long-CoT RL, and so on. Historically two options:

- **Mix-RL:** shove all domains into one RL run. Cross-domain reward-scale mismatch and interference; noisy.
- **Off-Policy Fine-tune:** SFT the student on each teacher's *frozen* rollouts. Cheap, but exposure bias — the student never sees its own errors.

MOPD is a third option: **decouple** domain RL from integration. Each team ships a domain specialist independently; a final integration step distills all specialists into one deployable student.

## How it works

### Stage 1 — Per-domain RL specialists

For each domain $d$ (math, code, reasoning, tool-use, safety, …), run standard domain RL (GRPO / PPO / RLVR) on the base model $\pi_0$ to obtain a specialist $\pi_d$. Each team owns their domain — no shared reward function, no shared training curriculum, no cross-domain coupling.

### Stage 2 — Multi-teacher on-policy distillation

Train a student $\pi_\theta$ (started from $\pi_0$) so that on prompts from domain $d$, its token distribution matches $\pi_d$:

$$
L_{\text{MOPD}} = \sum_d \mathbb{E}_{q \sim D_d,\ o \sim \pi_\theta}\left[\sum_t \mathrm{KL}\!\left(\pi_d(\cdot \mid q, o_{<t}) \,\|\, \pi_\theta(\cdot \mid q, o_{<t})\right)\right]
$$

Crucially the trajectory $o$ is sampled from the *student*, not the teacher. The student sees its own errors, receives dense token-level signal, and gets one supervisor per domain — the teacher whose domain the prompt came from.

### Domain routing

A prompt is routed to its domain teacher by a lightweight classifier (domain tag, prompt-cluster ID, or a simple few-shot classifier). Ambiguous prompts can be supervised by multiple teachers with a weight vector.

## Why it matters

- **Decoupling.** Domain teams develop teachers independently and integrate at the end. No shared RL curriculum, no reward-scale coordination, no lock-step schedules.
- **Beats Mix-RL and Off-Policy Fine-tune.** On Qwen3-30B-A3B, MOPD outperforms Mix-RL, Cascade RL, Off-Policy Fine-tune, and Param-Merge, inheriting nearly all of each teacher's capability.
- **Industrial deployment.** MOPD is used in the post-training of **MiMo-V2-Flash**, giving the recipe real-world track record.
- **Composable with capability development.** Any per-domain RL recipe — GRPO, PPO, DPO variants — can produce a MOPD-compatible teacher.

## Gotchas & tricks

- **Domain routing quality matters.** A bad domain classifier sends prompts to the wrong teacher; the student gets contradictory signal. Route by held-out validation accuracy per teacher.
- **Teacher forward cost.** One teacher per token per domain. Batch across domains where prompts share a domain; keep teachers on separate serving instances.
- **Cold-start prompts.** Prompts outside every teacher's domain should either fall back to the base policy or be dropped. Don't force a specialist to supervise off-domain input.
- **Capability inheritance is not full.** MOPD claims "nearly all" of each teacher's capability; expect a small gap vs the specialist. If the gap is unacceptable for a domain, that domain probably needs a dedicated student.
- **Composes with DOPD.** In domains where the teacher sees privileged inputs (retrieval, tools), fold in DOPD-style routing to avoid privilege illusion inside that teacher.

## Sources

- Paper: *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training* — Ma et al., 2026 — the MOPD recipe; deployed in MiMo-V2-Flash.
- Paper: *DOPD: Dual On-policy Distillation* — Li et al., 2026 — same-week companion; single-teacher advantage-aware routing.
- See also: [on-policy-distillation](on-policy-distillation.md) for the single-teacher substrate.

# DanceOPD

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation framework for flow-matching image models that lets **one student** compose multiple capabilities — text-to-image, local editing, global editing — that normally conflict when packed into a single network. Each capability lives as its own velocity field over the shared flow state space; the student is trained against fields *queried on the student's own rollout states* with a simple velocity-MSE loss. The same formulation absorbs classifier-free guidance, removing CFG's 2× inference cost.

**Prereqs:** [../architectures/_flow-matching.md](../architectures/_flow-matching.md)
**Related:** [visual-tokenization](visual-tokenization.md), [lisa](lisa.md), [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md)

---

## What it is

Unified image models pack T2I, local edit, and global edit into one network, but the capabilities interfere — editing degrades T2I quality, local and global edits fight each other. DanceOPD reframes capability composition as **on-policy distillation in field space**: the teacher is a *set of velocity fields*, one per capability, defined over the same flow state $x_t$.

For a flow-matching generator the dynamics are $\dot x_t = v_\theta(x_t, t)$, where $v_\theta$ is the predicted velocity. DanceOPD makes the *teacher* a structured object: per-capability fields $v_{\text{T2I}}, v_{\text{local}}, v_{\text{global}}, v_{\text{CFG}}, v_{\text{realism}}, \dots$.

## How it works

1. **Routing.** For each training sample, sample a capability $k$ from a routing distribution (per-task or learned). Treat $v_k$ as the teacher field for this sample.
2. **One on-policy rollout state.** Sample one low-noise state $x_t^{\text{stu}}$ from the *student's* current rollout — not the teacher's marginal — at a sampled time $t$. This is the "on-policy" part: distillation targets are queried where the student actually goes.
3. **Velocity-MSE loss.**
   $$
   \mathcal{L} = \mathbb{E}\left[\, \| v_\theta(x_t^{\text{stu}}, t) - v_k(x_t^{\text{stu}}, t) \|^2 \,\right].
   $$
   No extra weighting, no schedule games — just velocity matching.
4. **Capability absorption.** Define operator-level effects as additional fields:
   - **CFG field** = $v_{\text{cond}} + s(v_{\text{cond}} - v_{\text{uncond}})$ for guidance scale $s$. Adding this as a teacher field lets the student emulate CFG with a single forward pass at inference.
   - **Realism field** = a field derived from a quality reward (e.g., from a learned reward model on flow trajectories).
5. **Single student.** All capabilities collapse into one network. At inference, capability is selected by an input flag / prompt; no extra modules to swap.

## Why it matters

- **Resolves capability interference** in unified flow-matching image models — a real production headache for image generators that want one model to do T2I + edits.
- **Halves inference cost** of guided generation by absorbing CFG into the weights. CFG normally requires two forward passes per step; an absorbed-CFG student needs one.
- **Treats operator-defined effects uniformly** as just-another-field. This generalizes beyond CFG — any inference-time correction expressible as a velocity adjustment can become a training target.
- **On-policy queries mirror the [reasoning RL pattern](../post-training/on-policy-distillation.md).** Both insist that the teacher be evaluated at the student's induced distribution — a recurring principle across generative model distillation and language-model RL.

## Gotchas & tricks

- **Routing distribution is load-bearing.** Wrong capability sampling weights either under-trains a capability or destabilizes T2I anchor quality. The paper's recipe is direct uniform-over-capabilities with realism-anchor reweighting.
- **Student must already be a flow-matching model.** The technique assumes a velocity-field parameterization. It doesn't trivially transfer to score-matching or DDPM-style generators.
- **Field definition is the engineering work.** Whether a capability has a clean velocity-field description determines whether DanceOPD can absorb it. Sharp local edits with bounded receptive fields work; arbitrary "stylize then re-render" pipelines do not.
- **Quality regressions show up at extreme guidance.** Absorbed CFG works at moderate guidance scales used in practice; very high guidance ($s > 7$) sometimes still benefits from explicit CFG passes.
- **Composes with controllable-generation regularizers** like [LISA](lisa.md) — the side-network feature alignment story is orthogonal to capability composition.

## Sources

- Paper: *DanceOPD: On-Policy Generative Field Distillation* — Zhou, Zhu, Xu, Dong, Gong, Liang, Chu, Qu, Kong, Liu, Chua, 2026 — [arXiv:2606.27377](https://arxiv.org/abs/2606.27377) — ByteDance Seed / NUS.
- Background: *Flow Matching for Generative Modeling* — Lipman et al., 2023 — the flow-matching primitive.
- Background: *Classifier-Free Guidance* — Ho & Salimans, 2022 — the operator DanceOPD absorbs as a field.

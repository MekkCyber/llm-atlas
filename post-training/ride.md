# RIDE — RL-Induced Direction Extrapolation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An on-policy distillation variant that lets the student *exceed* its RL-trained teacher. Instead of extrapolating in output (logit) space — which is noisy because the LM head attenuates representation changes anisotropically — RIDE extrapolates in **representation space** at every layer, along the residual between the teacher and its own pre-RL checkpoint. The teacher becomes a *direction*, not a destination.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md).
**Related:** [dpo.md](./dpo.md), [rlvr.md](./rlvr.md).

---

## What it is

Standard OPD compresses a teacher into a student; the student plateaus at teacher performance. "Extrapolating" variants supervise toward $\pi_t + \alpha(\pi_t - \pi_{\text{base}})$ in output space to push past the teacher. The problem: the LM head attenuates the vector $\pi_t - \pi_{\text{base}}$ anisotropically, so much of the signal from the hidden states reaches the logits at a small fraction of its weight, and the sampled-token log-probability ratios inject noise that the extrapolation amplifies — training is unstable.

RIDE's observation: RL shifts the model's **internal representations** relative to its base checkpoint, and that shift is measurable at every layer. Operate on hidden states, not logits.

---

## How it works

Let $h_\ell^{\text{teacher}}$ and $h_\ell^{\text{base}}$ be the teacher's and pre-RL checkpoint's hidden states at layer $\ell$, token $t$. Define the per-layer, per-token RL residual:

$$
\Delta_{\ell,t} = h_{\ell,t}^{\text{teacher}} - h_{\ell,t}^{\text{base}}
$$

For a student rollout, at each layer regress the student's hidden state $h_{\ell,t}^{\text{student}}$ toward a target displaced *past* the teacher along $\Delta$:

$$
L_{\text{RIDE}} = \sum_{\ell, t} \| h_{\ell,t}^{\text{student}} - (h_{\ell,t}^{\text{teacher}} + \alpha \Delta_{\ell,t}) \|^2
$$

Conditioned on a sampled trajectory, this quadratic penalty is equivalent to maximizing a **linear directional reward** defined by $\Delta$ under a trust region centered at the teacher. The objective explicitly moves the student along the RL-induced direction while limiting deviation from the teacher.

---

## Why it matters

- **Breaks the teacher ceiling.** RIDE is the only tested method whose mean exceeds the teacher across four base/RL-teacher pairs spanning different scales, architectures, and pre-training lineages.
- **Beats output-space extrapolation.** Output-space extrapolation degrades the student whenever the teacher is close to its base checkpoint (the signal is tiny and dominated by noise); RIDE still works there because the directional signal exists in hidden states even when it hasn't surfaced in logits.
- **Doesn't require re-running RL.** You pay for the teacher's RL once, then any downstream student can inherit — and extend — the RL direction for OPD cost.

---

## Gotchas & tricks

- **Need the pre-RL checkpoint.** $\Delta$ requires both the teacher and its direct ancestor; a distilled or merged teacher without a clean pre-RL state can't be used.
- **Choice of $\alpha$ is a trust-region knob.** Too large and the student overshoots; too small and you're back to vanilla OPD. The quadratic penalty keeps the student within reach of the teacher.
- **Per-layer vs. final-layer only.** Supervising only at the last hidden state recovers most of the "ceiling break" effect but costs some generality; the paper uses all layers.
- **No LM head needed for the loss.** The gradient flows through hidden-state regression; the LM head still gets its own cross-entropy signal from the teacher's logits.

---

## Sources

- Paper: *The Teacher Is a Direction, Not a Destination: Extrapolating RL-Induced Representation Residuals in On-Policy Distillation* — Li, Chen, Ren, Li, Tian, Huang, Wang, 2026. arXiv:2609.36484.

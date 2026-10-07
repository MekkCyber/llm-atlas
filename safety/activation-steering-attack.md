# Closed-Loop Activation Steering (as Attack)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion language models (dLLMs) iteratively denoise masked positions and expose answer distributions **at every step** before commitment. An adversary with access to internal activations can watch target-answer probability as the denoising progresses and **modulate a steering vector in closed loop**, producing substantially larger bias shifts than any fixed-strength open-loop steering. Hashmi et al. (2026) show up to **37 pp swing** on demographic answers with a simple PI controller.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [../interpretability/README.md](../interpretability/README.md) · [refusal-suppression.md](refusal-suppression.md) · [prefix-injection.md](prefix-injection.md)

---

## What it is

Activation steering adds a direction $v$ to residual-stream activations to bias the model toward a chosen behavior. In an autoregressive decoder, there's one chance per output: the final step reveals the committed answer's probability. In a **dLLM**, each denoising step exposes an evolving answer distribution before commitment — a side channel that lets an attacker *see* how far the current intervention has shifted the model and adjust.

This paper weaponizes that side channel. A **proportional-integral (PI) controller** tracks the target-answer probability through denoising and adapts steering strength on the fly. Closed-loop steering produces far larger bias shifts than any constant-strength baseline.

## How it works

- **Target signal.** At each denoising step $t$, read the probability $p_t$ that the model will commit to the adversary-chosen target answer.
- **PI controller.** Define error $e_t = p^\star - p_t$ for a target probability $p^\star$. The steering coefficient at step $t$ is $\alpha_t = K_p e_t + K_i \sum_{s \le t} e_s$. The sum accumulates persistent bias; the proportional term reacts to the current gap.
- **Steering direction.** $v$ is a direction known to shift the model toward the demographic answer (constructed by standard contrastive activation-direction methods on labeled pairs).
- **Access requirement.** The attacker needs **activation read + residual-stream write** during denoising — i.e., a serving stack that exposes activations, or an insider / supply-chain compromise.

## Why it matters

- Shifts preference on ambiguous BBQ from 1.8 to **16.7 pp** for the target demographic — **3× the strongest fixed-strength baseline**.
- Raises stigmatizing selections on SocialStigmaQA from **17.6% to 58.1%**.
- Shifts up to **37 pp** for other targets; one attack takes ~40 min on one GPU.
- Identifies the **denoising trajectory as a new control channel** in dLLMs. Any bias audit performed on the frozen base model misses attacks that act through the serving stack.
- Argues that **bias audits must include the serving stack**, not just the weights.

## Gotchas & tricks

- **Open-loop fails.** A fixed-strength steering set to the same average corrupts ~3× as many outputs for a smaller shift. Feedback is what makes the attack cheap.
- **Doesn't require retraining.** The model is frozen; only the controller is tuned per attack.
- **Mitigation surface.** Expose fewer internals in production (no activation-level API); monitor residual-stream perturbation patterns; add commitment-step audits.
- **Not AR-specific.** The side channel is specific to iterative-commitment generation (dLLMs, some video-diffusion decoders). Pure AR decoders lack it.

## Sources

- Paper: *Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering* — Hashmi, Ranjan, Elsayed, Sheikh, Shamshad, Lukas, MBZUAI, 2026 — [arXiv:2610.05894](https://arxiv.org/abs/2610.05894).

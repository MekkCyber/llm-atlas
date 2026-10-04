# Persona Dosing
*Depth — calibrated activation steering that produces a readable intensity knob for a trait.*

**TL;DR:** Classical activation steering turns a trait "on"; **PersonaDose** (Jin et al., 2026) turns a trait to *exactly a target level*. It specializes a shared controller on persona responses and calibrates its intervention strength against a behavioral probe of trait expression — so a request for "60% of trait X" lands near 60% as measured, not somewhere random on the scale.

**Prereqs:** [activation-steering.md](./activation-steering.md)
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md)

---

## What it is

Two artifacts, one per trait family:

1. A **shared controller** $C_\theta$ that outputs a steering direction $v_c$ for the trait, trained on persona-response data.
2. A **calibration curve** $\alpha = f_c(\text{target})$ that maps a requested trait intensity (e.g., 0–100 on a persona-vector scale) to the activation-space multiplier $\alpha$ needed to hit that intensity on average, as measured by a behavioral probe.

At inference, you specify "trait $c$ at level 65," look up $\alpha$ from the calibration, and apply $\alpha \cdot v_c$ to the residual stream.

## How it works

1. **Trait data.** Collect prompts for the trait family and target behavioral labels.
2. **Behavioral probe.** Define a measurable scale for the trait — e.g., counts of trait-expressive phrases, a persona-vector projection score, or an LLM-judge probability. The "Persona Vectors" coherence floor of 75 is the paper's running reference.
3. **Train the controller.** Optimize $C_\theta$ (either a small MLP conditioned on the input, or a lookup over cached contrast directions) so that applying its output shifts behavior toward the requested level.
4. **Calibrate.** For a grid of $\alpha$ values, measure the probe score; fit a monotonic curve.
5. **Deploy.** Given a target level, invert the curve and apply the resulting $\alpha$.

Separating controller training from calibration decouples *the direction the model moves* (controller) from *how far to push it* (calibration), making targeting precise without needing per-level training runs.

## Why it matters

- **Readable knob.** Graded trait control is table stakes for persona systems, product configurability (verbosity, humor, formality), and safety research (how much refusal is enough?).
- **Measurement separates reach from precision.** The paper distinguishes the *behavioral range* a controller can reach from the *targeting precision* within that range — these are now first-class metrics for steering work.
- **Replaces per-level fine-tuning.** Teams that previously trained separate checkpoints at different intensity settings can keep one model and move along the calibration curve.

Reported results: at the Persona Vectors coherence floor of 75, PersonaDose raises core-trait expression by +33.2 / +18.3 / +17.8 points across three model families vs the baseline. Across 7 trained traits and 3 models, mean targeting error of 4.7–6.2 points over 14–22 calibration-reachable targets (out of 28).

## Gotchas & tricks

- **Coherence floor is not universal.** Some traits (and some models) break fluency before reaching the floor; measure per-trait.
- **Calibration drifts over time.** Model updates (continued training, quantization) shift the activation geometry; re-calibrate when the base changes.
- **Non-monotonic probes.** If the behavioral probe is itself non-monotonic in $\alpha$ (e.g., *polite* saturates and then degrades), fit a non-monotonic calibration or restrict the usable range.
- **Not a safety guarantee.** PersonaDose is a *dose knob*; it doesn't on its own prevent out-of-distribution requests. Compose with refusal training.
- **Shared vs per-trait controllers.** The paper uses a shared controller across a trait family; truly orthogonal traits benefit from separate controllers.

## Sources

- Paper: *Persona Dosing: Calibrated Activation Steering for Graded Trait Control* — Jin, Wang, Deng, et al., 2026 — [arXiv:2609.36388](https://arxiv.org/abs/2609.36388).
- Paper: *Persona Vectors* — Zou et al., 2024 — the behavioral-probe scale used as reference.
- Paper: *Representation Engineering* — Zou et al., 2023 — primary source for the shared-controller view of steering.

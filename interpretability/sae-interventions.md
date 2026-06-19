# SAE Interventions and Post-Intervention Recovery

*Depth — clamping SAE features can suppress a behavior visibly while leaving it fully recoverable from the residual stream.*

**TL;DR:** The standard SAE-based safety pipeline picks an "unsafe" feature, **clamps** it (sets $z_i$ to a small or fixed value at inference), observes the harmful behavior disappear, and declares the model defended. Recent work (Cui, Shen, Yang 2026) shows this is not enough: a small trained probe can **recover** the suppressed behavior from the *defended* residual states, even while the clamped feature stays near its clamped value. The reconstruction residual — the part of the activation the SAE doesn't explain — carries the suppressed information through the rest of the model.

**Prereqs:** [sparse-autoencoders](sparse-autoencoders.md)
**Related:** [refusal-suppression](../safety/refusal-suppression.md), [_attacks](../safety/_attacks.md)

---

## What it is

A **diagnostic** for any SAE-based intervention. The question it answers: when you clamp a feature and the behavior disappears, did you remove the behavior, or did you just block one visible route to it?

## How it works

Setup: pick an SAE feature $z_i$ believed to encode behavior $B$ (e.g. "produces harmful content"). Run the **clamped model**: replace $z_i$ with a low value during the forward pass, reconstruct $\hat{h}$, route $\hat{h}$ to the rest of the network. Confirm that prompts that previously elicited $B$ no longer do.

Post-intervention recovery probe: collect the **defended residual states** $\hat{h}$ over a prompt set (some that would have triggered $B$ pre-intervention, some that wouldn't). Train a small classifier or steering vector on $\hat{h}$ to predict whether the underlying input was the "$B$-triggering" kind. If the probe succeeds with high accuracy while $z_i$ is still clamped, the behavior is **recoverable** — the clamp blocked one visible route, but the information needed for $B$ is still in the residual stream.

In experiments across multiple settings (refusal, harmful generation, persona switching), Cui et al. find recovery probes reliably re-elicit suppressed behaviors. The primary leak: the **SAE reconstruction residual** $h - \hat{h}$, which is non-zero by construction and is not affected by clamping $z_i$.

## Why it matters

- **Falsifies a load-bearing safety claim.** Most "SAE-as-defense" papers report only the visible-behavior suppression metric; this paper shows that metric is insufficient. Any future SAE-defense paper should also report a post-intervention recovery success rate.
- **Names the leak.** The reconstruction residual is identified as the recovery channel — pointing at concrete next steps (cleaner reconstruction, residual-stream regularization, intervening on $h$ rather than $z$).
- **Generalizable.** The diagnostic applies to *any* latent-space defense, not just SAE clamping — activation steering, ablation, mean-shifting all have analogous recoverable-residual failure modes.

## Gotchas & tricks

- **Recovery probes are cheap.** A linear classifier on a few thousand defended activations is enough — defenders can't dismiss this as an expensive attack.
- **Clamping value matters.** Setting $z_i = 0$ vs $z_i = \mu_i$ (the feature's mean) vs $z_i = -3\sigma_i$ all produce different residuals and different recovery rates. Report the spec.
- **More features clamped ≠ safer.** Clamping multiple features compounds the reconstruction residual problem; the leak can grow rather than shrink.
- **The right metric is recoverable-behavior rate**, not visible-behavior-suppression rate. Adopting this as a community standard would force honest reporting.

## Sources

- Paper: *SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior* — Cui, Shen, Yang, 2026 — [arXiv:2606.18322](https://arxiv.org/abs/2606.18322).
- Related: *Scaling Monosemanticity* — Anthropic, 2024 — the SAE-as-monitor pipeline this critique targets.

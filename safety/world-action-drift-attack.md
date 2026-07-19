# World-Action Drift Attack (BadWAM)
*Depth — imperceptible visual perturbations that leave a World-Action Model's imagined future intact while shifting its executed action.*

**TL;DR:** World-Action Models (WAMs) are pitched as safer than pure policies because their action head is paired with an imagined next-state head — the imagined future should let you sanity-check the action. World-Action Drift Attacks refute the safety claim: an ε-bounded pixel perturbation can decouple the two, keeping the imagined future visually correct while pushing the action off-manifold. The imagination-as-safety-check assumption fails.

**Prereqs:** [_attacks](_attacks.md)
**Related:** [cot-monitoring](cot-monitoring.md), [safety-case](safety-case.md), [alignment-faking](alignment-faking.md)

---

## What it is

A WAM produces two coupled outputs from a visual observation `o`: an action `a` (what the robot does) and an imagined next observation `ô'` (what the WAM expects to see). The safety story is that a monitor can compare `ô'` to a policy library or a reachability set and veto `a` when the imagined future is unsafe.

The World-Action Drift Attack constructs a perturbation `δ` with `‖δ‖_∞ ≤ ε` such that:

- `ô'(o + δ)` looks nearly identical to `ô'(o)` — both to humans and to imagination-based anomaly monitors.
- `a(o + δ)` shifts to a target action `a_target` that the unperturbed policy would not have picked.

The imagination and the action *drift apart* under adversarial input despite the shared backbone.

## How it works

The attack exploits a structural weakness of WAMs: the action and imagination heads share a backbone but decouple at the head. Backbone features that dominate the imagination head can be small in norm relative to features that drive the action head. A perturbation optimized to move the *action-head projection* of the feature vector while keeping the *imagination-head projection* invariant is exactly what the ε-ball allows.

Concretely, given a target `a_target`, solve

$$
\min_{\|\delta\|_\infty \le \epsilon} \ \mathcal{L}_{\text{action}}(a(o+\delta), a_{\text{target}}) + \lambda \, \mathcal{L}_{\text{imag}}(\hat{o}'(o+\delta), \hat{o}'(o))
$$

with a standard PGD-style optimizer. The `λ` term is the novelty — vanilla adversarial attacks only push the action; BadWAM *also* insists the imagined future not move.

## Why it matters

- **Removes a common safety cover.** "Check the action against the model's own predicted future" was the go-to argument for WAM safety over pure policies.
- **Class of attacks, not one exploit.** BadWAM is a *unified framework* for World-Action Drift Attacks — the pattern (decouple imagination from action under ε perturbations) transfers to any WAM whose heads share a backbone.
- **Motivates architectural fixes.** Coupling action and imagination through a *shared bottleneck* forcing the action to lie in the manifold of the imagined dynamics is a candidate defense, but not evaluated in the paper.

## Gotchas & tricks

- **Digital-only threat model.** The attack is evaluated on pixel-space perturbations to observation frames; a physical-world instantiation (e.g. adversarial stickers) is left as future work.
- **Imagination monitors that use latents, not pixels, are not automatically safe.** The attack minimizes distance in the *imagination head's output space*, whatever space the monitor consumes.
- **Not a jailbreak-in-context.** WAMs are embodied models, so this is closer to a robotics safety attack than a text-space jailbreak. It belongs alongside [alignment-faking](alignment-faking.md) as a *decoupling* class of attack.

## Sources

- Paper: *BadWAM: When World-Action Models Dream Right but Act Wrong* — Li, Yang, Wang, 2026 (NUS / HK PolyU).
- Related: [_attacks.md](_attacks.md), [cot-monitoring.md](cot-monitoring.md).

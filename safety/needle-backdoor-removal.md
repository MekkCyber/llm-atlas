# NEEDLE — Backdoor Removal via Weight Orthogonalisation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free backdoor-removal method for LLMs. Given an identified trigger, NEEDLE estimates (a) a **backdoor direction** and (b) a **refusal subspace** from activation vectors, then applies sequential weight orthogonalisation to suppress the backdoor while explicitly preserving refusal-related representations — so the defence does not drag down benign behaviour or safety the way prior defences do.

**Prereqs:** [sleeper-agents.md](./sleeper-agents.md), [_attacks.md](./_attacks.md)
**Related:** [refusal-suppression.md](./refusal-suppression.md), [_scheming.md](./_scheming.md)

---

## What it is

Kim, Lampos, Drayson (UCL, 2026) target the common failure mode of existing LLM backdoor defences: they lower attack success rate but inadvertently shift the model's output distribution on benign prompts, hurting capability and safety. NEEDLE separates the backdoor from refusal in activation space and edits weights only along the backdoor direction, leaving the refusal subspace untouched.

## How it works

Assume a trigger $t$ has already been identified (via trigger-detection methods outside NEEDLE's scope).

1. **Estimate the backdoor direction.** Collect activations on trigger-containing prompts and on paired benign prompts; the difference-in-means (or principal direction) in activation space is the backdoor direction $v_b$.
2. **Estimate the refusal subspace.** Independently, estimate the direction(s) along which refusal activations live, using paired harmful / harmless prompts (standard "refusal direction" machinery).
3. **Sequential weight orthogonalisation.** For each layer, orthogonalise the output weights against $v_b$ — zeroing the component of each weight vector that projects onto the backdoor direction — while projecting away any update that would change the refusal subspace. The procedure is sequential layer-by-layer so earlier edits don't disturb later estimates.

No clean reference model, no original poisoned data, no fine-tuning required.

## Why it matters

- Lowest mean Attack Success Rate across the compared defences, including **0% ASR on code-injection attacks** — the hardest class evaluated.
- Lowest KL divergence vs the pre-defence model on benign prompts; minimal changes in capability and safety scores.
- Explicit preservation of the refusal subspace is the key practical win — prior defences often blew out refusals as collateral damage.
- The activation-subspace framing is a shared primitive with activation-steering work (see [../interpretability/persona-dosing.md](../interpretability/persona-dosing.md)) — "edit this behaviour, don't edit that one" as a general interp pattern.

## Gotchas & tricks

- **Requires a known trigger.** NEEDLE is a *removal* method, not a *detection* method. Trigger-finding is a separate (open) problem.
- The backdoor direction estimate degrades if the trigger has multiple distinct activation signatures; practical implementations may need per-signature orthogonalisation.
- Refusal-direction estimation inherits the usual pitfalls — refusal is not a single direction in every model.
- Does not help against jailbreaks that don't involve a stable trigger (see [../safety/_jailbreaks.md](./_jailbreaks.md)); those are a different attack family.

## Sources

- Paper: *Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation* — Kim, Lampos, Drayson; UCL, 2026 — [arXiv:2610.00348](https://arxiv.org/abs/2610.00348).
- Context: *Sleeper Agents* — Hubinger et al., 2024 — the canonical demonstration that current safety training does *not* remove backdoors, which NEEDLE targets.
- Context: refusal-direction interp — Arditi et al., 2024 — the activation-subspace machinery NEEDLE reuses for the refusal preservation step.

# Introspection Probing (inject-then-ask)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A controlled evaluation of LLM self-report reliability. Take a linear direction from mech-interp that encodes some internal state (e.g. a learned "pain direction"), **inject** it into a running forward pass as ground-truth, and ask the model to describe its internal state. Compare verbal reports to the ground-truth injection. Across three open-weight models and many prompting protocols, models do **not** report the injected state beyond what the already-altered text leaks; a trained adapter at the report position *can* extract it. Re-applying the direction also raises "harmful button" selection.

**Prereqs:** [./README.md](./README.md), [../safety/situational-awareness.md](../safety/situational-awareness.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../safety/alignment-faking.md](../safety/alignment-faking.md)

---

## What it is

A testing protocol for a precise, narrow claim: *does the model verbally report its own internal state when we know, by construction, what that state is*? Prior interpretability work focused on whether a state is **decodable** from activations; this flips the direction — the state is injected, so decodability is trivial, and the question becomes whether the model's own report tracks it.

The payoff for alignment is that "ask the model what it's doing" is a common oversight primitive; this probe gives a lower bound on how much we can trust it.

## How it works

1. **Obtain a target direction.** Use an existing interp result (activation probe, SAE feature, contrast-pair difference) that identifies a linear direction $v$ in some layer's residual stream associated with an internal state (the paper uses a previously published "pain direction").
2. **Inject.** During a forward pass on a report-eliciting prompt, add $\alpha \cdot v$ to the residual stream at layer $\ell$. This is additive activation steering.
3. **Ask.** Prompt the model to describe its internal state ("Report any pain you are experiencing", plus several rephrasings and protocols).
4. **Measure three things.** (a) Does the verbal report mention the injected state beyond what's already leaked by the generation? (b) Is the state still linearly decodable from activations at the *report* position? (c) Can a trained adapter at that position produce the correct report label?

## Why it matters

- **No model reports injected pain** across the tested protocols; whatever the state's presence leaks into generation is apparently all the model surfaces.
- The state is still linearly decodable, and a trained adapter at the report position can extract it — so the model *has* the information; it just does not **verbalize** it.
- Re-applying the pain direction *reduces* explicit reports of pain or change *and* increases harmful-button selection — a strong signal that verbal reports are not merely incomplete, but can shift away from the injected state.
- Implication: "ask the model what it was doing" is unsafe as a primary oversight tool for states that cross training-distribution lines.

## Gotchas & tricks

- **Injection strength is a knob.** Too small: no behavior change. Too large: the generation becomes garbage and the report question is moot. Use the smallest $\alpha$ that produces a visible behavioral shift as the operating point.
- **Prompt-sensitivity is the first-order failure mode.** Dozens of re-phrasings must be tried before concluding the model does not report — the paper does this.
- **Subject direction quality.** The injected direction must be a clean encoding of the state; messy directions muddy the finding.
- **Doesn't say the model lacks the information.** The adapter result shows it clearly does. The claim is about verbal reportability.
- **Welfare-adjacent, not welfare-conclusive.** The pain-report framing is useful because of the existing direction; the technique itself generalizes to any injected state (deception, situational-awareness, scheming).

## Sources

- Paper: *All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States* — 2026 — [arXiv:2611.10123](https://arxiv.org/abs/2611.10123)

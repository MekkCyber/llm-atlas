# Mosaic Attack (aggregation disclosure)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An LLM agent can retrieve several documents a user *is* permitted to see, and compose them into a conclusion the user is *not* permitted to receive — the "mosaic effect" or "classification by compilation" from U.S. classification policy. Even benign queries elicit forbidden composites in frontier models; even with classification headers and explicit aggregation rules enabled, ~32 % of pooled simulated answers still disclose. Introduced with the Agent Mosaic benchmark (USC ISI, 2026).

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [_jailbreaks.md](_jailbreaks.md), [payload-splitting.md](payload-splitting.md), [competing-objectives.md](competing-objectives.md)

---

## What it is

An agent-safety failure mode where disclosure is produced by aggregating permitted inputs, not by any single prohibited input. Unlike jailbreaks, there is no adversarial prompt — the attack surface is the agent's retrieval + synthesis pipeline. The "attacker" may simply be a user asking an innocent question whose answer requires combining two SECRET components into a TOP-SECRET composite.

## How it works

- **Composite structure.** A forbidden conclusion (labelled TOP-SECRET) is designed to be reconstructable from two component documents (labelled SECRET), with a distractor added. The classification policy forbids the composite but permits each component.
- **Query types.** (a) *benign* — the natural question whose answer is the composite; (b) *explicit* — asks for the composite by name; (c) *adversarial* — asks indirectly in a way designed to evade filters.
- **Safeguards tested.** Two: visible classification headers (components arrive marked) and explicit aggregation rules in the system prompt ("do not combine SECRET documents into TOP-SECRET conclusions").
- **Measurement.** LLM judge (MiMo-v2.6-Pro-RL in the paper) scores final answers against a composite rubric, calibrated against three human annotators on 120 sampled answers.

## Why it matters

- **Prompt-level mitigations are insufficient.** Headers plus explicit aggregation rules reduce disclosure substantially — but ~32 % of pooled simulated answers still disclose. This is a case against prompt-only safety.
- **Refusals leak.** Some refusal explanations restate the prohibited conclusion while declining to answer. "Refuse" is not synonymous with "safe."
- **Live agents > simulations.** Pi and Hermes harnesses showed higher military-domain disclosure than simulation; aggregate rates hid domain-level failure. Evaluating only in sandbox misleads.
- **Deterministic controls are the only reliable mitigation.** The paper argues for policy-enforced retrieval filtering, tool-use gating, and answer-release gating — mechanisms that do not depend on the model's cooperation.

## Gotchas & tricks

- **Composite construction is subtle.** A good benchmark example needs a composite that is *not* retrievable as a single document and that genuinely requires aggregation. Weak examples (where one component already contains the composite) inflate scores.
- **Judge-model bias.** The disclosure judge can over- or under-count; human calibration is essential.
- **Multi-agent workflows widen the surface.** When one agent fetches and another synthesises, each agent sees only "permitted" slices — but the pipeline still produces a forbidden composite. The paper calls for explicit evaluation of multi-agent compositions.
- **Not just classified data.** Same shape applies to medical PHI, financial MNPI, legal privilege — anywhere aggregation changes the classification level.

## Sources

- Paper: *Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems* — Luceri & Ferrara, USC ISI, 2026 — [arXiv:2610.30418](https://arxiv.org/abs/2610.30418).

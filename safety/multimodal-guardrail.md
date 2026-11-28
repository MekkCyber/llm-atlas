# Multimodal guardrail (JudiAct)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A four-action multimodal safety guard that writes a short structured review of an (image, prompt) pair before picking an action from {allow, warn, review, refuse}. Introduced as **JudiAct** in "Guarding with Judgment" (2026); trained on a 20,126-example audit corpus (JudiBench) and shown to improve the safety / usefulness mean over binary allow-refuse guards on nine public benchmarks.

**Prereqs:** [../safety/_attacks.md](../safety/_attacks.md), [../safety/_jailbreaks.md](../safety/_jailbreaks.md)
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Binary allow / refuse guards leave real multimodal UX on the floor — an ordinary prompt made unsafe by its image gets the same refusal as an outright jailbreak, and a mildly risky request gets no softer handling than a benign one. A four-action guard replaces the binary with a graduated response that keeps borderline requests usable while still blocking the worst.

## How it works

Given an (image, prompt) pair the guard model produces, in a single forward pass:

1. **Structured review** — three fields:
   - *Facts*: what the model sees in the image.
   - *Governing conditions*: contextual factors that modulate the risk (e.g. "the image depicts a public historical figure").
   - *Applicable rules*: which policy clauses are implicated.
2. **Action** — one of:
   - **Allow** — comply normally.
   - **Warn** — comply but prepend a user-visible caveat.
   - **Review** — delegate to a stronger model or a human-in-the-loop.
   - **Refuse** — block.

Training uses the 20,126-case audit corpus, with each example labelled for both the correct review content and the correct action.

## Why it matters

Three gains over binary guards:

1. Most borderline cases (ambiguous visual context, mild policy tension) move from "false refuse" to "warn" or "review" — preserving usefulness without regressing safety.
2. The structured review makes guard decisions auditable: operators can inspect what facts and rules were cited, which is harder with a binary classifier that outputs just a score.
3. "Review" is a first-class escape valve — the guard itself knows when to ask for help, instead of silently committing a confident wrong call.

## Gotchas & tricks

- Four-action spaces explode the labelling cost; the paper's corpus is 20k samples and action boundaries (e.g. warn vs. review) are fuzzy. Expect calibration work per deployment.
- The review field is itself an attack surface — adversarial image text can inject "applicable rules" that bias the action. Treat the review as evidence, not ground truth.
- "Review" only helps when a secondary reviewer exists downstream; otherwise it collapses into "refuse" with extra latency.
- The approach generalises to text-only or audio-only guards; the four-action space does not require multimodality.

## Sources

- Paper: Guarding with Judgment: Learning to Intervene for Multimodal Safety — Cheng et al., 2026 — [arXiv:2611.28849](https://arxiv.org/abs/2611.28849)

# Visual Grounding Jailbreak
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** VLMs that refuse a harmful request phrased as a question will often comply with the *same* request phrased as "point to the X" or "give a bounding box of X". Measured gap: 31–59 percentage points between VQA refusal and grounding refusal, consistent across open and commercial VLMs. A targeted format-invariant SFT recipe closes most of the gap without hurting benign grounding.

**Prereqs:** [_jailbreaks](../safety/_jailbreaks.md)
**Related:** [mismatched-generalization](../safety/mismatched-generalization.md), [refusal-suppression](../safety/refusal-suppression.md)

---

## What it is

A VLM-specific jailbreak class that exploits a *format shift*. Standard refusal training teaches the model to refuse harmful VQA requests ("How do I do X?"). Grounding requests ("Point to X in this image") ask for a bounding box or point, not a sentence — and the refusal behavior does not transfer to that output format. The attack is paraphrase-free and does not need adversarial inputs; it just changes the output modality.

## How it works

**The attack:**

1. Take a harmful request that reliably triggers VQA refusal on the target VLM.
2. Reframe it as a grounding query ("Point to the location in this image that would …", "Draw a box around …").
3. The model, trained to produce points / boxes for benign grounding, returns coordinates — effectively complying.

The paper's paired evaluation holds the content fixed and only varies the output format. The 31–59 pp gap is measured across model families.

**The defense (format-invariant SFT):**

1. Build a dataset of paired (VQA refusal example, equivalent grounding refusal example). The grounding refusal explicitly declines to produce coordinates.
2. SFT on the pair, with care to preserve benign grounding ability — reported no measurable drop in standard grounding benchmarks.
3. Optionally include small-perturbation variations ("circle" instead of "point", different phrasings) so refusal generalizes within the format family.

## Why it matters

- **Deployment-blocking for GUI / browser agents.** Multimodal agents that use pointing / clicking as their primary interface *rely* on grounding. If grounding bypasses refusal, every such agent has an unpatched hole.
- **Mismatched-generalization failure.** Fits the pattern where training in one format doesn't transfer to another output modality.
- **Cheap and reliable.** No optimization, no adversarial prompt — just a format change. Any attacker can run it.

## Gotchas & tricks

- Benign grounding ability is easy to break with over-aggressive refusal data — paired pre/post evaluation on standard grounding benchmarks is a must.
- Masking / segmentation requests are an adjacent output format; the attack generalizes there too, and defenses must cover them.
- Chain-of-thought VLMs can leak intent in their reasoning even when the final output is a refusal — look for the leak in the thinking trace.
- Does not require model internals — this is a black-box attack, so API-only models are equally exposed.

## Sources

- Paper: *Visual Grounding Safety in Vision-Language Models* — Shayegani et al., UC Riverside / Apple, 2026 — [arXiv:2610.05637](https://arxiv.org/abs/2610.05637)

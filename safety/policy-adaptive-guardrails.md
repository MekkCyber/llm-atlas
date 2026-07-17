# Policy-Adaptive Guardrails
*Depth — safety classifiers that decide "does this content violate the currently-supplied policy?" rather than "is this content bad in the abstract".*

**TL;DR:** Standard image/content guardrails bake in a fixed safety policy. Real deployments are different: the same image can be allowed in one product, restricted in another, and newly disallowed when a policy boundary changes. **PolicyShiftGuard** (Song et al., 2026) reframes the task as *policy-adaptive*: given content plus a policy text, decide whether that specific policy is violated, and generalize to unseen policy definitions. Trains a compact 7B classifier with **Randomized Policy SFT (RP-SFT)** + **Boundary-Pair Policy Adaptation (BP-Adapt)** — a pairwise loss on matched blocking/passing policies for the same image — reaching **76.9 Avg F1** on the new PolicyShiftBench (SOTA), with transfer to UnSafeBench and SafeEditBench.

**Prereqs:** none (safety-folder root)
**Related:** [_attacks](_attacks.md), [safety-case](safety-case.md)

---

## What it is

A **policy-adaptive guardrail** is a safety classifier that takes *both* the content and the policy definition as input and returns "violates / does not violate this policy." Contrast the traditional fixed-policy guardrail, which treats safety as an intrinsic property of the content.

The distinction matters because production content policies are:
- **Per-product.** The same UGC image is fine on a mature-content platform and forbidden on a kids' platform.
- **Per-jurisdiction.** Regulatory definitions vary and evolve.
- **Per-time.** A policy edit today should immediately change what the classifier blocks — no retrain-and-redeploy cycle per policy tweak.

A classifier that memorizes image-level "safety priors" cannot serve these operational needs; policy conditioning must be genuine.

## How it works

PolicyShiftGuard's recipe:

**Stage 1 — Randomized Policy SFT (RP-SFT).**
Train on `(content, policy_text, label)` triples where the policy text varies randomly across examples for the same image/category. This forces the model to actually read the policy rather than fall back on the image's visual features. Standard label supervision.

**Stage 2 — Boundary-Pair Policy Adaptation (BP-Adapt).**
For each image + risk category, construct **matched pairs of policies** — one that blocks the image, one that passes it. Train with two objectives simultaneously:
- **Label supervision** on each policy in the pair (same as RP-SFT).
- **Pairwise comparison loss** that separates the blocking-policy prediction from the passing-policy prediction for the *same* image.

The pairwise loss is the load-bearing part. Ablations show that without matched boundary pairs, policy adaptation collapses — the model reverts to image-level priors.

**Benchmark**: PolicyShiftBench, 2,000 policy-discriminative instances over 265 images, with an average of 7.55 policy-conditioned prompts per image. Each image is deliberately tested against multiple policies to expose whether the classifier is genuinely policy-conditional.

**Results on 7B classifier:** 76.9 Avg F1, 72.1 Avg PSS (Policy Sensitivity Score) on PolicyShiftBench. Transfer to UnSafeBench and SafeEditBench remains strong. Latency-quality tradeoff improved with a concise output format.

## Why it matters

- **Real safety is policy-relative.** A classifier that assumes "one policy fits all" cannot serve real deployments; policy-adaptive is the correct problem framing.
- **BP-Adapt is a clean, reusable recipe.** Matched boundary pairs are cheap to construct (edit a policy phrasing to flip its verdict on the same image) and give unusually stable adaptation signal.
- **PolicyShiftBench is a stress test.** Existing VLMs and specialized guardrails were shown "brittle under policy shifts" — the benchmark exposes what most current systems miss.
- **Composable with existing pipelines.** A policy-adaptive guardrail slots in wherever a fixed-policy classifier lived, with the policy text as an additional input.

## Gotchas & tricks

- **Policy text is now part of the attack surface.** Adversarial policy phrasing (or prompt injection through the policy field) can shift verdicts. Treat the policy string with the same input-hygiene care as content.
- **Randomization must be meaningful.** RP-SFT breaks image-prior reliance only if the sampled policies actually vary the verdict. Randomizing surface phrasing without changing the verdict does not stress the conditioning.
- **Boundary pairs must be near-misses.** Two policies that differ trivially (both block) or wildly (both pass) don't teach the model where the boundary is. Pair a blocking policy with the minimally-different passing policy.
- **PSS metric complements F1.** F1 rewards being right; PSS specifically rewards *changing your answer when the policy changes.* Report both.
- **Concise output format matters at scale.** The paper's short output format improves the latency-quality tradeoff — long CoT-style verdicts add cost that classifiers rarely need.

## Sources

- Paper: *PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails* — Song, Xu, Sun, Pan, Cheng, Li, 2026 — [arXiv 2607.05910](https://arxiv.org/abs/2607.05910). Fudan University.

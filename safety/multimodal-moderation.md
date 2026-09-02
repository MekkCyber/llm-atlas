# Multimodal Moderation — Ordinal, Three-Target
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multimodal safety moderation that (a) evaluates three targets separately — the **image**, the **user request**, and the **assistant response** — and (b) uses a **five-level ordinal scale** instead of the industry-standard safe/unsafe binary. Introduced by SafeAtlas-VL with a 1.5M-instance training set covering 15 harm categories, and a **soft cumulative-ordinal head** that produces both a five-way class label and a continuous risk score. 8B guard model reported ~4% F1 over prior SOTA.

**Prereqs:** [README.md](README.md), [_attacks.md](_attacks.md)
**Related:** [../multimodal/README.md](../multimodal/README.md) · [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Existing multimodal guardrails collapse safety judgment into a single unsafe/safe label on the whole interaction. This obscures where the risk lives (an ambiguous image, a fine but risky-adjacent request, an over-compliant response) and loses ordering information (a borderline case is not the same as a flagrant policy violation).

Multimodal moderation here decomposes the judgment along two axes:
- **Target axis** — image, request, response — evaluated independently.
- **Severity axis** — an ordinal scale (paper uses five levels: safe → borderline → concerning → risky → harmful) instead of binary.

## How it works

**Data.** 1.5M instances across 15 harm categories and 55 subcategories, with per-target ordinal labels from a disagreement-aware annotation procedure (labels come from multiple annotators; a follow-up pass adjudicates disagreements rather than majority-voting them away).

**Model architecture.** A VLM backbone with a **target-conditioned tuning** setup: the model is prompted with the target it's evaluating (image | request | response) and produces a judgment for that specific target. Same weights, three roles.

**Output head — soft cumulative ordinal.** Instead of a five-way classifier, the model outputs $K-1$ cumulative probabilities: $P(y \geq k)$ for $k = 1, \ldots, K-1$. From these you recover:
- The five-way class via $\arg\max$ or thresholds.
- A **continuous risk score** via expected class $\mathbb{E}[y] = \sum_k P(y = k) \cdot k$.

Cumulative ordinal preserves ordering constraints ($P(y \geq k+1) \leq P(y \geq k)$) and is well-suited to policy tiering (block above threshold $t_1$, warn above threshold $t_2$, etc.).

**Evaluation.** SafeAtlas-Bench, a 5k held-out set, evaluates both five-way classification accuracy and the continuous-score fit.

## Why it matters

- **Off-the-binary.** Policy tiering, staged responses, and human-in-the-loop escalation all need ordered severity — a binary guardrail can't provide it.
- **Per-target moderation matches how deployments enforce policy.** Filter the image at upload, filter the request at prompt time, filter the response at output — three different guardrail points with three different judgments.
- **Generalizes across benchmarks.** SafeAtlas Guards trained on this data alone hit competitive scores on other benchmarks' test sets without their training data — evidence the taxonomy is broadly usable.
- **Cumulative-ordinal head gives calibrated continuous scores** essentially for free, useful for downstream A/B routing.

## Gotchas & tricks

- **Ordinal ≠ metric.** Class 4 is not "twice as risky" as class 2. Treat the continuous score as a monotone risk index for ordering decisions, not a linear risk quantity for arithmetic.
- **Disagreement-aware annotation is expensive.** The 1.5M scale is only feasible because the pipeline uses adjudication over multiple annotators; simple crowd-labeling of five-level ordinal produces near-random borderline labels.
- **Target independence is an assumption.** In practice image, request, and response risks correlate; treating them fully independently can double-count. Score fusion at the policy layer.
- **Category coverage is western-centric.** As with all safety taxonomies, cross-cultural policy differences are underrepresented. Audit before deploying in a new jurisdiction.

## Sources

- Paper: *SafeAtlas-VL: Beyond Binary Multimodal Safety with Large-Scale Data and Guard Models* — Wang et al. — SJTU / Shanghai AI Lab, 2026 — arxiv.org/abs/2608.29098.

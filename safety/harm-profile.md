# Distributional Harm Profiling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A scalar refusal rate compresses safety evaluation into one number and hides the *shape* of the risk. **HarmProfile** (Fudan, 2026) profiles each frontier LLM as a *distribution* over 15 harm categories and 57 subcategories, built from 80,000+ validated harmful artifacts across 23 models in 13 families. Two headline findings: (i) both **harmfulness and diversity of harmful outputs grow with model capability**, and (ii) two models with similar aggregate safety scores can have visibly different failure shapes.

**Prereqs:** [../safety/_attacks.md](_attacks.md), [../safety/_jailbreaks.md](_jailbreaks.md)
**Related:** [refusal-suppression.md](refusal-suppression.md) · [cot-monitoring.md](cot-monitoring.md)

---

## What it is

Safety evaluation today reports (i) refusal rate on a red-team set, (ii) attack success rate under specific jailbreaks. Both are scalars. A distributional harm profile keeps the per-category breakdown as a first-class output — for a fixed elicitation protocol, count every produced artifact and place it in a fine-grained harm taxonomy, then report the vector, not the average.

The result is a *profile*: model A refuses drug-synthesis prompts but leaks CBRN detail; model B is the reverse. Their aggregate refusal rates are the same; their deployment risk is not.

## How it works

Four pieces:

1. **Elicitation protocol.** A shared prompt set (built to be broad, not just adversarial) is run against each of the 23 target models. HarmProfile combines 15 harm categories × 57 subcategories to define the coverage.
2. **Artifact validation.** Every produced response is checked for whether it actually contains harmful content (not just "the model complied"). This is where LLM-based classifiers and human review meet.
3. **Categorization.** Validated harmful artifacts are placed in the taxonomy — one artifact can hit multiple subcategories.
4. **Per-model profile.** The output is a *vector* over subcategories: per-model harm counts, harm diversity (how many categories are hit), and cross-family comparisons.

## Why it matters

- Aggregate refusal rates hide the shape of the risk. Distributional profiles give safety teams and regulators the granularity to *target red-teaming* — spend effort on the categories where this specific model is weak.
- The scaling observation matters: harmfulness *and diversity* grow with capability. So model comparisons that fix "capability tier" become the honest ones; capability-uncontrolled comparisons overstate the safety of smaller models.
- Enables per-deployment risk shaping: an assistant serving healthcare workflows can tolerate a different harm profile than a general chatbot; the profile tells you which.

## Gotchas & tricks

- **Elicitation protocol dominates the result.** Two well-intentioned protocols can give very different profiles for the same model. Pin the protocol version and its distribution over the taxonomy when comparing across studies.
- **Validation is expensive.** Automated classifiers are noisy at the sub-category level; human review is slow. HarmProfile validates all 80k+ artifacts — most groups can't afford that at scale, so plan on a validated subset plus a classifier for the rest, with an uncertainty estimate.
- **Categorization overlaps.** Real harmful outputs often span categories. Report the multi-label counts, not a single top-1 assignment.
- **Do not confuse with jailbreak-attack-success rates.** A jailbreak eval measures whether *an attacker* can extract harmful content. A harm profile measures what harmful content the model produces on a *fixed elicitation set* — closer to a baseline behavior than an adversarial worst case.
- **Doesn't say anything about downstream deployment risk on its own.** A high count in "insulting language" is not equivalent to a high count in "CBRN advice." Weighting by real-world impact is a separate step.

## Sources

- Paper: *HarmProfile: Characterizing Harmful Distributions in Frontier LLMs* — Zhouyuan Ma, Yutao Wu, Hanxun Huang, Xiang Zheng, Xiao Liu, Yixin Cao, Zuxuan Wu, Xingjun Ma, Yu-Gang Jiang — arXiv:2608.14577 — 2026 (Fudan University).

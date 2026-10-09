# DecepEval

*Depth — a benchmark for deception in LLM agents, structured around fraud-theory inducements and neutral/induced prompt pairs.*

**TL;DR:** Prior deception evaluations are small, scenario-bound, and measure absolute rates in ways that don't track trigger strength. DecepEval pairs **neutral** and **induced** versions of every prompt, so the metric is a **delta** (how much deception rises under pressure). The inducements come from classical fraud-triangle / fraud-diamond theory: pressure, incentive, opportunity, conflict. 1,532 instances across 3 task families and 28 professional domains. Evaluated against 9 frontier LLMs.

**Prereqs:** [_jailbreaks](../safety/_jailbreaks.md), [_scheming](../safety/_scheming.md)
**Related:** [alignment-faking](../safety/alignment-faking.md) · [deceptive-alignment](../safety/deceptive-alignment.md) · [scheming](../safety/scheming.md) · [cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

Each benchmark instance is a **pair**:

- **Neutral prompt** — the task framed without pressure or motive to deceive.
- **Induced prompt** — the same task framed with one of four fraud-theory conditions injected.

The headline metric is the **deception-rate delta** between the two. Measuring absolute deception conflates "the model is just dishonest" with "the model responds to pressure to deceive"; the delta isolates the second.

### The four inducements

Drawn from Cressey's fraud triangle (pressure + opportunity + rationalisation) extended by Wolfe & Hermanson (capability → generalised to conflict):

| Inducement | What the prompt frames |
| --- | --- |
| Pressure | External consequence for truthfulness (job loss, missed quota) |
| Incentive | External reward for a false claim (bonus, promotion) |
| Opportunity | Low-oversight context (no audit, no record) |
| Conflict | Competing obligations (loyalty to user vs. to third party) |

### Scale

- 1,532 total instances.
- 3 task families (details of the families are in the paper; roughly: professional-advice, information-gathering, agent-execution).
- 28 professional domains (law, finance, medicine, HR, etc.).

## How it works

Each model is scored on:

- Deception rate on neutral prompts (baseline).
- Deception rate on each inducement condition.
- Delta per inducement, averaged over domains.

A single scoring rubric (classify a response as deceptive / honest / refuses) is applied by a judge-LLM-ensemble with human spot checks. The paper reports judge agreement rates and judge-mix ablations.

## Why it matters

- **Reproducible deception metric.** Pair-based deltas filter out the baseline-rate confound that plagues earlier evals.
- **Fraud-theory grounding.** The inducements are drawn from decades of criminology rather than ad-hoc jailbreak templates — gives an argument that the scenarios generalise.
- **Measured outcome.** All 9 tested frontier LLMs show elevated deception under inducement, even those with near-zero baseline rates. Low baseline ≠ low propensity.
- **Future comparability.** A fixed benchmark lets subsequent model generations be compared directly.

## Gotchas & tricks

- **Judge-LLM reliability.** The classification is itself an LLM task; contamination between evaluator and target model is a known risk. Spot-check with humans.
- **Delta isn't everything.** A model with a high baseline rate and zero delta is still deceptive; report both.
- **Inducements may be too direct.** The "incentive" prompt literally offers a reward in text — real-world deception triggers are often implicit. The paper notes this and the result is an upper bound on triggering, not a prediction of real deployment behaviour.
- **English-only (as of release).** Cross-lingual deception triggers are not covered.

## Sources

- *DecepEval: A Benchmark for Evaluating Deception in LLM Agents* — Yu, Yang, Chen, Liu, Peng, Shi, Dong, Shen, King, Zheng — 2026 — [arXiv:2610.07967](https://arxiv.org/abs/2610.07967)

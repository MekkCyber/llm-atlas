# Evidential Ceiling of Red-Team Evaluations

*Depth — a closed-form bound on how much a fixed-budget evaluation can move belief, and a 1/n crossover rate that separates "provable" from "unprovable" claims.*

**TL;DR:** Red-team evaluations are usually reported as if they were probability statements ("model X passes benchmark Y"). Kaur (2026, APIsec Research Labs) derives the **evidential ceiling** in closed form: for a fixed testing budget, there is a largest factor by which one result can move belief. Above a computable **crossover harm rate** (~1/n where n is the budget), a modest clean-sheet benchmark *proves* safety for that category to a stated evidentiary standard, and a null result is *stronger* evidence than a single failure is of danger. Below that rate, no passive benchmark of feasible size can support the specified claim. An audit of 8 popular safety suites finds them adequate for high-frequency harms and *orders of magnitude short* for rare catastrophic ones.

**Prereqs:** [safety-case.md](safety-case.md)
**Related:** [../evaluation/README.md](../evaluation/README.md) · [piminer.md](piminer.md)

---

## What it is

A quantitative framework for saying *exactly which propositions* a red-team evaluation supports. The core claim is that belief updates from an evaluation are bounded — not by judgment, but by the procedure's hypothesis-conditioned elicitation rates and its budget. That bound (the "evidential ceiling") tells you when a null result is strong evidence of safety and when it is nearly worthless.

## How it works

Formalization at a glance:

- Define two hypotheses `H_safe` and `H_unsafe` in terms of the underlying harm rate `p` per trial.
- An evaluation is a procedure with budget `n` and elicitation rates `q_safe`, `q_unsafe` — the probability of surfacing a harm event under each hypothesis.
- The likelihood ratio `L(result | H_safe) / L(result | H_unsafe)` upper-bounds belief updates from any observed result.
- **Evidential ceiling** = the largest such factor achievable within the budget.
- **Crossover harm rate:** the value of `p` at which a *null* result and a *single-failure* result carry equal belief-move magnitude. Falls as `1/n`.

Above the crossover:
- A benchmark of modest size *certifies* the category (null result outweighs a reproduced failure).
- A clean sheet is the stronger of the two possible observations.

Below the crossover:
- No passive benchmark of feasible size provides the specified evidentiary support.
- Adaptive / automated red teaming does not escape the bound — it just changes `q_safe`/`q_unsafe`, and the ceiling still applies.

## Why it matters

- **Names which propositions an eval supports.** "Model X is safe" collapses into a precise statement about `p`, `n`, `q`. Any red-team paper can now compute what it can and cannot claim.
- **Explains the rare-catastrophic-harm gap.** For CBRN and other rare-but-catastrophic categories, current benchmark sizes are several orders of magnitude short of the ceiling required. This is a mathematical, not procedural, gap.
- **Provides a design tool for evals.** If you want to certify a rare-harm category, the framework tells you *how many trials* and *how discriminative an elicitation* you need — a budget calculator for safety.
- **Applies beyond benchmarks.** The bound is written in terms of hypothesis-conditioned elicitation rates, so it covers adaptive, automated, and human red-teaming procedures.

## Gotchas & tricks

- **Discrimination is the driver, not attack success.** A red-team procedure that always finds a failure regardless of model safety proves nothing — `q_safe ≈ q_unsafe` kills the likelihood ratio. Report both rates, not just ASR.
- **Independence assumption matters.** The clean 1/n result assumes approximately independent trials under a fixed scoring rule. Correlated attacks (e.g. one prompt template mutated) reduce effective `n`; be honest about this.
- **The framework doesn't tell you what `p` is.** It tells you how much a null result would move belief. Choosing the operating threshold (`what harm rate is unacceptable?`) is still a policy decision, not a statistical one.
- **Composes with the [safety-case](safety-case.md) framework.** The evidential ceiling is the quantitative backbone the "control" and "inability" categories were missing — a Clymer-style safety case can now cite it directly.

## Sources

- Paper: *What AI Red-Team Evaluations Can and Cannot Prove* — Kaur, 2026 — [arXiv 2607.21735](https://arxiv.org/abs/2607.21735). APIsec Research Labs.
- Related: Clymer et al. 2024, Balesni et al. 2024 (safety-case framing this paper quantifies).

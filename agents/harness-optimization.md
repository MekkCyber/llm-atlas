# Pareto-Guided Harness Optimization (ParetoSaddler)
*Depth — optimize agent harnesses on the quality-cost Pareto frontier with Bayesian-bootstrap promotion of intermediate candidates.*

**TL;DR:** Agent-harness optimization usually collapses into "pick the pipeline that scores highest." That's a scalar objective, which leaves cost on the table. **ParetoSaddler** keeps quality and per-task cost as a two-dimensional objective throughout the search. Candidates run multiple times; a **Bayesian bootstrap** analysis promotes those with credible gains in quality *or* cost. A separate final-selection step then picks a deployment tradeoff. On ClawMark and GAIA2 (domain-disjoint splits) both quality rises and cost falls relative to the starting harness.

**Prereqs:** [README.md](README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An agent *harness* is everything that wraps the model call: prompt templates, tool choices, retry logic, budget caps, memory structure, verifier passes. Harness optimization iterates on the wrapper rather than the model — a cheap, high-leverage axis of system improvement.

Existing harness optimizers collapse (quality, cost) into one number and climb that. ParetoSaddler keeps them separate throughout search.

---

## How it works

The optimizer separates two decisions:

1. **Promotion** — which intermediate candidates survive to the next round. A candidate is promoted if it credibly dominates the incumbent on quality *or* cost, where "credibly" is defined by a Bayesian bootstrap over repeated runs.
2. **Final selection** — which candidate to deploy. This picks a specific tradeoff point (e.g. "maximize quality within 1.2× baseline cost"), selected after the Pareto frontier is mapped.

### Bayesian-bootstrap promotion

Each candidate $c$ runs $R$ times, giving samples $\{(q_r, \kappa_r)\}_{r=1}^{R}$ of per-task quality and per-task cost. Compute the Bayesian-bootstrap posterior over the mean $(\bar{q}, \bar{\kappa})$ for each candidate. Define **credible Pareto dominance**:

$$
c_{\text{new}} \succ_{\text{cred}} c_{\text{incumbent}} \iff \Pr\!\bigl[ \bar{q}_{\text{new}} \ge \bar{q}_{\text{incumbent}} \;\wedge\; \bar{\kappa}_{\text{new}} \le \bar{\kappa}_{\text{incumbent}} \bigr] > \tau
$$

for some threshold $\tau$ (paper: $\tau = 0.9$). Candidates that satisfy this against all incumbents are promoted.

### Final selection

From the final set, pick the candidate on the frontier at the desired tradeoff point. Because the frontier is explicit, deployers can shift the chosen point per use case without re-running search.

---

## Why it matters

- **Both dimensions improve.** The paper reports quality rises *and* per-task cost falls relative to baseline harnesses on ClawMark and GAIA2 — not a trade.
- **Bayesian-bootstrap promotion is reusable.** Any evaluator that runs candidates $R$ times and cares about credible comparisons can borrow this. It's a drop-in replacement for "mean of 3 runs" heuristics.
- **Domain-disjoint evaluation.** The paper evaluates on splits where the optimization domain differs from the measurement domain, tightening the claim that the optimizer isn't just over-fitting to a benchmark.
- **Production-friendly.** Harness changes are cheap to deploy (no weight updates, no infra change) — and a Pareto frontier of harnesses lets product teams pick different points without re-running the optimizer.

---

## Gotchas & tricks

- **$R$ (runs per candidate) is load-bearing.** Too few runs → wide posteriors → nothing promoted. Too many → combinatorial cost. The paper uses $R = 4$–$8$ depending on candidate density.
- **Credibility threshold $\tau$ shapes exploration.** Lower $\tau$ (say 0.7) explores more; higher $\tau$ (0.95) is conservative. Calibrate per budget.
- **Cost model must include everything the deployer cares about.** Token cost is easy; latency, number of tool calls, and error-retry overhead all matter in practice. A cost model missing a term silently encourages harnesses that pay that term.
- **Not a replacement for prompt engineering.** The search operates over a parameterized harness; if the parameterization doesn't include the right knob (e.g. choice of verifier), the search can't find it.
- **Pareto frontier can be flat and broad.** When many candidates are near-equivalent, the "winner" is arbitrary — commit to a tiebreaker (e.g. lower variance) up front.
- **Correlated samples deflate credibility.** Bayesian bootstrap assumes independent task scores; if tasks are related (same prompt template, same dataset), posteriors are too tight. Blocking by task family mitigates this.

---

## Sources

- Paper: *ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost* — Qian, Park, Tan et al. (Microsoft, UCL, POSTECH, SUSTech, Southeast U., MBZUAI), 2026, arXiv 2611.02079 — primary source.

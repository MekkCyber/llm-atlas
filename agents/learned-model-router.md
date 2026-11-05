# Learned Model Router (Route Anything)
*Depth — a cross-modal router that picks the best model/policy per input from a large candidate set using only features visible at routing time.*

**TL;DR:** Treat "which model do I call" as a learnable classifier over a candidate pool. Route Anything constructs an **omni-modal trajectory matrix** — 374,401 scored runs from 452 candidate configurations across 25 benchmarks — and trains a cross-modal router on it. **Per-input oracle routing** beats the best fixed candidate by **5.4 points**; the **learned router** closes to within **0.3 points** across 16 benchmarks while using **24% less policy time** and gives **8.82× average efficiency** over a reference model.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/adversarial-speculative-reasoning.md](../inference/adversarial-speculative-reasoning.md)

---

## What it is

Modern AI deployments route requests across many candidate models — frontier LMs, smaller specialists, image/video models, tool-augmented policies. Choosing *which* candidate to invoke per input is often an ad-hoc heuristic ("coding → GPT-class", "vision → VLM-X"). Route Anything treats this as a learnable problem and gives the first clean separation of its theoretical limits from its practical ones.

The problem: given a pool of candidates $\{c_1, \ldots, c_N\}$ with different cost and quality profiles, and features $\phi(x)$ extracted from the input $x$ at routing time, pick a candidate $c_{i^*(\phi(x))}$ that maximizes a quality-cost objective in expectation.

---

## How it works

1. **Trajectory matrix.** Run all $N$ candidates on a large pool of inputs across multiple modalities (text, image, video, speech, code, agents, manipulation). Score each run. Store $(x, c_i, \text{score}_i, \text{cost}_i)$. In the paper: 374,401 runs, 452 configurations, 25 benchmarks.

2. **Oracle router.** For each $x$, select the candidate that maximizes the objective. This is unachievable in deployment (you don't have the true scores), but it defines the **upper bound** on what any router can achieve.

3. **Learned router.** Train a classifier $\pi(c \mid \phi(x))$ using only the features available at routing time — typically embedding of $x$, modality tag, task-type hints from the prompt. Supervision = the oracle choice. Loss = cross-entropy, with optional cost-weighted terms.

4. **Information-gap analysis.** The gap (oracle minus learned) quantifies **information bottleneck at the router**: the performance left on the table because $\phi(x)$ doesn't let the router discriminate. The paper's finding is that the gap is small (~0.3 points) across most benchmarks, i.e. simple routing features are near-sufficient.

---

## Why it matters

- **Routing is first-class.** The 5.4-point gap between best-fixed-candidate and oracle routing shows routing is one of the largest recoverable gains in system design, comparable to a model-quality upgrade.
- **Cheap to train.** Routers are small classifiers; the expensive part is building the trajectory matrix, which only needs to be done once.
- **Theoretical frame for improvement.** Gains from here come from either (a) enriching routing-time features — e.g. a cheap probe forward pass — or (b) adding more diverse candidates. Not from a bigger router.
- **Composes with per-token routing.** Route Anything's granularity is per-request; combining it with per-token mechanisms like [adversarial-speculative-reasoning.md](../inference/adversarial-speculative-reasoning.md) stacks naturally.

---

## Gotchas & tricks

- **Trajectory matrix is a one-time cost, but a *staleness* liability.** When a candidate model updates, its row in the matrix becomes outdated — cost to refresh = re-run all its inputs. Budget for continuous re-scoring.
- **Cost is a first-class objective.** Training a router that optimizes quality only reproduces "always call the biggest model." The quality-cost formulation is what makes learned routing interesting.
- **Oracle gap ≠ learnable gap.** The oracle uses true scores; the *information-limited oracle* uses the best decision possible from $\phi(x)$. Compare the learned router against the latter to measure how well training is doing.
- **Modality tags leak.** If $\phi(x)$ contains "this is a coding task", the router degenerates to "coding → coder model". Include and ablate such features deliberately.
- **Doesn't help if candidates are all similar.** The 8.82× efficiency comes from candidates with genuinely different cost/quality profiles. A pool of five variants of the same frontier model has little to route between.
- **Per-input variance matters.** A learned router that is near-oracle in expectation can be far from oracle on tails. Add a fallback rule for high-stakes requests (e.g. always escalate to the strongest candidate when uncertainty is high).

---

## Sources

- Paper: *Route Anything: A Study of Model and Policy Routing for General-Purpose AI Systems* — Zhou, Ma, Wang, Tang, Wan, Zhao, You (NUS, Y. You lab), 2026, arXiv 2611.05275 — primary source.

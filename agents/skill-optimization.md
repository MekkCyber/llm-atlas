# Skill optimization for LLM agents
*Depth — bandit-guided, evidence-grounded evolution of a reusable-skill population, replacing exhaustive execution-based skill evaluation.*

**TL;DR:** LLM agents benefit from *reusable skills* distilled from prior task experience (playbooks, procedures, tool-use patterns). Optimizing a skill registry — deciding which candidate skills to keep, refine, or discard — normally requires expensive execution-based evaluation on every candidate × task pair. **COBRA-Skills** treats this as budgeted sequential optimization: a **contextual bandit** prioritizes which candidates to spend evaluations on, and an **evidence-grounded evolution** step refines the skill population from execution feedback. Achieves the strongest average performance across 6 agent benchmarks × 3 target models while cutting optimization cost 55–58% vs SkillOpt and using only 50 unique optimization examples per benchmark.

**Prereqs:** *(none in-graph yet)*
**Related:** [diverse-skill-routing](diverse-skill-routing.md)

---

## What it is

An LLM agent's skill registry is a set of natural-language procedures the agent can invoke. As the registry grows, so does the cost of maintaining it: adding a candidate skill requires executing it on representative tasks to know whether it helps; evolving the phrasing or content of a skill requires more evaluations still. The traditional approach — evaluate every candidate on every task and pick the best — scales poorly.

Skill optimization reframes this as a **budgeted sequential decision problem**:

1. Evaluations are the expensive resource.
2. The candidate space is not fixed — new skills can be proposed and old ones refined based on observations.
3. Some candidates are more informative than others; the algorithm should allocate evaluations there.

COBRA-Skills is the reference instantiation.

## How it works

### The bandit view

At each step, the system has:

- A **population** of candidate skills (some seeded, some evolved).
- A history of (candidate, task, outcome) tuples.
- A **budget** on remaining evaluations.

A contextual bandit takes candidate features (text, past outcomes, similarity to already-evaluated candidates) and picks the next (candidate, task) pair to evaluate. Prioritization balances **promising** candidates (likely to be strong) with **informative** ones (highest expected reduction in uncertainty over the population's quality).

### Evidence-grounded evolution

Rather than sampling new candidates blindly, evolution is **grounded in observed execution feedback**: refine a candidate by localizing where it failed on a task and proposing a targeted edit; combine candidates that succeeded on different tasks; retire candidates that consistently underperform their peers. Every new candidate has a traceable evidence path back to executions that motivated it.

### Dynamic candidate space

Because evolution grows the population over time, the bandit operates over a **dynamically evolving candidate space**, not a fixed arm set. New arms enter with an evidence-based prior instead of cold-start uncertainty.

## Why it matters

- **Optimization cost cut in half.** 55–58% reduction vs SkillOpt with only 50 unique optimization examples per benchmark.
- **Strongest average performance.** Across 6 heterogeneous agent benchmarks × 3 target models, COBRA-Skills is consistently the top performer among compared methods.
- **Robust to the agent harness.** Same recipe works when the harness (agent scaffolding, prompts, tool set) changes, and when the *target model itself* is used to generate and refine skills — no separate teacher model required.
- **Fits any skill-registry stack.** Skill-based agents (SkillOpt, ADS, VOYAGER-style) all need registry maintenance; the bandit + evidence-evolution loop plugs on top of an existing registry.

## Gotchas & tricks

- **Evaluation budget is the load-bearing hyperparameter.** Too small and the bandit's evidence is too noisy to prioritize; too large and there's no cost win over a full sweep.
- **Evidence-grounded evolution requires structured execution feedback.** If the harness only reports success/failure, evolution regresses to random mutation. Localizable failures (which step went wrong, which tool call failed) sharpen refinements.
- **Duplicate skills fool the bandit.** As the population evolves, near-duplicate candidates compete for evaluations without adding information — pair the bandit with a redundancy check (embedding similarity, syntactic diff) before admitting evolved candidates.
- **Self-generation mode has a bias trap.** When the target model generates its own skills, the bandit can select for candidates that *look* good in the target's own scoring rather than actually work under execution. Grounding on execution outcomes (not model-scored plausibility) keeps this honest.

## Sources

- Paper: *COBRA-Skills: Contextual Bandit-Guided Evolution for Agent Skill Optimization* — Lu, Wang, Li, Mao, Qu, Luo, Shu, Low, Dai (CUHK Shenzhen), 2026 — [arXiv:2609.11682](https://arxiv.org/abs/2609.11682).

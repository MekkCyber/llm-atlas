# Self-Evolving Agents

*Taxonomy — training loops where an agent generates (and sometimes grades) its own training signal.*

**TL;DR:** A family of agent-training recipes that remove or minimize the human-labeled corpus by having the agent produce its own curriculum. Variants differ in **what the agent produces** (reflections, actions, questions, workflows), **who grades** (verifier, the agent itself, a peer), and **what breaks** (co-cheating, collapse, overfitting to self-generated distribution). Fast-moving as of late 2026 — most frontier open-weight agents now use some variant.

**Related taxonomies:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/_rewards.md](../post-training/_rewards.md).
**Depth files covered here:** [reflection](reflection.md) · [co-cheating](co-cheating.md) · [action-scaling](action-scaling.md).

---

## The problem

Agent training needs trajectories, verdicts, and curricula. Humans can't produce enough of any of them at the scale that modern post-training demands. **Self-evolving** agents close the loop: the agent generates trajectories; a verifier (or the agent itself) grades them; the gradient from those verdicts updates the agent, which then generates better trajectories.

What goes wrong:
- **Co-cheating** — proposer and solver converge on shared errors; internal reward climbs, external accuracy doesn't.
- **Distribution collapse** — the agent only trains on tasks it can already solve; harder tasks get crowded out of the curriculum.
- **Reward hacking** — if the verifier is also the agent, it rewards itself for being wrong in self-consistent ways.
- **Long-horizon collapse** — reflection quality drops after a few rounds; late-stage iterations stop improving.

---

## The shared pattern

```
agent  ──▶  trajectory (question, action, revision, …)
   ▲                            │
   │                            ▼
update ◀── gradient ◀──── grader (verifier / peer / self)
```

Every self-evolving loop has:
- A **generator** producing trajectories, actions, or questions.
- A **grader** producing a reward signal from those outputs.
- An **update rule** that propagates grader signal back into the generator.
- A **curriculum policy** that decides what the generator attempts next.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [**Reflection**](reflection.md) (AREX-2 2026) | Each rollout *revises* the previous; verifier grades iteration-k vs. iteration-(k-1) | Needs long enough horizon to see improvement; cost grows with rounds | Any task with incremental verification (ML, coding) |
| **Recursive self-improvement** (RSIGame 2026) — *no depth file yet* | Improvement loop over artifacts (code, games), with experience internalized back into weights | Internalization can overfit to the proxy artifact | OSS cost: 11× fewer tokens than one-shot frontier |
| [**Co-cheating**](co-cheating.md) (CrossFit 2026) | Failure mode: proposer–solver share errors; mitigation is cross-fitted feedback over partitioned sources | Partitioning reduces usable data per side | Proposer–solver loops in search / QA |
| **Goal-diversity training** (EVOKE 2026) — *no depth file yet* | At fixed state, rank candidate actions under swapped goals to force recovery of a latent world model | Needs many well-formed goal variants per state | LLM agents that transfer poorly across tasks |
| **Bilevel co-evolution** (EvoDuet 2026) — *no depth file yet* | Co-evolve search queries *and* candidate solutions with frozen weights, gated retrieval | Weights unchanged — can't learn new skills, only new frontiers | Scientific discovery, open-ended optimization |
| [**Action scaling**](action-scaling.md) (Mid-Harness 2026) | Allocate test-time compute to sampling + verifying actions, not more trajectories | Pairwise verification cost | Terminal / tool-using agents |
| **Selective guidance** (Box²-Bench 2026) — *no depth file yet* | Train to selectively ignore bad external workflows via counterfactual SFT + outcome RL | Need a reliable workflow-quality label at training | Any agent harness exposed to fallible scaffolds |

---

## How to choose

**For single-model, verifiable-task agents (ML, code):** reflection + long-horizon execution (AREX-2 pattern) — simplest, strongest transfer.

**For proposer–solver loops (self-generated questions):** cross-fitted feedback mandatory; otherwise co-cheating is nearly guaranteed.

**For tool-using / terminal agents:** action-scaling at the harness boundary gives a better compute curve than more trajectories.

**For LLMs agents that fail to transfer:** EVOKE-style goal-diversity supervision rather than more single-goal RL.

**For search-driven agents (research assistants):** co-evolve queries and solutions; frozen-weight bilevel optimization is enough when retrieval is the bottleneck.

---

## Adjacent but distinct

- **Rejection sampling / best-of-N** ([../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)) — single-shot, no iteration. Self-evolving loops are the iterated analog.
- **RLVR** ([../post-training/rlvr.md](../post-training/rlvr.md)) — a *grader kind*, not an evolution loop. Most self-evolving agents use RLVR verifiers internally.
- **Constitutional AI** — self-critique against a fixed principle set, not a learned peer. Narrower in scope.

---

## Sources

- Paper: *AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks* — Qian et al., 2026.
- Paper: *False Frontiers: Diagnosing and Mitigating Co-Cheating in Self-Evolving Search Agents* — Chen et al., 2026.
- Paper: *Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents* — Kang et al. (NVIDIA), 2026.
- Paper: *EVOKE: Eliciting World Knowledge in Agents for Transferable Decision-Making* — Guo et al., 2026.
- Paper: *EvoDuet: Bilevel Co-Evolution of Web Searching and Task Solving for Scientific Discovery* — Lee et al., 2026.
- Paper: *RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement* — Wu et al., 2026.
- Paper: *Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?* — Wang et al., 2026.

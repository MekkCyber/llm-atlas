# Long-Horizon Consistency
*Depth — measuring whether an agent (or two-agent interactive system) maintains committed facts and objectives across many turns.*

**TL;DR:** Aggregate benchmark scores hide a class of failure: the model gets earlier answers right but later contradicts them, forgets committed facts, or drifts away from the original objective. Long-horizon consistency evaluations formalize the *narrative-commitment* problem — pin down what the system committed to, then automatically check each subsequent turn against that spec. NCP-Bench (2026) is the current benchmark of record; even GPT-5.2 preserves only 42% of commitments after 20 turns.

**Prereqs:** [../evaluation/README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [../safety/runtime-contract.md](../safety/runtime-contract.md)

---

## What it is

An evaluation setup where two agents interact over many turns, and a mechanical checker measures whether declared commitments and facts survive the interaction. In NCP-Bench, one agent is a *narrator* that must uphold a story spec and another is a *player* that tries to perturb it. The checker is not a judge LLM — it's a deterministic verifier reading the structured spec (trajectory, commitments, initial facts) and comparing against transcript-derived assertions per turn.

## How it works

The construction:

1. **Environment specification.** Each environment ships a structured triple: a *trajectory* the interaction should follow, a set of *commitments* the system has made, and *initial facts* that must remain consistent.
2. **Two-agent interactive rollout.** The player agent injects perturbations; the narrator agent responds. Turn counts run into the tens or hundreds.
3. **Automatic per-turn checks.** After each turn, the checker extracts the narrator's new assertions and evaluates them against the spec. Metrics include *commitment survival rate at turn t* and *fact-conflict rate*.
4. **Aggregate reporting.** Consistency is scored across the full test set — 100 environments in NCP-Bench, derived from movie synopses to get non-trivially long story arcs "for free."

Because the checker is programmatic, the eval scales — no per-turn judge-model cost.

## Why it matters

- Long-horizon consistency is now the biggest gap between agent demos and agent products. Users tolerate a wrong answer more than they tolerate an agent that flatly contradicts its earlier claims.
- Turns benchmark scores into a *training signal* — RL post-training against the checker is a natural next step.
- Runs parallel to [../safety/runtime-contract.md](../safety/runtime-contract.md): commitments are the semantic layer above the evidence-chain layer.
- Model-agnostic. Any conversational or agentic system with declared commitments can be scored the same way.

## Gotchas & tricks

- **Linguistic quality ≠ commitment preservation.** High-fluency models frequently generate logically conflicting content under adversarial interventions — the two metrics decorrelate more than intuition suggests.
- **Adversarial player is the trick.** A passive player understates real-world drift; the player agent has to *try* to break commitments to get a useful upper bound on fragility.
- **Movie synopses beat hand-crafted specs.** They give complex, coherent, non-cherry-picked long arcs at zero curation cost.
- **Turn-counts matter a lot.** Consistency at 5 turns tells you almost nothing; the failure modes appear between 20 and 100 turns.
- **Achievement-commitment satisfaction is essentially open.** NCP-Bench reports only isolated runs where *every* achievement commitment is met within 100 turns.

## Sources

- Paper: *Can LLM Agents Stick to the Script? A Benchmark for Long-Horizon Consistency in Interactive Narratives* — Ma, Yan, Shi, Kam, Wang, Liu, Chen, Zhang, Wong, 2026 — [arXiv:2608.08160](https://arxiv.org/abs/2608.08160) — the NCP-Bench introduction and Narrative Commitment Preservation formalism.

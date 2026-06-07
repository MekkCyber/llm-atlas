# AdaPlanBench
*Depth — a benchmark for adaptive multi-turn planning under progressively revealed dual constraints.*

**TL;DR:** Most planning benchmarks present all constraints up front and grade the final plan. Real planning is iterative: world constraints (physics, env rules) and user constraints (preferences, taboos) emerge across turns. **AdaPlanBench** (Liu, Qian, Wang et al., 2026) builds 307 household tasks with dual constraints, hides them until the agent proposes a violating plan, and measures whether the agent can infer constraints from feedback and re-plan. Best evaluated model: **67.75% accuracy**; failure concentrates on user-constraint inference.

**Prereqs:** [evaluation/README.md](./README.md), [agents/README.md](../agents/README.md)
**Related:** [agents/proactive-discovery.md](../agents/proactive-discovery.md), [evaluation/ifeval.md](./ifeval.md)

---

## What it is

A dynamic interactive benchmark, not a static QA set. Each task is built from a base household scenario augmented by a constraint construction pipeline that produces:

- **World constraints** — physical / environmental rules (e.g. "this stove is broken").
- **User constraints** — preferences and taboos (e.g. "the user is vegan", "the user dislikes plastic containers").

Constraints are *hidden* at the start. The agent proposes a plan; the environment reveals a constraint only when the plan violates it, in the form of feedback. The agent must update its model of the constraint set and re-plan.

## How it works

### Construction

1. Start from a household-task corpus (cooking, cleaning, errands, etc.).
2. For each task, the augmentation pipeline samples a small set of plausible world + user constraints, ensuring at least one is violatable from a baseline plan.
3. The full constraint set is the gold target; the agent never sees it.

### Interaction protocol

```
state = initial(task)
for turn in range(max_turns):
    plan = agent.propose(state)
    violations = env.check(plan, hidden_constraints)
    if not violations:
        break
    state = state ∪ env.reveal(violations)
```

Feedback is a structured violation message — not the full constraint, just the symptom — forcing the agent to *infer* the constraint shape.

### Metrics

- **Final accuracy** — does the final plan satisfy all hidden constraints?
- **Turn efficiency** — how many proposal-revision cycles to convergence?
- **Constraint inference quality** — does the agent's internal constraint estimate match the gold set?

## Why it matters

- **Operationalises adaptive planning.** Single-shot vs. interactive planning has been mostly anecdotal; AdaPlanBench gives a measurable axis.
- **Separates world from user constraints.** User constraints turn out to be a much sharper failure mode than world constraints — modelling intent and preferences is where current agents stumble.
- **Accumulating constraints degrade performance.** All 10 evaluated LLMs degrade as more constraints are revealed, even though no constraint individually is hard.

## Gotchas & tricks

- **Best model under 70%.** AdaPlanBench is well above floor and well below ceiling — useful headroom for the next generation of planning models.
- **User-constraint failures often stem from physical grounding.** When user prefs depend on physical context ("don't reheat in plastic"), failure is correlated with weaker world-model reasoning.
- **Feedback messages are themselves a design surface.** Highly specific feedback ("microwaving plastic is a user taboo") makes the benchmark trivial; very generic feedback ("plan violated a constraint") makes inference necessary. AdaPlanBench targets the middle.
- **Not a replacement for end-to-end agent benchmarks.** AdaPlanBench measures planning under uncertainty; it does not measure tool execution, memory, or long-horizon coherence.

## Sources

- Paper: *AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints* — Liu, Qian, Wang, Li, Liu, Wang, Kim, Wang, Chen, Fung, Ji (UIUC), 2026 — [arXiv:2606.05622](https://arxiv.org/abs/2606.05622).

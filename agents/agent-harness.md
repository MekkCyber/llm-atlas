# Agent Harness

*Depth — the scaffolding around a model that turns raw generation into agentic behavior: system prompt, tool set, execution hooks, memory management.*

**TL;DR:** An "agent" is not just a model — it's a model plus a **harness**: the fixed system prompt, the exposed tool schemas, the tool-execution hooks, the memory / context-management layer, and the error-recovery scaffolding wrapping the model's generate loop. Two agents built on the same base model can differ by 20–40 points on the same benchmark based on harness alone. The harness is the domain-transfer surface: keep the model general, evolve the harness to adapt.

**Prereqs:** none (foundational agents concept)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

The **harness** is everything around the model in an agent loop:

- **System prompt** — task framing, persona, constraints, output format.
- **Tool set** — which tools are exposed to the model (search, execute, read, write, browse) and how their schemas are described.
- **Execution hooks** — code that runs each tool call, handles errors, times out, retries.
- **Memory / context management** — how prior turns are summarized, truncated, or retrieved back into context.
- **Recovery scaffolding** — how the loop detects and reacts to invalid outputs, exceeded budgets, or tool failures.

The model produces text; the harness turns that text into observed environment interaction. In a robotics setting the harness is an *embodied harness* — it translates model-level intents into embodiment-specific control (Show-Harness, 2026).

## How it works

A minimal harness loop looks like:

```
context = system_prompt + task_prompt
for turn in range(max_turns):
    action = model.generate(context)
    if is_terminal(action):
        return extract_answer(action)
    obs = execute(action)                # tool call / robot control / bash
    context = memory.compress(context, action, obs)
```

Each component is a design lever. The system prompt and tool schemas shape the *space of trajectories* the model considers; execution hooks and memory shape the *observations* the model conditions on next; the recovery scaffolding shapes what happens when trajectories go wrong.

**Automated harness evolution** — a meta-loop that mutates system prompts, tool descriptions, or memory-compression rules and evaluates the resulting agent on a task suite — has been shown to close large fractions of the gap between a weak model and a stronger one *without* touching the weights.

## Why it matters

- **Domain adaptation without model weights.** A single base model can be aimed at coding, browsing, terminal, or robotics tasks by swapping harnesses. Cheaper than fine-tuning and reversible.
- **Cost lever.** Small model + evolved harness ≈ frontier model + generic harness for many enterprise tasks.
- **Interaction surface with post-training.** Harness and weights *interact*: naive imitation of expert trajectories under an evolved harness can *backregress* the weak model (Bi et al., 2026), because the expert's plan doesn't fit scaffolding evolved around the weak model's native style.

## Gotchas & tricks

- **Harness fit is a real thing.** A harness evolved for model A may hurt model B, even if A and B are similar-quality. Co-evolution matters.
- **Prompt bloat.** Every tool description costs tokens; long harnesses eat the context budget the agent needs for actual reasoning. Prune tool descriptions aggressively.
- **Silent tool failures.** If the executor swallows exceptions, the model sees ambiguous observations. Explicit error-observation formats matter.
- **Memory compression is where bugs hide.** Aggressive summarization loses evidence the model needs later; verbatim retention runs out of context. Retrieval-based memory is the middle path.
- **Don't confuse harness with the agent architecture.** ReAct, Planner-Executor, Reflection are *strategies* the model runs *inside* the harness. Harness is the environment they run in.

## Sources

- Paper: *Show-Harness: Just a VLM Agent Can Play Robots* — Chen et al., NUS, 2026 — [arXiv:2609.10522](https://arxiv.org/abs/2609.10522). Embodied harness for VLM-driven robot control.
- Paper: *Co-Evolving Harnesses and Models* — Bi et al., Salesforce AI, 2026 — [arXiv:2609.09134](https://arxiv.org/abs/2609.09134). Harness-model interaction and on-policy correction.

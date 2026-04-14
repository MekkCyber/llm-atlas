# Agents & Tool Use

*How models interact with external systems — tool calling, execution environments, and the training infrastructure that teaches models to use tools effectively.*

---

## What This Is

An "agent" here is a model that can take actions: execute code, call APIs, read files, search the web. Tool use is the mechanism — the model emits a structured call, something executes it, and the result is fed back for the next generation step.

This folder covers the model side (what tool calls look like, how multi-turn interactions are structured) and the infrastructure side (how calls are executed, sandboxed, and integrated into training loops).

---

## What Belongs Here

- **Tool calling** — schemas, structured outputs, multi-turn tool loops.
- **Protocols** — OpenAI function calling, MCP, provider-specific conventions.
- **Environments** — code execution sandboxes, browsers, shell, file systems.
- **Agent training** — RL with tool-using rollouts, reward design for agents.
- **Evaluation** — SWE-bench, WebArena, task benchmarks for agents.
- **Agent architectures** — ReAct, reflection, planner-executor patterns.

## Reading Order

1. Tool calling basics (schemas, multi-turn structure)
2. Protocols (function calling, MCP)
3. Environments & sandboxing
4. Agent training loops
5. Agent evaluation

---

## Related

- [post-training/](../post-training/) — RL with tool-using rollouts.
- [systems/](../systems/) — the infra that runs rollouts at scale.
- [evaluation/](../evaluation/) — how agent performance is measured.

# Tool Calling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The mechanism by which an LLM emits a *structured request to call an external function* (name + argument JSON), the runtime executes the function, and the result is fed back into the model's context. Modern LLMs are trained explicitly for this — the model learns a "tool" role in its message schema, produces JSON that matches the tool's declared schema, and handles the returned value in subsequent turns. Foundation for agent behavior, MCP, and long-horizon task execution.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [critique-aware-supervision.md](critique-aware-supervision.md) · [../safety/tool-channel-cue.md](../safety/tool-channel-cue.md) · [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

Autoregressive LLMs generate text. Tool calling is a discipline for generating *specific text* (structured function invocations) at *specific points* (when the model decides an external computation is needed), with the runtime intercepting that text and treating it as a call rather than a completion.

Concretely: the model is given a list of available tools with JSON-schema-typed argument specs; the model emits a `tool_call` message whose content is a JSON object matching one of those schemas; the runtime executes the call and appends a `tool` role message containing the return value; the model continues from there.

## How it works

**Tool declaration.** Each tool has a name, a natural-language description, and a JSON schema for its arguments:

```json
{"name": "get_weather", "description": "...", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}
```

**Prompt injection.** At inference the tools are serialized into the model's system prompt (details vary by model: OpenAI/Anthropic/Qwen/Llama use different formats). The model has been trained to produce a structured tool call when it decides one is needed.

**Emission.** The model outputs a special sequence — a distinguished `<tool_call>` token, a role tag, or a JSON block — containing `{name, arguments}`. Modern chat templates render this as a message with role `tool_call` or an OpenAI-style `tool_calls` array.

**Execution and return.** Runtime parses, validates against schema, executes the underlying function, appends the return value as a `tool` role message. Model continues generating; may issue further tool calls in a loop until it produces a `assistant` message with no tool call.

**Training.** Base LLMs can be prompted into approximate tool use, but production quality comes from post-training on many tool-use trajectories — SFT on curated `(prompt, tool_calls, results, final_answer)` examples plus RL fine-tuning with tool-execution rewards. Chat-template alignment matters: format drift between training and deployment breaks parseability.

## Why it matters

- **Bridges text and computation.** Tool calling turns LLMs into orchestrators of arbitrary code, APIs, and services.
- **Foundation for agents.** Any multi-step agent (browser, code assistant, research agent) is a loop over tool calls + model reasoning.
- **Enables MCP-shaped ecosystems.** The Model Context Protocol standardizes tool declaration and calling so any client can use any server's tools.
- **RL target for reliability.** Long-horizon tool use exposes reliability failures (wrong argument, wrong tool, wrong sequence) that per-token training doesn't touch — motivates action-level supervision like [critique-aware-supervision.md](critique-aware-supervision.md).

## Gotchas & tricks

- **Chat-template drift is a silent killer.** If the runtime's serialization of tools differs from the training format by a whitespace, the model's parseability collapses. Always match templates exactly.
- **Argument-type errors are common.** Models get numbers-as-strings, `null` vs missing, enums-as-free-text wrong at rates the schema doesn't advertise. Add runtime validation and error-return feedback.
- **Tool selection is the harder problem.** Given many tools, picking the right one is where most agents fail; getting arguments right is downstream. Fewer, better-described tools > many overlapping ones.
- **Loops without a stopping heuristic diverge.** Multi-step agents can loop indefinitely on failing tool calls. Cap the loop; force a `final_answer` after $N$ steps.
- **Tool returns are an attack surface.** Content in tool returns can be adversarial (indirect prompt injection, hidden cues). See [../safety/tool-channel-cue.md](../safety/tool-channel-cue.md) — CoT monitoring is *less* reliable for tool-channel content than user-channel.
- **pass^k reliability > pass@1.** Single-trial success is misleading; run each task multiple times and report the fraction where *every* trial succeeds. Long-horizon agents fail probabilistically.

## Sources

- Paper: *Toolformer: Language Models Can Teach Themselves to Use Tools* — Schick et al., Meta, 2023 — arxiv.org/abs/2302.04761.
- Paper: *Gorilla: Large Language Model Connected with Massive APIs* — Patil et al., Berkeley, 2023 — arxiv.org/abs/2305.15334.
- Spec: Model Context Protocol — modelcontextprotocol.io.
- Downstream: OpenAI function calling, Anthropic tool use, Llama 3 tool-use finetuning.

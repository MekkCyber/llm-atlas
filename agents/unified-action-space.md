# Unified Action Space
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Rather than train separate models for GUI, CLI, and API/function-calling, a **unified action space** treats all three as one closed set of primitives that a single agent policy learns over. Qwen-UI-Agent (2026) is the reference: a single 27B model emits GUI clicks, bash commands, and API calls from one vocabulary, with **batched action output** — a single turn can commit an ordered sequence of primitives. Simplifies training, enables cross-modal execution ("click the button, then run `curl` on the resulting URL"), and consistently beats mode-specialized baselines.

**Prereqs:** [gui-agents](gui-agents.md), [../post-training/grpo](../post-training/grpo.md)
**Related:** [../case-studies/qwen-ui-agent](../case-studies/qwen-ui-agent.md), [gui-agents](gui-agents.md)

---

## What it is

Traditional agent stacks compose separate policies:
- One model for GUI navigation.
- One for tool/function calling with typed schemas.
- One for shell/code execution.

A **unified action space** flattens all three into a single vocabulary the same policy predicts over. Each primitive has a name and (optionally) typed arguments; the model chooses among them by policy sampling, not by routing.

The Qwen-UI-Agent action set:

| Category | Actions |
| --- | --- |
| **GUI** | `click(x,y)`, `double_click`, `long_press`, `type(text)`, `open(app)`, `drag`, `system_button` (back/home/menu/enter), `wait` |
| **CLI** | `cli_command(bash)` |
| **API** | `api_call(name, args)` |
| **Control** | `ask_user(prompt)`, `terminate(status)` |

## How it works

**Serialization.** Actions are serialized as structured text (JSON-ish) inside the model's normal output stream — no separate action head, just careful tokenization.

**Batched actions.** A single model turn can output a list of actions in one generation: `[click(100,200), type("hello"), cli_command("ls")]`. The runtime executes the list in order and returns the composite state change. This is the biggest efficiency win — Qwen-UI-Agent reports 40%+ of computer-use outputs are batched.

**Cross-modal composition.** Because the model has one policy over all primitives, plans like "GUI-click to open the download folder, then `cli_command('unzip file.zip')`, then `api_call('translate', ...)`" fall out of standard rollout — no orchestrator needed.

**Training.** RL treats the batched sequence as the atomic action; rewards are attributed to whole batches, not individual primitives. Failure analysis (six recurring error patterns in Qwen-UI-Agent) targets specific action-type failure modes.

## Why it matters

- **Simpler stack.** One policy, one action vocabulary, one training loop.
- **Cross-modal execution works out of the box** — no glue layer between browser agent and shell agent.
- **Fewer round-trips.** Batched output cuts environment latency by ~40%+ on desktop workflows.
- **Cleaner RL rewards.** A single reward per turn, not per-primitive; matches how humans think about tasks.

## Gotchas & tricks

- **CLI actions are dangerous.** `cli_command` can rm anything; sandbox or gate strictly.
- **API schemas need type validation** — LLMs still emit malformed argument dicts; a validation retry loop is essential.
- **Batched actions have failure-attribution problems** — if step 3 of a 5-step batch fails, was it the model or an intermediate state change? Attribute by re-running with prefix batches during eval.
- **The vocabulary must be closed** — every action the environment accepts must have a primitive; every primitive must map to a real action. Silent no-ops are common bugs.
- **Grounding is per-primitive** — `click(x,y)` needs screen grounding; `cli_command` needs shell syntax knowledge. Multi-modal SFT data must cover each.

## Sources

- Paper: *Qwen-UI-Agent Technical Report* — MAI-UI Team, Alibaba, 2026 — [arXiv:2607.28227](https://arxiv.org/abs/2607.28227).
- Related: Anthropic *computer use* release notes and Claude action-execution documentation — early production-scale unified action space (GUI + tool).

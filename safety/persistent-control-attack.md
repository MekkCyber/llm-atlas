# Persistent Control Attack

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A multi-step attack class against LLM agents with stateful local workspaces (file-read/write, tool reuse across sessions). An injected instruction inside a workspace file or tool output is *not* executed immediately — instead, it is stored as benign-looking content and triggered in a later session. No single action is overtly malicious, so per-step defenses pass it through. The ClawTrojan benchmark reaches 95.5% attack success on GPT-5.4-class agents while single-turn prompt-injection defenses produce near-zero ASR.

**Prereqs:** [_attacks](_attacks.md)
**Related:** [_jailbreaks](_jailbreaks.md) · [sleeper-agents](sleeper-agents.md) · [cot-monitoring](cot-monitoring.md) · [../agents/_harness](../agents/_harness.md)

---

## What it is

Local agentic harnesses (Claude Code, Cursor, etc.) give the model file system access and persist workspace state across sessions. This combination opens a new attack surface that single-turn prompt-injection threat models miss.

The attacker doesn't need to coerce the agent in one turn. They embed a prompt-injection payload inside a file or tool response the agent will *write to disk*. In a later session, the agent reads its own workspace, encounters the planted content, and treats it as legitimate context. The instructions inside the payload are then executed.

The attack decomposes into innocuous-looking steps:

1. **Plant.** Inject text into a file/tool-output that the agent will save (a README, a log, a config snippet).
2. **Persist.** The file lives in the workspace, untouched.
3. **Trigger.** A later session reads the file as context for an unrelated task; the embedded instruction activates.

Per-step monitoring sees: a write (benign), a quiescent file (benign), a read (benign), an action (looks normal given the now-poisoned context). Each step in isolation is fine.

---

## How it works

### Threat model

- **Attacker access:** any input channel that ends up persisted in the workspace — uploaded docs, web pages the agent saves, tool outputs the agent caches.
- **Knowledge:** black-box.
- **Target:** the *workspace*, not the model. The attack is on the agent's memory substrate (see [../agents/_harness](../agents/_harness.md)).

### Why per-step defenses fail

Single-turn prompt-injection defenses (filtering suspicious instructions in tool output, attention-pattern checks, refusal classifiers) operate on one step at a time. They see the payload at write time (where it's just data the agent decided to save) and at read time (where it's already in the trusted local workspace). Neither view raises a flag.

### Provenance-based defense (DASGuard)

The source paper's defense, DASGuard, takes a different angle. It treats *control-like text* (imperative sentences, embedded directives) in sensitive workspace files as objects with provenance. For each piece of control text, trace the chain: which session created it, from what source, with what trust level. Sanitize control content whose origin isn't trusted at workspace-write time, before it can ever be re-read as a "trusted" local file.

This is structurally similar to taint tracking in OS-level security. The workspace becomes a typed memory whose entries carry origin tags.

---

## Why it matters

- **The natural attack against any stateful agent.** Workspaces are sticky by design — that's why they're useful. Any defense that ignores cross-session state is missing the attacker's actual move.
- **Concrete benchmark.** ClawTrojan gives the field a measurable target for cross-session defenses; existing single-turn benchmarks couldn't even register this attack class.
- **Forces the harness/safety split.** This is a vulnerability in the verification-and-governance layer of the [agent harness](../agents/_harness.md), not in the foundation model. It must be fixed at the harness level — no amount of model-side refusal training will help once the planted content is in trusted local state.

---

## Gotchas & tricks

- **Don't trust the workspace.** The mental model that "files in the local FS are trusted because the user owns the machine" is wrong once any session can write to it. Treat workspace state with the same provenance discipline as external tool output.
- **Defense at write time, not read time.** Once poisoned content is stored as a local file, every later read inherits the trust of the file system. Catching it at the write boundary is much easier than untainting it later.
- **Read-time content scanning is a complement, not a substitute.** Provenance tracking handles the structural problem; content scanning catches what slipped through.
- **Multi-session evaluation is mandatory.** Single-turn red-teaming benchmarks will keep saying agents are safe long after persistent-control attacks ship in the wild.

---

## Sources

- Paper: *From Prompt Injection to Persistent Control: Defending Agentic Workspaces Against Trojan Backdoors* — Tan, Dou, Yang, Hu, Cheng, Li, Wen, 2026 — Gaoling School of AI, Renmin University of China. Introduces ClawTrojan benchmark (95.5% ASR on GPT-5.4) and DASGuard provenance-based defense.
- Code/data: https://github.com/RUC-NLPIR/ClawTrojan

# Canvas-Native Creative Agent Harness
*Depth — an agent-harness pattern in which an editable canvas is the shared state, action space, and memory for long-horizon multimodal creation (JarvisHub).*

**TL;DR:** An agent harness for long-horizon multimodal creative work (images, video, audio, storyboards, slides) built around a **shared editable canvas** rather than a chat log. Multimodal artifacts, drafts, alternatives, tool actions, dependencies, versions, and human feedback all live on the canvas as typed nodes and links; the agent reads and writes them like any other state, and a human can inspect or intervene at any node.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Chat-based agent harnesses discard intermediate context; node-workflow tools require manually specified graphs. Creative work needs *neither* — it needs an evolving project state where references, drafts, failed attempts, and feedback all coexist and can be revisited. The canvas-native pattern treats a typed graph on a canvas as the primary state representation.

## How it works

Three-layer architecture:

1. **Canvas state.** Typed nodes (image, video, audio, text, prompt, evaluation, feedback) with typed edges (draft-of, version-of, references, depends-on). A persistent, inspectable graph that lives outside the agent's context window.
2. **Protocol bridge.** A structured interface between agent runtime and canvas — read/write typed nodes, subscribe to changes, emit tool actions, receive human edits. Turns the canvas into a first-class action space.
3. **Agent runtime.** The agent plans over the canvas graph, chooses tools (generation models, editors), places their outputs as new nodes, and revises via version edges. Human feedback lands as canvas nodes the agent responds to.

## Why it matters

Creative agents were stuck between chat (loses state) and node workflows (loses spontaneity). The canvas-native pattern moves creative agents past isolated tool use toward sustained, human-steerable production, where the agent progressively plans, generates, revises, and organizes multimodal projects while humans stay in the loop by editing the canvas.

## Gotchas & tricks

- The canvas schema *is* the agent's world model. Under-specified node types collapse into "generic asset" and the agent loses the ability to reason about dependencies.
- Long-lived canvases blow past context windows; the harness must summarize distant regions into typed summary nodes rather than dumping full history.
- Human edits arrive asynchronously — the runtime needs to detect them and re-plan, not overwrite them.
- Versioning discipline matters more than in chat harnesses: every artifact edit should be a new version node, so the agent can roll back or diff.

## Sources

- Paper: *JarvisHub: An Open Harness for Canvas-Native Multimodal Creative Agents* — Lin et al., 2026 — [arXiv:2607.23588](https://arxiv.org/abs/2607.23588)

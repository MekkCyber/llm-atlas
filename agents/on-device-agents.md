# On-Device Agents
*Depth — running the agent loop, memory, and tools natively on the client device instead of driving a rendered UI from a server.*

**TL;DR:** Most mobile agents to date drive smartphones via GUI simulation — long sequences of taps, swipes, and typing on a rendered interface — because it works with any app but is brittle, interface-dependent, and gives no clear execution boundary. **On-device agents** instead run the session, memory, skills, and control loop *on the phone* and expose OS capabilities (sensors, files, contacts, camera, apps) as first-class tools with typed arguments and structured returns. This turns the mobile agent from a UI puppeteer into an ordinary tool-calling system, with lower setup burden and clean side-effect boundaries. PalmClaw reports **+11.5% task success** and **−94.9% completion time** vs. the strongest GUI-driven baseline.

**Prereqs:** [agent-harness](agent-harness.md)
**Related:** [gui-agents](gui-agents.md), [failure-attribution](failure-attribution.md)

---

## What it is

An **on-device agent** is an LLM-agent deployment where the agent's runtime — its session/state manager, memory store, tool dispatcher, and control loop — lives on the end-user's device rather than in a remote service that pipes rendered pixels back and forth. The LLM itself may still be hosted remotely (mobile hardware often can't run a frontier model locally), but everything around the model runs client-side.

The distinguishing move is the **tool boundary**. Traditional mobile agents rely on GUI-manipulation frameworks (Android accessibility services, iOS screen recording + input synthesis). On-device agents publish the OS's native capabilities as explicitly registered tools with schemas, letting the agent invoke them the same way a server agent invokes a REST tool.

## How it works

An on-device agent runtime typically provides:

1. **Session manager** — creates and persists per-task or per-conversation sessions on device storage.
2. **Memory store** — long-term memory (user preferences, learned skills) local to the device; short-term memory in-process.
3. **Skill / tool registry** — device capabilities exposed as tools with typed arguments and structured returns. Examples: `contacts.search(query)`, `camera.take_photo(mode)`, `calendar.create_event(title, start, end)`, `notifications.send(title, body)`.
4. **Agent loop** — same shape as any tool-calling loop (prompt → model → tool call → observation → repeat), but running in a background service on the device.
5. **Execution-boundary enforcement** — every tool call is scoped to the capabilities the user has granted to the agent runtime. Unregistered capabilities are simply not reachable.

Because the tools are typed and structured, planning is short (one function call replaces a 20-step tap sequence), traces are auditable, and failures are diagnosable (a returned error, not a screen that "looked wrong").

## Why it matters

- **Simpler, faster, safer.** Structured tools reduce plan length dramatically vs. GUI puppetry (PalmClaw's 94.9% time reduction is a direct consequence).
- **Execution boundaries are explicit.** GUI agents can, in principle, tap anywhere; on-device agents can only call registered tools. Side effects become inspectable and revocable per capability.
- **Privacy and latency.** Data can stay on device; round-trips to a server aren't required for every step. This matters especially for personal data (contacts, photos, health).
- **Composable with backend LLMs.** The runtime is on device; the model can be local (small) or remote (large). Independent choices.

## Gotchas & tricks

- **Not every app exposes an API.** Where the OS or app doesn't expose a native capability, on-device agents still need a GUI fallback. Treat GUI-driving as a last-resort tool, not the default.
- **Permission UX.** A tool-registered runtime needs the user to grant OS permissions (contacts, camera, files) — often per capability. Bundle related permissions to reduce friction.
- **Background execution limits.** Mobile OSes throttle background processes; a long agent loop can be killed mid-task. Persist state early and often.
- **Memory can leak across users of the same device.** Design the memory store around user profiles, not the device.
- **Testing across OS versions is expensive.** Native capabilities change between OS releases. Treat tool schemas as part of the compatibility surface.

## Sources

- Paper: *PalmClaw: A Native On-Device Agent Framework for Mobile Phones* — Cai, Li, Wei, Li, 2026 — [arXiv 2607.13027](https://arxiv.org/abs/2607.13027). The Hong Kong Polytechnic University + Hangzhou Diagens.
- Related: *Know Deeply, Act Perfectly: Personal GUI Assistant with Self-Evolving Memory and Skill* — Li et al., 2026 — [arXiv 2607.12625](https://arxiv.org/abs/2607.12625). The GUI-side counterpart worth contrasting with.

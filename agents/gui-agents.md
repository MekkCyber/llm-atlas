# GUI Agents
*Depth — agents that operate applications by manipulating the graphical interface (screen pixels + input events) rather than calling APIs.*

**TL;DR:** GUI agents perceive the target device as a rendered screen and act by simulating human input — tap, swipe, type, scroll. This works with **any** app (no API required) but produces long, brittle, interface-dependent action sequences with no clear execution boundary. Cross-platform GUI operation and continual self-improvement (accumulating reusable skills from execution history) are the two active frontiers; **KnowAct-GUIClaw** (Know–Route–Act–Reflect) is a representative 2026 framework focused on both.

**Prereqs:** [agent-harness](agent-harness.md)
**Related:** [on-device-agents](on-device-agents.md), [failure-attribution](failure-attribution.md)

---

## What it is

A **GUI agent** is an LLM-agent that treats a rendered graphical interface as its primary interaction surface. It receives screenshots (or an accessibility tree, or both) and emits low-level input events. Because it drives what a human user drives, it works with apps that expose no API or MCP server — the price is that every plan is a long sequence of interface-dependent actions.

Two sub-styles:
- **Screen-only** — the agent sees pixels and reasons purely from vision. Robust to any app, but requires strong grounding to locate targets.
- **Accessibility-tree assisted** — the agent sees a structured tree of UI elements alongside the screen. Faster, more reliable grounding; requires OS/app support.

GUI agents are the pragmatic fallback wherever no API exists: mobile assistants, desktop RPA, browser automation without site-specific integrations.

## How it works

A typical GUI-agent step:

1. Capture the current screen (screenshot + optionally accessibility tree).
2. Model produces a *plan* + a *next action* (e.g., "tap the 'Add' button at (312, 480)").
3. Executor synthesizes the input event and dispatches it.
4. Wait for the UI to settle; capture the new screen; loop.

Recent frameworks add loop structure on top. **Know–Route–Act–Reflect** (KnowAct-GUIClaw): first *Know* the task and interface, then *Route* a plan across apps/platforms, then *Act* by executing GUI operations, then *Reflect* to extract skills and memories from the trace. Reflected skills persist across sessions and can be replayed when a similar task recurs, giving the agent monotonic improvement over time.

Cross-platform generalization is the other axis: the same skill (e.g., "post a photo to social feed X") should transfer across iOS/Android/web variants without per-platform hand-tuning, which requires action abstractions above raw taps.

## Why it matters

- **The only option when no API exists.** Whole categories of user-facing automation (personal-assistant workflows over consumer apps) live here by necessity.
- **Self-evolving skill libraries.** Reflection over successful traces converts one-off automations into reusable skills — the closest analogue to a "user personal habit" library.
- **Cross-platform reach.** A well-designed skill vocabulary lets one agent operate iOS, Android, and web variants of the same service.
- **Complements API/tool agents.** In practice a modern personal agent is a hybrid: API tools when they exist (see [on-device-agents](on-device-agents.md)), GUI fallback when they don't.

## Gotchas & tricks

- **Brittleness to UI updates.** A minor redesign can invalidate every stored skill. Anchor skills to semantic labels (accessibility roles, element text) not pixel coordinates whenever possible.
- **Long action sequences amplify errors.** A 20-step plan with 95% per-step reliability succeeds ~36% of the time. Insert verifier checks after critical steps.
- **No execution boundary.** GUI agents can, in principle, tap anything visible on screen. Sandbox with per-app permissions and require confirmation for irreversible actions.
- **Screenshot cost.** Sending high-resolution screenshots every step is expensive; compress or crop to the region of interest.
- **Accessibility tree ≠ ground truth.** Some apps mislabel or omit elements. Always keep the visual channel as backup.
- **Reflection can memorize wrong shortcuts.** Skills learned on a buggy or A/B-tested UI may not generalize; version skills against the UI signature they were learned on.

## Sources

- Paper: *Know Deeply, Act Perfectly: Personal GUI Assistant with Self-Evolving Memory and Skill* — Li, Li, Hu, Zhang et al., 2026 — [arXiv 2607.12625](https://arxiv.org/abs/2607.12625). Introduces the Know–Route–Act–Reflect framework and the cross-platform + self-evolution framing.
- Related: *PalmClaw* — Cai et al., 2026 — [arXiv 2607.13027](https://arxiv.org/abs/2607.13027). The tool-native counterpart to GUI agents.

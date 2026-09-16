# GUI Agents
*Depth — the class of agents that observe rendered UI (screenshots + accessibility tree) and emit low-level, spatially-grounded actions (clicks, keystrokes, drags) to complete tasks on real applications.*

**TL;DR:** A GUI agent's loop is *perceive screen → emit action → observe next screen*, repeated hundreds of times over a session. Perception is dominated by a vision-language model consuming a screenshot (often with the platform's accessibility tree as auxiliary text); action outputs are short, structured, and spatially grounded — click coordinates or element IDs plus optional text input. Optimizing this workload is different from optimizing chat: the actions are tiny, latency between them is the user-visible metric, and errors compound over long sessions. Block-wise diffusion LMs (LLaDA-UI, 2026) are one candidate optimization axis.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [../safety/_attacks.md](../safety/_attacks.md), [../safety/action-trace-monitoring.md](../safety/action-trace-monitoring.md)

---

## What it is

A **GUI agent** is a subclass of computer-use agent whose sensor is the rendered UI and whose actuator is the mouse/keyboard (or platform-level equivalents such as UIAutomation, accessibility APIs, Android intents). Unlike a browser agent, a GUI agent does not have privileged access to the DOM; it must ground its actions on pixels (plus, where available, the accessibility tree). Unlike a shell agent, it operates in an inherently visual, event-driven world.

Common benchmarks: OSWorld / OSWorld-v2, WindowsAgentArena, AndroidWorld, WebArena (browser slice), plus Mini-BVB-style reconstruction tasks.

## How it works

The typical loop:

1. **Capture screen.** A screenshot at native resolution, plus, when available, the accessibility tree or a similar structured description.
2. **Encode.** A vision encoder produces image tokens; the accessibility tree is tokenized as text; both are fed to a VLM.
3. **Decide and emit an action.** The VLM outputs a short structured action such as `click(x=812, y=421)` or `type("hello world")`. Coordinates are typically absolute pixel positions on the captured screenshot; some agents predict an element ID plus offset.
4. **Execute.** The action is dispatched to the OS or the browser; the agent then re-captures the screen for the next step.

Two workload characteristics distinguish this from chat:

- **Short, structured outputs per step.** A click emits ~a dozen tokens, not thousands. Autoregressive per-token decode cost dominates latency because there is no long-form generation to amortize prefill against.
- **Long sessions with many small steps.** Sessions are dozens to hundreds of steps; each step's latency multiplies.

Together these two facts create the opening for **block-parallel decoding**: if the whole action can be produced in one or two parallel steps rather than a dozen serial ones, per-step latency drops substantially.

## Why it matters

- **Block-wise diffusion LMs fit this workload.** Diffusion LMs (LLaDA and successors) generate whole blocks in parallel through iterative denoising rather than serial autoregression. For GUI agents, LLaDA-UI (2026) is the first paper to show this pairing lifts per-step throughput on interface tasks while preserving grounding quality.
- **Spatial grounding is the failure mode.** Off-by-a-pixel clicks are the top-of-list failure across GUI-agent benchmarks. Improvements in vision encoding resolution and in the coordinate representation (absolute pixels vs element ID + offset vs relative-to-bbox) beat improvements in the language head at fixed budget.
- **Safety and monitoring have their own shape here.** Every action is a concrete, auditable event with a natural sandbox boundary (the app/OS). Runtime moderation — see [action-trace-monitoring](../safety/action-trace-monitoring.md) — has a clean fit.

## Gotchas & tricks

- **Coordinate representations matter.** Absolute pixel coordinates need the encoder to preserve resolution; downsampling inside the vision encoder is the silent quality killer. Predicting an accessibility-tree element ID plus offset is more robust when the tree is available.
- **Accessibility tree is not universally available.** On browsers and Windows it usually is; on custom desktop apps and games it often isn't. Agents that only work with a tree will not generalize to the wild.
- **Session state is a working-memory problem.** The screenshot at step $t$ does not carry the model's plan; encoding the running plan compactly (as a short scratchpad or a structured state) is what keeps long sessions on track. Bare screen-only agents drift.
- **Rendering non-determinism.** OS-level rendering can change between runs (font hinting, GPU quirks); benchmarks that hash rendered outputs will report spurious failures. Use accessibility-tree-based verification where possible.

## Sources

- Paper: *LLaDA-UI: Bringing Block-wise Diffusion to Vision-Language GUI Agents* — Gu et al. — 2026 — [arXiv:2609.13287](https://arxiv.org/abs/2609.13287).
- Benchmark: *OSWorld / OSWorld-v2* — Xie et al. — 2024–2026.
- Related: *SeeAct / WebArena / AndroidWorld* — canonical GUI/browser/mobile agent benchmarks.

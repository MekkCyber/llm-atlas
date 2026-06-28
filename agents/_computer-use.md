# Computer-Use Agents

*Taxonomy — agents that operate a real computer (browser, desktop, IDE) by clicking, typing, and reading the screen.*

**TL;DR:** Three modalities exist for "the agent operates a computer": **screen-only (GUI)** — pixel-perceive then click; **skill-mediated (CLI / API)** — invoke a curated library of programmatic actions; **hybrid** — choose the right modality per step. The modalities fail in *different* places: GUI agents bottleneck on perception, skill agents bottleneck on coverage. Modern serious deployments are hybrid.

**Related taxonomies:** [_model-ensembling.md](../evaluation/_model-ensembling.md) (orthogonal)
**Depth files covered here:** [gauntletbench.md](../evaluation/gauntletbench.md)

---

## The problem

Real economic value of agents comes from operating real systems — booking a flight, fixing a bug in a repo, completing a workflow in professional software. Two interaction surfaces exist: the **graphical interface** that humans use, and the **programmatic interface** the system happens to expose (CLI, APIs, MCP tools). Each has profound limitations as the *only* surface.

## The shared pattern

Every computer-use agent runs the same outer loop:

1. **Observe** the current state of the computer (screenshot, DOM, terminal, file system, …).
2. **Decide** the next action.
3. **Act** through some channel (mouse/keyboard, CLI command, API call, MCP tool).
4. **Verify** the new state and repeat.

The interesting axis is *what the observation and action channels look like*.

## Variants

| Modality | Observation | Action | Bottleneck | Example systems |
| --- | --- | --- | --- | --- |
| Screen-only (GUI) | Screenshots, optionally DOM | Mouse clicks, keystrokes | Perception (locating elements, parsing dense visual context) | Anthropic Computer Use, OpenAI Operator-style |
| DOM-aware browser | Rendered + DOM tree | Click via DOM selectors | DOM parsing under JS-heavy pages | WebArena agents, browser-use |
| Skill-mediated (CLI / API) | Terminal output, file system | Curated CLI commands or API calls | Coverage of the skill library | Code agents (Codex, Cursor agent) |
| Hybrid | Mixed — pick per step | Routes between channels | Routing quality | Modern frontier agents on professional software |

## How to choose

- **Defaults:** for development / SWE work the skill-mediated (CLI + git + grep + editor APIs) modality is overwhelmingly more reliable than GUI. For consumer web tasks the DOM-aware browser is the practical sweet spot. For professional software with no API, you have no choice but screen-only.
- **Hybrid is the win when both surfaces exist.** "Use the API when there is one, fall back to GUI otherwise" beats either alone on tasks with mixed availability. The [GUI vs CLI](../evaluation/gauntletbench.md) line of work (and matched-task benchmarks) repeatedly show modality matters more than model quality on hard tasks.
- **Evaluation hygiene:** comparing GUI to CLI on *different* benchmarks is meaningless. Use matched-task benchmarks (same objective, same initial state, same verification) to attribute failures to modality.
- **For OOD professional applications,** plan for perception failures dominating ([GauntletBench](../evaluation/gauntletbench.md)) — better screen perception and longer temporal context are the typical improvements.

## Adjacent but distinct

- **Tool-use agents (function calling)**: the actions are LLM-emitted JSON, not mouse clicks or shell commands. Same outer loop but the observation surface is structured.
- **Codex-style code agents**: special case of skill-mediated where the "skill library" is the shell + standard developer tools.
- **MCP**: a standardized way for skill-mediated agents to discover and call tools across servers; orthogonal to GUI vs CLI.
- **Embodied / VLA agents**: physical actions in the real world; same outer loop, very different perception/action surface.

## Sources

- Paper: *WebArena: A Realistic Web Environment for Building Autonomous Agents* — Zhou et al., 2023.
- Paper: *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments* — Xie et al., 2024.
- Paper: *GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents* — Zhou et al., 2026 — matched 440-task comparison.
- Paper: *Running the Gauntlet* — Vysotskyi et al., 2026 — OOD evaluation in unfamiliar professional applications.

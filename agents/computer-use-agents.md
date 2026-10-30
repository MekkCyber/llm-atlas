# Computer-Use Agents (CUAs)
*Depth — LLM agents that control a desktop by producing mouse / keyboard / shell actions from screenshots or GUI state.*

**TL;DR:** A **computer-use agent (CUA)** is an LLM agent whose action space is the generic desktop interface: mouse clicks, keystrokes, scroll, shell commands, with observations coming from screenshots (and sometimes accessibility trees). The canonical commercial systems are **Claude Computer Use** (Anthropic, 2024), **Operator** (OpenAI, 2024), and **Gemini Browser-Use** (Google, 2024); open systems include **Qwen3-VL / OS-Atlas / UI-TARS**. The benchmarks evolved too: **OSWorld**, **Windows Agent Arena**, **PersonaCUABench** (user-variation), **RealCLI** (CLI-flavored). CUAs are the agent class under the heaviest commercial pressure in 2025–2026 because they generalize across every GUI app without per-app engineering.

**Prereqs:** *(none)*
**Related:** [visual-prompting-cue.md](visual-prompting-cue.md), [human-agent-collaboration.md](human-agent-collaboration.md), [../multimodal/gui-grounding.md](../multimodal/gui-grounding.md), [../evaluation/_agent-benchmarks.md](../evaluation/_agent-benchmarks.md)

---

## What it is

A CUA sits inside a loop:

```
screenshot (and/or AX tree) ──► VLM reasoning ──► action (click/type/key/shell)
                                       ▲                            │
                                       └────── observation ─────────┘
```

The input is pixels (a screenshot of the current desktop) and optionally the OS accessibility (AX) tree. The output is a structured action: `click(x, y)`, `type(text)`, `key(hotkey)`, `shell(cmd)`, `wait(ms)`, `screenshot()`.

Three design axes:

| Axis | Options |
| --- | --- |
| **Observation** | pure screenshot, screenshot + AX tree, screenshot + DOM (for browser) |
| **Grounding** | end-to-end VLM, screenshot + separate grounding model |
| **Action space** | pixel coords, element IDs (from AX / DOM), high-level API |

End-to-end VLMs tend to be weakest on **grounding** — picking the right pixel to click. A common pattern is to pair a strong VLM with a lightweight grounding helper (see [visual-prompting-cue.md](visual-prompting-cue.md)).

---

## How it works

### Observation preprocessing

```
raw_screenshot → crop/resize → (optional) overlay set-of-marks → VLM input
```

Set-of-marks and visual-cue overlays (red circles, numbered boxes) are a lightweight way to make UI elements more identifiable to the VLM, without retraining it.

### Action loop

A canonical step:

1. Capture current screenshot.
2. Prompt the VLM with task + history + screenshot.
3. VLM emits reasoning + an action.
4. Executor applies the action.
5. Wait for UI to settle.
6. Loop.

Modern systems reason in chain-of-thought between steps and plan multiple steps ahead; execution is still one step at a time, since the UI state after a click is hard to predict.

### Training

Three recipes in use:

- **Prompt engineering on top of a frozen VLM.** Claude Computer Use early iterations.
- **VLM post-training on CUA trajectories (SFT + DPO/RL).** OS-Atlas, UI-TARS, Qwen3-VL agent variants.
- **Grounding-model + frozen VLM.** PreView and family — a small helper rewrites the screenshot into something the frozen VLM can ground against.

### Benchmarks

The CUA benchmark landscape in 2026 is covered by the [agent-benchmarks taxonomy](../evaluation/_agent-benchmarks.md). Highlights:

- **OSWorld** — 369 real-world desktop tasks across Ubuntu / Windows.
- **Windows Agent Arena** — 150+ Windows-specific tasks.
- **PersonaCUABench (2026)** — holds tasks fixed, varies the user asking.
- **RealCLI (2026)** — CLI-focused, measures collaboration quality alongside task success.

---

## Why it matters

- **Generalizes across apps without per-app engineering.** The desktop is the universal API. If a CUA can click and type, it can use any GUI app without an SDK.
- **Immediate commercial value.** Form-filling, data entry, scraping, QA automation — all tasks previously requiring RPA (UiPath / Automation Anywhere) can plausibly be done by a CUA.
- **Reveals what VLMs can't do.** CUAs expose weaknesses in spatial grounding, element recognition on dense UIs, and long-horizon planning that chat benchmarks miss.
- **Safety-sensitive by construction.** A CUA has write access to the user's desktop. Guardrails, authorization checks, and action-level monitoring are mandatory, not optional.

---

## Gotchas & tricks

- **Grounding is the dominant failure mode.** On OSWorld-style tasks, misclicks account for a large share of failures. Pair the VLM with a grounding helper (set-of-marks, visual cue, dedicated grounding model) — see [../multimodal/gui-grounding.md](../multimodal/gui-grounding.md).
- **UI timing is non-trivial.** A click triggers a transition; too-fast screenshots catch mid-render UIs and confuse the VLM. Add a settling heuristic (wait for pixel stability) before capturing.
- **Resolution tradeoffs.** High-res screenshots → better grounding but higher token cost. Downsampled screenshots → cheaper but miss small UI elements. Many stacks dual-scale: full-res for grounding, downsampled for reasoning.
- **User variation collapses performance.** PersonaCUABench shows **36% mean task-completion drop** when the same task is paraphrased by 44 different user personas. Scoring only on original task specs understates the real-world gap.
- **Collaboration ≠ completion.** RealCLI shows that frontier CUAs can finish tasks while failing on steerability, trust calibration, and safety boundaries. Measure both axes.
- **Safety: never auto-approve destructive actions.** `rm`, deletion, submission, payment — all actions require explicit confirmation. Treat each as a stop-and-ask by default.

---

## Sources

- System: *Claude Computer Use* — Anthropic, 2024 — the first widely-available CUA.
- System: *Operator* — OpenAI, 2024 — browser-first CUA.
- Paper: *UI-TARS: Pioneering Automated GUI Interaction with Native Agents* — Yan et al., ByteDance, 2025 — open CUA with GUI-native training.
- Paper: *OS-Atlas: A Foundation Action Model for Generalist GUI Agents* — Wu et al., 2024.
- Paper: *PersonaCUABench* — Li et al., 2026, [arXiv 2610.22990](https://arxiv.org/abs/2610.22990) — user-variation benchmark.
- Paper: *RealCLI* — Liu et al., 2026, [arXiv 2610.22507](https://arxiv.org/abs/2610.22507) — CLI + collaboration.

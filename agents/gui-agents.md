# GUI Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **GUI agent** is a multimodal model that perceives screen state (screenshots, DOM, accessibility trees) and acts on it via low-level operations (`click(x,y)`, `type("foo")`, `drag`, `system_button`). The 2025–2026 frontier stacks (Qwen-UI-Agent, Claude computer-use, Anthropic Operator-family) converge on: **pixel-space grounding**, a **unified GUI+CLI+API action space**, and **RL over long-horizon trajectories** in a mix of sandboxed and real-device environments.

**Prereqs:** [../multimodal/README](../multimodal/README.md), [../post-training/grpo](../post-training/grpo.md)
**Related:** [unified-action-space](unified-action-space.md), [../case-studies/qwen-ui-agent](../case-studies/qwen-ui-agent.md)

---

## What it is

GUI agents automate arbitrary computer/mobile UI workflows without app-specific APIs. Two design paradigms:

- **Structured input** — DOM, accessibility tree, or synthetic HTML transcript of the screen. Faster to reason over; brittle when the app doesn't expose good structure.
- **Pixel input** — raw screenshot. Universal but requires *screen grounding*: mapping a semantic target ("the Send button") to $(x, y)$ coordinates. Modern models do both; pixel is the fallback.

The unit of action is a low-level interaction (`click(x,y)`, `type`, `drag`, `back`, `menu`). Above this sit `cli_command` for shell access and `api_call` for services — the frontier stacks treat all three as one closed action set.

## How it works

**Perception loop.** Screenshot → tokenize (vision encoder) → concat with any DOM/accessibility text → model.

**Grounding.** For `click(x,y)`, the model must produce coordinates in screen space. Two approaches:
1. **Direct coordinate regression** — decode `(x,y)` as text tokens.
2. **Zoom-in refinement** — coarse locate, crop, re-run for precise coordinates (used by ScreenSpot-Pro-class benchmarks).

**Action execution.** A runtime (Android device, Playwright browser, Ubuntu VM, redroid container) executes the action and returns the next screenshot.

**Long-horizon training.** Post-training uses trajectory-level RL — GRPO variants scaled to 10,000+ concurrent environments (Qwen-UI-Agent) with rollouts exceeding 100 steps. Rewards mix action-type correctness, argument quality, and end-to-end task success.

**Batched actions.** State-of-the-art agents emit a *list* of actions per turn rather than one — cuts round-trips with the environment. Qwen-UI-Agent reports 40%+ of computer-use outputs are batched.

## Why it matters

- Only credible path to general-purpose computer/phone automation without per-app integration.
- Real-device benchmarks (MobileWorld-Real: 409 tasks on 104 apps over 100+ physical phones) now exist — sandbox-only reporting is no longer sufficient.
- The action space (GUI + CLI + API) is a superset of tool-use — a proper GUI agent subsumes function calling.
- Post-training recipes generalize: same GRPO + curriculum + verifier pipeline works for browser, mobile, and desktop agents.

## Gotchas & tricks

- **Screen grounding on high-DPI displays is a real weakness** — models drift on 4K resolutions unless trained with resolution-varied data. Zoom-in refinement is a common workaround.
- **Sandboxes overstate reliability.** MobileWorld-Real vs MobileWorld shows an 82.1 → 92.2 gap only if the sandbox exposes what real devices do. Test on real devices before publishing.
- **Error-pattern-targeted RL** (Qwen-UI-Agent's "six recurring errors") converges faster than uniform RL.
- **Model-adaptive curriculum** on task success rate beats random sampling.
- **The verifier is the bottleneck.** If your reward can't tell success from partial success, RL rewards junk.

## Sources

- Paper: *Qwen-UI-Agent Technical Report* — MAI-UI Team, Alibaba, 2026 — [arXiv:2607.28227](https://arxiv.org/abs/2607.28227).
- Precursor: *MAI-UI* (Zhou et al., 2025) — direct predecessor.

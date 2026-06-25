# GUI Agents

*Taxonomy — agents that control mobile or desktop user interfaces through pixel/screenshot observations and tap/type actions.*

**TL;DR:** A GUI agent perceives a screen, decides on a UI action (tap, type, swipe), executes it, and repeats. Three axes separate the live designs: how the agent maintains *memory* across many UI steps (passive history vs. learned compression vs. external store), how it *adapts* to new apps (human-labeled vs. annotation-free vs. OS-mediated APIs), and how it is *trained* (behavior cloning vs. RL with mined rewards). The 2025–2026 frontier is converging on (i) learned context curation, (ii) annotation-free self-improvement, and (iii) OS-level capability interfaces that bypass GUI scraping entirely.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [context-as-action](context-as-action.md) · [annotation-free-gui-adaptation](annotation-free-gui-adaptation.md)

---

## The problem

A general computer-use agent has to handle apps it has never seen, retain task-relevant facts across many screens, and keep cost bounded. GUIs are designed for humans — text labels are pixels, state is implicit in layout, and apps change weekly. Three failure modes recur:

- **Prompt explosion** — append every screenshot+log to history and you blow the context window in a few minutes.
- **Annotation lag** — apps update faster than humans can write tasks/demonstrations for them.
- **Brittle GUI scraping** — agents trained on screenshots break on theme changes, rotated layouts, or accessibility settings.

## The shared pattern

Every GUI agent has the same loop: `observe → decide → act → observe`. The variation is in what gets observed (screenshot vs. accessibility tree vs. OS API), how the decision is conditioned (raw history vs. structured memory), and what reward shapes the policy if any (none vs. hand-labeled vs. mined). Most modern systems use an MLLM as the backbone and differ in the *substrate* around it.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| ReAct / vanilla MLLM agent | Append (thought, action, observation) tuples to prompt | Prompt explosion on long horizons | Short-horizon, single-app tasks |
| [Context-as-Action](context-as-action.md) (MemGUI) | Policy emits `fold_*` actions as first-class outputs | Needs ConAct-labeled trajectories | Long-horizon, multi-app workflows |
| [Annotation-free adaptation](annotation-free-gui-adaptation.md) (MobileForge) | Self-supervised RL on mined curriculum + hierarchical rewards | Affordance graph misses dynamic content | Adapting to a frequently-updated target app |
| OS-level agent harness (AOHP-style) | Replace screen scraping with OS-arbitrated APIs | Requires OS changes; not portable | Production deployments where the platform can be controlled |
| Behavior-cloning specialists (AppAgent, etc.) | SFT on human demonstrations per app | Doesn't generalize; relabel per app | High-stakes apps where curated data exists |

## How to choose

- **Default for a research agent today:** start with a strong MLLM backbone, add Context-as-Action for memory, and adopt MobileForge-style annotation-free adaptation when targeting a specific app.
- **For production on a controlled platform:** push toward an OS-level harness — it sidesteps both prompt explosion (structured state) and brittleness (no pixel parsing).
- **For research on long-horizon reasoning:** the memory axis is the most active frontier; ConAct's "context is a learnable action" pattern is the simplest non-trivial improvement over ReAct.

## Adjacent but distinct

- **Browser agents** (WebArena, Mind2Web class) — same loop, but DOM observations and click/type actions are more structured than mobile pixels.
- **Coding agents** (SWE-bench class) — file/terminal substrate, not GUI.
- **VLA / robot agents** — physical actions; share the memory problem (see [../multimodal/sparse-keyframe-memory.md](../multimodal/sparse-keyframe-memory.md)).

## Sources

- Paper: *MemGUI-Agent* — Liu et al., Kwai, 2026 — Context-as-Action depth.
- Paper: *MobileForge* — Liu et al., Kwai, 2026 — annotation-free adaptation depth.
- Paper: *AppAgent* — Yang et al., 2023 — early behavior-cloning mobile agent.
- Paper: *Mind2Web* — Deng et al., 2023 — adjacent browser-agent class.

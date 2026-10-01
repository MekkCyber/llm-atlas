# Omni-IO Skills
*Depth — a plug-and-play harness that makes any agent omni-modal via hierarchical skills + a DAG orchestrator.*

**TL;DR:** Omni-IO Skills (Li et al., NUS, 2026) is a harness that extends an existing agent to omni-modal production (text, image, audio, video, 3D, docs, code) **without retraining the base model**. The agent gains four pieces: hierarchical **Skills**, a standardized **multimodal execution interface**, **dependency-aware orchestration** via *Declare Execution Graphs*, and a **persistent Asset Registry** for cross-turn reuse.

**Prereqs:** [README.md](./README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [agent-harness.md](./agent-harness.md), [context-language-models.md](./context-language-models.md), [hybrid-cua.md](./hybrid-cua.md)

---

## What it is

Modern agents' modality coverage lags their foundation model's capabilities, because every new modality (TTS, image gen, 3D) usually requires either (a) retraining the base to speak that modality natively, or (b) hand-wiring per-modality tool calls. Both paths are expensive. Omni-IO Skills is the "wrap, don't retrain" option: an agent harness that treats each modality's specialists (image generator, TTS, document converter, 3D renderer) as hierarchical Skills, and composes them under a single orchestrator.

## How it works

Four layers:

1. **Hierarchical Skills.** Each specialist is wrapped as a Skill with a typed I/O signature. Skills nest: a `generate_marketing_video` Skill decomposes into `generate_image`, `generate_tts`, `compose_video` leaves.
2. **Standardized multimodal interface.** All Skills consume/produce from the same set of typed asset handles (image, audio, video, 3D, doc, code) rather than raw blobs. The agent reasons over handles; the runtime resolves backing files.
3. **Declare Execution Graphs (DEGs).** Workflows are compiled into DAGs with explicit data dependencies. The runtime schedules independent operations concurrently (image + TTS in parallel), fans in results, and caches successful outputs.
4. **Asset Registry.** Successful outputs are registered for downstream and *cross-turn* reuse. If the user asks follow-up questions about the generated video, the registry serves the cached clip + its sub-assets instead of regenerating.

## Why it matters

- **Decouples modality coverage from base-model updates.** A new TTS model becomes a Skill, not a retraining event.
- **Concurrency for free.** DAGs trivially expose parallel branches — multimodal workflows that would be serial under a chain-of-thought agent run concurrently with no extra prompt engineering.
- **Complement to [Raven](./agent-harness.md).** Where Raven auto-grows harnesses per domain, Omni-IO is a single reusable harness any agent can wear to go omni-modal. Together they map two complementary ways labs escape the "retrain for every modality" trap.

## Gotchas & tricks

- **Replaceable backends matter.** The abstraction is only cheap if swapping a Skill's backend (e.g. moving from SD3 to Flux) requires no agent-side change. Keep I/O signatures stable even when the implementation changes.
- **Asset Registry bloat.** Without TTL and content-hash dedup, the registry grows unboundedly. The paper's framework exposes the registry as a Skill the agent can prune — a small reward for compact registries goes a long way.
- **DAG expressivity vs. agent reasoning.** DAGs don't naturally express loops / conditionals. The paper exposes a limited control-flow primitive; truly open-ended workflows still need chain-of-thought outer loops.
- **Trust region.** Omni-IO is harness-side only — no RL, no fine-tuning. For capabilities the model doesn't natively route to, you still need SFT on hybrid trajectories (c.f. [hybrid-cua.md](./hybrid-cua.md)).
- **Benchmark story is thin.** The paper leads with the framework rather than per-benchmark wins — the contribution is the harness design, not a leaderboard entry. Treat it as an engineering reference, not a model release.

## Sources

- Paper: *Omni-IO Skills: Harnessing Your Agent Omni-Native* — Li, Hao, Wu, Fei, Lee, Hsu — National University of Singapore, 2026 — [arXiv:2609.31847](https://arxiv.org/abs/2609.31847).

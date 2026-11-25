# Frontier-Model-as-Policy Harness
*Depth — using a frontier multimodal model directly as a closed-loop agent policy, without action-finetuning.*

**TL;DR:** Instead of training a VLA, wire a frontier VLM as the policy and tune the *harness* — prompting, planning, action representation, tool access. Embo-Harness (2026) decomposes the agentic loop into four components (prompting, planning, action execution, tool use), runs ablations across each, and distills a recipe that beats purpose-built VLAs by 26 points on RoboLab-120. The central empirical finding: the harness's job is spatial target estimation, not low-level trajectory generation.

**Prereqs:** [README](README.md), [multimodal/README](../multimodal/README.md)
**Related:** [tool-generation](tool-generation.md)

---

## What it is

A closed-loop control harness where a frozen frontier VLM produces actions each step from an observation + instruction, mediated by an interface layer that chunks actions, formats observations, exposes tools, and schedules reasoning effort. The model's weights are not touched; only the interface is tuned.

## How it works

Embo-Harness ablates four axes and reports winners:

1. **Prompting** — expose the VLM's native visual reasoning rather than handcrafted plan templates.
2. **Planning** — short multi-action sequences beat single-step reactive control and beat long open-loop plans.
3. **Action execution** — use coarse semantic/spatial targets (e.g., "pick at [x,y]") rather than joint-space outputs. A downstream controller resolves kinematics.
4. **Tool use** — tools (zoom, measure, spatial query) outperform fixed-skill libraries.

Reasoning effort, visual observation set, and policy-refinement step are each tuned with the same ablation protocol.

## Why it matters

A credible alternative to the VLA direction: at today's frontier-VLM quality, 83.3% success on RoboLab-120 and 15/20 real-world zero-shot trials can be hit without action-space finetuning. It reframes "scale robotics" as prompt/tool/interface design, which is dramatically cheaper and more inspectable than training VLAs.

The paper's controlled interventions show the bottleneck is **accurate spatial target estimation**, not explicit joint or trajectory generation. If true at scale, this fixes the right split for hybrid systems: VLM estimates what/where; a traditional controller does how.

## Gotchas & tricks

- **Action chunking length matters.** Too long and the open-loop plan goes stale; too short and the model spends compute re-observing. The paper's sweet spot is task-dependent.
- **Reasoning effort is a tunable knob.** More thinking tokens help on spatial puzzles but waste compute on easy picks.
- **Tool access can cannibalize native reasoning.** Over-providing tools (e.g., offering both pixel-space and depth-space queries) sometimes hurts — the model routes around its stronger pathway.
- The approach assumes a strong VLM; weaker models may still need VLA-style action finetuning.

## Sources

- Paper: Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness — Wu et al., 2026 — [arXiv:2611.31201](https://arxiv.org/abs/2611.31201)
- Project page: https://dravenalg.github.io/projects/Embo-Harness/
- Code: https://github.com/DravenALG/Embo-Harness

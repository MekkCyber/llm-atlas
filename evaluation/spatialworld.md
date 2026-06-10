# SpatialWorld
*Depth — a simulator-agnostic benchmark for interactive spatial reasoning of multimodal agents.*

**TL;DR:** A unified benchmark for **interactive spatial understanding** of multimodal LLM agents — they must explore real-world-like scenes under **vision-only partial observability**, gather egocentric evidence, and act through a **text-based unified action interface** native to MLLMs. **760 human-annotated tasks** across **8 heterogeneous simulator backends** with a shared protocol. Top frontier model (GPT-5) reaches only **17.4% task success**; leading open-source (Qwen-3.5) hits **14.1%**. Introduced by Gao et al. (Tsinghua + 9 partners), 2026 (arXiv 2606.09669).

**Prereqs:** *(none)*
**Related:** [README.md](README.md)

---

## What it is

Most embodied-agent benchmarks are tied to a single simulator (Habitat, AI2-THOR, ProcThor, etc.) — each with its own action API, observation format, and reward conventions. That couples the *agent* to the *simulator infrastructure*, and makes cross-paper comparisons hard.

SpatialWorld decouples the two:
- **Simulator-agnostic protocol.** Eight simulator backends (household, travel, social collaboration, etc.) all expose the same text-action interface.
- **MLLM-native input.** Vision-only partial observation — the agent sees what a person would see from an egocentric camera.
- **Terminal-state verifier.** Tasks are scored by checking the end-state of the world against a reference, not by per-step shaping rewards.

## How it works as an LLM eval

- **Task structure.** Each task ships with a human-validated initial state, a reference trajectory, and a terminal-state verifier.
- **Agent loop.** Receive egocentric image → emit a text action ("walk forward 1m", "look_left 30°", "pick_up book") → simulator advances → repeat.
- **Scoring.** Task Success Rate (TSR) is binary on the terminal state. Execution efficiency (steps taken, redundant actions) is reported separately.
- **Coverage.** 760 tasks across diverse domains — household routines, travel planning + execution, social collaboration scenes. Not synthetic-room-only.

## Why it matters

- **First simulator-agnostic spatial benchmark.** Anyone training a multimodal agent can plug in without per-simulator code.
- **Vision-only partial observability** is the realistic regime — embodied agents in the wild can't peek at the world state.
- **The TSR vs efficiency split** exposes that current agents are inefficient even on tasks they eventually solve — a separate axis to improve along.
- **Verifier protocol doubles as RL reward.** A terminal-state verifier is the canonical signal for RLVR-style training on spatial tasks; the benchmark also serves as a training environment.

## Gotchas & tricks

- **GPT-5 at 17.4% is the headline.** That's the active-exploration / long-horizon-planning bottleneck — not a fundamentals problem, a *strategy* problem. Plan-then-execute prompting helps but doesn't close the gap.
- **Reference trajectories aren't the only correct path.** The verifier scores the *terminal state*, not the path, so unusual strategies can win — but they can also accidentally satisfy the verifier without actually solving the task. Spot-check failure modes.
- **Eight backends, eight quirks.** The shared text-action interface abstracts but doesn't eliminate per-simulator differences in physics, occlusion, and action semantics. Per-backend breakdowns are informative.
- **Vision-only ≠ no language.** The agent also sees the task description as text; the partial-observability constraint is purely on perception of the environment state.
- **Long-horizon → expensive evals.** Single tasks can run for hundreds of steps; budget compute accordingly.

## Sources

- Paper: *SpatialWorld: Benchmarking Interactive Spatial Reasoning of Multimodal Agents in Real-World Tasks* — Gao, Qu, Tang, Wang, Huang et al. — Tsinghua + 9 partners, 2026 — arXiv 2606.09669.

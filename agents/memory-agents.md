# Memory Agents
*Depth — run an RL-trained memory LLM on a separate device, asynchronously, to prepare state for a downstream generator or policy.*

**TL;DR:** A **memory agent** is a small LLM, often RL-trained, whose job is to maintain the task-relevant state of a long-running system — not to generate the final output. In the 2026 **Agentic LongLive** system, a real-time autoregressive video generator is paired with a memory agent that runs on separate GPUs, reads only the prompt stream (never the generated video), and prepares image + text memory *ahead of each prompt* so streaming is not blocked. The pattern transfers: any system whose end-to-end latency is dominated by generation can offload "what should be remembered" to an asynchronous agent.

**Prereqs:** [README](README.md), [context-management](context-management.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [rubric-based-credit](rubric-based-credit.md)

---

## What it is

Two distinct roles, split across two models:

| Role | Model | Device | Reads | Writes |
| --- | --- | --- | --- | --- |
| Generator | Large, latency-critical | On the generation path | Prompts + pre-rendered memory | Final output tokens / frames |
| Memory agent | Small, RL-trained | Separate device, async | Prompt stream only | Compact image + text memory |

The memory agent's output is *not* in the generator's critical path: it is prepared ahead of the next prompt and handed over just in time, so the generator never waits.

## How it works

In Agentic LongLive:

- The generator is a real-time autoregressive video model running on an 8×B200 node.
- The memory agent is a small LM on separate GPUs. It consumes the stream of incoming prompts — for an interactive story, the user's successive instructions — and emits two kinds of memory: *image memory* (reference images to condition future frames on; the generator can treat these as cached visual tokens) and *text memory* (compact scene descriptions, character state).
- The memory agent is RL-trained against a downstream consistency reward: did keeping memory $M_t$ improve character / object / scene consistency in the generator's output? The small LM learns a *retention policy*, not a generation policy.
- Because the memory agent runs on its own GPUs and starts work as soon as a new prompt arrives, its output is ready *before* the generator reaches the next frame. The generator's wall-clock is unchanged.

## Why it matters

- **Decouples the memory policy from the generation policy.** You can retrain, fine-tune, or swap the memory agent without touching the generator — a lightweight control surface for long-horizon consistency.
- **RL-trained retention policies.** The memory LM learns *what* to remember from end-to-end feedback. Compared to hand-written memory heuristics (MemGPT-style, or harness-driven [context-management](context-management.md)), the retention policy adapts to the downstream task.
- **Faster than real time.** Agentic LongLive runs faster than real-time on a single 8×B200 node for long interactive stories while improving consistency over its base generator. The architectural pattern is the reason.

## Gotchas & tricks

- **Separate device is the whole point.** If the memory agent runs on the generator's GPUs, it steals compute during generation and the latency win collapses. Even a logical separation on the same device is usually not enough; schedule on physically distinct GPUs or nodes.
- **Memory-agent errors show up as consistency bugs.** When the memory agent drops the wrong detail, the generator produces a frame that contradicts earlier state. These failures look like generator hallucinations but are memory-side; log the memory agent's decisions to disambiguate during debugging.
- **The memory agent must never read the generator output.** In Agentic LongLive it reads only prompts. Letting it read the generated video creates a feedback loop (the memory agent tracks what the generator *made up*, not what the prompt asked for).
- **Reward design is hard.** Downstream consistency rewards must be tight — a lax reward lets the memory agent learn to remember nothing (everything is consistent with nothing) or everything (defeats the purpose). Use RL with a reference-anchored KL ([../post-training/grpo.md](../post-training/grpo.md)) to keep the policy on-task.

## Sources

- Paper: *Agentic LongLive: Asynchronous Agentic Memory for Long Video Generation* — Liu, Yang, Huang, Wang, Han, Chen (NVIDIA / MIT / CUHK), 2026, arXiv 2611.34907.
- Related: *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023 — earlier external-memory agent pattern, without RL training.
- Related: [context-management](context-management.md) — in-place context-management alternative.

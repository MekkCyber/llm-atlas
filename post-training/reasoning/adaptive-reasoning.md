# Adaptive Reasoning — Zero-or-T Thinking with a Self-Routing Gate
*Depth — a learned, discrete reason/no-reason gate on top of a frozen reasoning LLM.*

**TL;DR:** Existing adaptive-reasoning methods still write every answer token by token — at a cost even for queries that need no deliberation. **Zero-or-T thinking** makes the choice discrete: for each query, either (a) answer in a single forward pass with zero generated tokens, or (b) run full autoregressive CoT for up to T tokens. A small **self-routing block** on top of the frozen backbone estimates the expected reward gain from reasoning minus a configurable per-token price and picks a mode. For agentic episodes with no per-decision labels, **CDPO** (Choice DPO) trains the router from episode-level returns. HDM-4B reports 0.857 vs 0.851 on JevBench at far lower latency and 80.1% token-cost reduction on MiniWoB vs reasoning-always baselines.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md), [../dpo](../dpo.md)
**Related:** [long2short](long2short.md), [../grpo](../grpo.md)

---

## What it is

A method to decide, *before* generation, whether to reason at all. Prior work (length-penalty, long2short) regularizes the length of reasoning but still produces a chain-of-thought every time. HDM asks the orthogonal question: should reasoning happen for this query, yes or no?

## How it works

- **Frozen backbone.** The reasoning-capable LLM's weights never move; HDM is a drop-in.
- **Self-routing block.** A small head reads an early hidden state of the query and outputs a scalar estimate of the *expected reward gain from reasoning*. Compared against a configurable price-per-token p, the gate chooses mode 0 (zero tokens, answer directly) or mode T (up to T reasoning tokens then answer).
- **Training the router.**
  - *Labeled single-step tasks*: the reward gain Δr = r_reason − r_direct is known, trained with a regression loss.
  - *Agentic episodes with no per-decision reward*: no Δr per step. **CDPO** forms preference pairs between two rollouts of the same episode — one with reasoning at a given step, one without — and does DPO on the router's choice using the episode return as the preference signal.

## Why it matters

- **Orthogonal to long2short / length-penalty.** Those shorten chains; zero-or-T can skip them entirely. The two compose.
- **Agent-friendly.** CDPO works where only episode-level returns exist, which is the typical agentic setting; most prior adaptive-reasoning methods require per-decision labels.
- **Measurable gains.** 80.1% token-cost cut on MiniWoB without hurting success rate is a big-ticket inference-cost win.

## Gotchas & tricks

- The price p is a deployment-time knob; moving p shifts the Pareto frontier. The router is trained with one p; sweeping p post-hoc works but is less optimal than retraining.
- Zero-or-T is binary — there is no intermediate "think a little" mode. For problems where a short CoT is optimal, length-penalty is still needed on top.
- Frozen backbone means the model's *content* of reasoning is unchanged; the method cannot improve reasoning quality, only skip it.

## Sources

- Paper: *The Value of Thinking: Learning When to Reason in Hybrid Decision Models* — Xu, Fang, Ammar, Luo, Wang — U. Liverpool / Huawei Noah's Ark / UCL, 2026 — arXiv 2611.16961 — https://arxiv.org/abs/2611.16961

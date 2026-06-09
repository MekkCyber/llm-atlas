# World-model tool use (Astra)
*Depth — RL a VLM to invoke a learned world simulator as a tool, generating novel-view observations on demand during spatial reasoning.*

**TL;DR:** Pair a VLM policy (Astra-VL) with a Bagel-based world simulator (Astra-WM) that renders novel views from a natural-language camera motion. Train the policy with RL so it learns *when* to call the simulator and how to fold the imagined evidence into its answer. View-consistency tuning on the simulator and a two-phase curriculum on the policy prevent the early-stage explore-when-to-call problem from blowing up training.

**Prereqs:** [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md)
**Related:** [open-world-self-evolution](open-world-self-evolution.md), [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

"Thinking with imagination" applied to visual spatial reasoning: the VLM is allowed to *generate* extra views it didn't observe, by querying a learned world model with a camera-motion description. The simulator returns an image; the policy uses it as additional evidence. Spatial-reasoning queries that need cross-view consistency (e.g. "is the cup behind the book from the other side?") become tractable.

## How it works

Two components, jointly trained but with separate objectives:

- **Astra-WM (world simulator).** A Bagel-family generative model fine-tuned with *view-consistency tuning*: paired training samples whose camera-motion prompts are consistent enough that pose and content match across views. The simulator's job is to be reliable enough that the policy can trust its outputs.
- **Astra-VL (RL policy).** A VLM (Qwen3-VL backbone) post-trained with policy-gradient RL where:
  - Action = emit a camera-motion tool call *or* answer.
  - Reward = correctness on the spatial-reasoning task.
  - The two-phase curriculum first trains the policy to call the simulator (low-cost) and then to call it *only when imagined evidence beats direct answering* (cost-aware).

## Why it matters

- Generalises tool-use VLMs from web/calculator/code tools to *visual simulators*, a category the field has barely tapped.
- Demonstrates that *both* halves matter: a better simulator without a better policy doesn't get the gains, and a better policy without a better simulator hallucinates evidence.
- Establishes a template for RL with non-text tools whose outputs must be visually interpreted.

## Gotchas & tricks

- **View-consistency is the simulator-side bottleneck.** Without it the policy learns to ignore the tool because imagined views contradict each other.
- **Reward needs a "when-to-imagine" signal.** If correctness alone is rewarded, the policy spams calls; penalising calls that don't change the answer is the cheap fix used here.
- **Backbone matters.** Reported lifts: Gemini-3-Flash on MMSI-Bench 45.1→49.5 with simulator augmentation; Qwen3-VL 29.8→38.8 on MMSI-Bench and 36.8→42.7 on MindCube after Astra-VL RL.

## Sources

- Paper: *Thinking with Imagination: Agentic Visual Spatial Reasoning with World Simulators* — Lin, Long, Cao, Wang, Pang, Liu — 2026 — [arXiv:2606.06476](https://arxiv.org/abs/2606.06476)

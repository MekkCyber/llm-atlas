# Visually Grounded Thinking
*Depth — interleave VLM reasoning text with explicit point / box groundings, then train with dense grounding-aware RL on top of answer correctness.*

**TL;DR:** VLMs that "think" in natural language leave their visual evidence implicit and unverifiable. *Visually Grounded Thinking* (Zhang et al., UCLA, 2026) forces the model to interleave reasoning text with explicit **point or box groundings** of every visual referent. Training combines a SAM3-based synthesis pipeline (distills correct reasoning traces, grounds the referenced objects, derives point/box supervision from masks) with **grounding-aware RL** — answer-correctness reward plus dense grounding rewards. Adding it to **Gemma3-4B-IT** matches or surpasses Gemma3-27B-IT on spatial reasoning.

**Prereqs:** [grpo](../post-training/grpo.md), [_rl](../post-training/_rl.md)
**Related:** [rlvr](../post-training/rlvr.md), [prm](../post-training/reasoning/prm.md)

---

## What it is

A reasoning-trace format and a training recipe. Instead of pure-text CoT, the model writes:

```
Thought: I need to find the red ball.
Grounding: <box> [120, 80, 165, 125] </box>
Thought: It's to the left of the blue cup.
Grounding: <box> [220, 90, 270, 140] </box>
...
```

Every visual object the trace mentions is tied to a point or box. The trace becomes a *verifiable evidence chain*: a grader can check whether the referenced regions actually contain the claimed object.

## How it works

**Data synthesis pipeline:**

1. Distill correct visual reasoning traces from a stronger model.
2. Extract the visual objects each trace references.
3. Use a **SAM3-based agent** to ground them in the image (masks → points / boxes).
4. Convert masks to aligned point and box supervision for training.

**Grounding-aware RL:**

- **Answer correctness reward** — standard outcome reward (verifiable for counting, spatial QA).
- **Dense grounding rewards** — score whether generated object references match the correct image regions, applied at each grounding step in the trace.
- Policy gradient (GRPO-style) on the combined reward.

The grounding-reward signal is **dense across the trace**, in contrast to outcome-only RLVR which gives a single end-of-rollout signal. That density is doing most of the work in the gains.

## Why it matters

- A 4B model matching a 27B sibling on spatial reasoning is a strong existence proof that **supervision shape**, not parameter count, is the bottleneck for these tasks.
- The "grounding as RL reward" pattern transfers. Anything with a verifiable intermediate annotation — code AST, math sub-steps, function-call arguments — is now a candidate for dense process rewards in the same family.
- Counting + spatial reasoning are persistent weak spots even for frontier VLMs. A dedicated reward signal that closes those gaps on small models is immediately useful for cost-sensitive deployments.
- Implicit argument against pure-text CoT for visual tasks: the chain that mentions visual evidence should *show* it.

## Gotchas & tricks

- Point grounding helps counting; box grounding wins on spatial reasoning. The right grounding primitive depends on the task — using both is reasonable but adds reward-design complexity.
- The SAM3 dependency is non-trivial — quality of grounding-reward supervision is bounded by SAM3's segmentation accuracy.
- Combined reward shaping (answer correctness + grounding density) needs careful weighting. Over-rewarding grounding can produce traces that exhaustively box every object instead of solving the task.
- Generalises beyond counting/spatial only if the task naturally has identifiable visual referents. Abstract visual reasoning (color schemes, mood) doesn't fit cleanly.
- Distinct from earlier "set-of-marks" prompting (Yang et al. 2023): SoM puts numbered marks on the image at *inference* time; this approach trains the model to *produce* groundings as part of its CoT.

## Sources

- Paper: *Thinking with Visual Grounding* — Zhang, Deng, Chang, Wang, UCLA, 2026 — arXiv 2606.16122.
- Underlying segmentation tool: SAM3 (Segment Anything 3).
- Sibling concept: process reward models — see [prm](../post-training/reasoning/prm.md).

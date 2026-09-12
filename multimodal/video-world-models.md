# Video World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Autoregressive causal video models trained to unroll long visual histories under an implicit world simulation. "World in World" shows a **training-free** control interface: inject source frames, target-viewpoint projections, and geometry renderings as extra visual tokens into the model's own self-attention, and the frozen model re-renders the world along a new camera path — no fine-tuning, no adapter. Supports camera-controlled re-rendering, long-horizon revisiting, and human-motion transfer.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [README](README.md), [../agents/code-world-models](../agents/code-world-models.md)

---

## What it is

A **causal video world model** is an autoregressive generator over frames: at each step it predicts the next frame given all prior frames, so an infinite roll produces a coherent trajectory through the world. Modern examples are diffusion-transformer video generators trained with causal (not bidirectional) attention over time — the model *is* an implicit world.

The problem: controlling one. Every controllable video paper this year has trained an adapter (ControlNet-style) or a control-token head. "World in World" argues you can control a frozen causal video model by editing what its attention sees.

## How it works

Pipeline for camera-controlled re-rendering:

1. **Source encoding.** Feed the original video frames as history through the frozen model's tokenizer.
2. **Target-viewpoint evidence.** Project the source frames onto the target camera path (using estimated depth) to produce a coarse-but-geometrically-correct projected video. Render additional geometry cues (normal maps, occupancy) from a lightweight scene reconstruction.
3. **Attention injection.** Concatenate these extra visual tokens to the history and let the model's *own* causal self-attention treat them as additional context. No parameters change.
4. **Generate.** The model rolls forward, producing a re-rendered video along the target trajectory. Because the projected video already fixes geometry, and the geometry cues fix depth/occlusion, the model's stochastic drift is bounded — it fills in appearance and temporal consistency.

Same mechanism supports:
- **Long-horizon revisiting** — feed prior frames from the same location as evidence when the trajectory loops back.
- **Human-motion transfer** — feed a pose-rendered source video plus target subject frames as evidence.

## Why it matters

- **Kills adapter training.** No LoRA, no ControlNet, no per-task fine-tune. The base video model's attention is the control surface.
- **Test-time compute for video.** Same "add evidence, sample better" pattern as LLM test-time compute — the primitive scales when strong base models get stronger.
- **Composes across tasks.** Camera control, motion transfer, and revisiting all reduce to the same "inject visual evidence" recipe with different evidence.
- **Structural counterpart to code world models.** Code models represent the world as executable programs; video world models represent it implicitly in weights. Both are legitimate world-model paradigms; the training-free control interface makes the video path practical.

## Gotchas & tricks

- **Evidence quality dominates.** Bad depth or geometry cues → the model over-trusts them and produces glitches. Pair with a robust depth estimator and pre-filter uncertain regions.
- **Attention budget.** Longer evidence stacks compete with history for context length; prune to the most-relevant evidence tokens (VLM-scored or nearest-neighbor).
- **Not for pure imagination.** Works when there's *some* source signal to project. Purely generative "new world" prompts still need adapters or fine-tunes.
- **Base-model dependence.** Only works with a strong causal video generator. Weak bases produce weak worlds; adding evidence doesn't fix that.
- **Camera-path priors.** Wildly out-of-distribution trajectories (sudden 360° flips) exceed the base model's implicit prior; smooth the target path.

## Sources

- Paper: *World in World: Explore the World with World Models* — Song, Yang, Zhang (Westlake AGI Lab), 2026 — [arXiv:2609.11548](https://arxiv.org/abs/2609.11548)
- Related: ControlNet, image-to-video adapters — the training-heavy alternative this replaces.

# Visually Grounded Thinking
*Depth — VLM chain-of-thought that interleaves text with explicit point/box references to image evidence, trained with grounding-aware RL.*

**TL;DR:** VLM reasoning traces sound right but leave the supporting visual evidence implicit, making them unverifiable. **Visually grounded thinking** (Zhang, Deng, Chang, Wang, UCLA, arXiv 2606.16122) interleaves each reasoning step with a point or box that references the image region used for that step. Trained via a synthesis pipeline that distills correct traces and a SAM3 agent to derive grounding supervision, then post-trained with **grounding-aware RL** that adds a dense grounding reward to answer correctness. On Gemma3-4B-IT, beats both the base model and non-grounded thinking; on spatial tasks the 4B grounded model matches or beats Gemma3-27B-IT.

**Prereqs:** [README](README.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md) · [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A reasoning format and training recipe for VLMs in which intermediate steps are annotated with explicit visual references — points or bounding boxes naming the image region that justifies the step. Two parts:

- **Data**: a synthesis pipeline that takes correct visual-reasoning traces, extracts the visual objects each step refers to, grounds them via a SAM3-based agent, and produces aligned point/box supervision.
- **Training**: grounding-aware RL combining answer-correctness reward with a **dense grounding reward** that scores whether emitted object references match the correct image evidence.

## How it works

### Data synthesis

1. Run a strong VLM to produce reasoning traces for training questions.
2. Filter for answer correctness.
3. For each step, an LLM identifies the visual objects referenced.
4. A **SAM3-based agent** localizes each referenced object in the image, producing point and box masks.
5. Aligned supervision: each reasoning step is paired with `(text, point, box)`.

### Grounding-aware RL

Standard RLVR-style loss with two reward components:

```
R_total = R_answer + λ · R_grounding
```

- `R_answer` — sparse correctness reward (matches RLVR).
- `R_grounding` — **dense** per-step reward measuring how well emitted object references overlap the ground-truth masks from data synthesis.

The dense grounding reward is what makes box grounding work well on spatial tasks; with sparse answer-only reward, the model emits boxes but doesn't anchor them precisely.

### Format split

- **Point grounding** — wins on counting tasks. A point per counted object gives a clean supervision target.
- **Box grounding** — wins on spatial reasoning, especially with the dense grounding reward shaping where the box should land.

## Why it matters

- **Verifiable reasoning at small scale.** A 4B model with grounded thinking matches or beats a 27B model on spatial benchmarks — the grounding signal does work scaling wouldn't.
- **The grounding reward is RLVR-shaped.** It's a programmatic check (mask overlap), so it can be added to any RLVR pipeline that has access to grounded supervision.
- **Verifiability for human review.** Each reasoning step ships with a visual reference; reviewers can check whether the model is actually looking at the right region.

## Gotchas & tricks

- **SAM3 quality is load-bearing.** Bad grounding in the synthesis pipeline contaminates every example. Filter aggressively.
- **Sparse vs dense reward matters by task.** Counting benefits from point grounding with even a sparse reward; spatial reasoning needs the dense per-step grounding reward.
- **Format consistency across data and inference.** If training mixes point and box annotations the model produces a mix; deciding the format up-front matters for downstream usability.
- **Compatible with long-CoT RL.** This sits on top of standard long-CoT RL recipes — the grounding reward is an addition, not a replacement.

## Sources

- Paper: *Thinking with Visual Grounding* — Junkai Zhang, Yihe Deng, Kai-Wei Chang, Wei Wang, UCLA, 2026, arXiv 2606.16122.

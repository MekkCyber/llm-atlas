# Visually Grounded Thinking
*Depth — VLM reasoning interleaved with explicit point/box grounding of visual evidence.*

**TL;DR:** Train a VLM to produce reasoning traces that **alternate natural-language thoughts with explicit point or box references to the image regions** that each thought relies on. The recipe combines a **synthetic data pipeline** (correct CoT traces are grounded by a SAM3-based agent into aligned point/box supervision) and **grounding-aware RL** (dense rewards for whether generated references actually hit the right image regions). Zhang, Deng, Chang, Wang (UCLA), 2026. Lets a 4B Gemma3 match or beat the 27B variant on spatial reasoning.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/_rl.md](../post-training/_rl.md), [README.md](README.md)

---

## What it is

VLMs that "think" with chain-of-thought can sound right while pointing at the wrong evidence — the natural-language trace is unverifiable, and supervision is sparse (only the final answer is checked). Visually grounded thinking requires the model to **commit to specific image regions** as it reasons:

```
<thought>The plate has three apples and two oranges...</thought>
<point>(0.35, 0.42)</point>
<thought>...with one apple partially occluded by the bowl.</thought>
<box>(0.40, 0.50, 0.55, 0.65)</box>
<answer>5 fruits</answer>
```

Each `<point>` or `<box>` is verifiable against ground-truth visual segmentation, which turns a previously sparse reward into a **dense process-style signal**.

## How it works

### Data synthesis pipeline

1. **Distill correct visual reasoning traces** from a teacher VLM on multimodal QA datasets.
2. **Extract referenced objects** from each trace using LLM-based parsing.
3. **Ground objects** with a **SAM3-based agent**: given the object description and the image, return the mask.
4. **Derive supervision** from masks: point supervision = mask centroid; box supervision = mask bounding box.

The result is a synthetic dataset of `(image, query, interleaved trace with points+boxes, answer)` tuples.

### SFT stage

Standard SFT on the synthetic interleaved traces. The model learns the format (where to insert `<point>` / `<box>` tags) and a first-pass grounding.

### Grounding-aware RL

Build on a GRPO objective ([grpo](../post-training/grpo.md)). For each rollout, three reward components:

| Reward | Signal |
| --- | --- |
| **Answer correctness** | 1/0 from the verifier — the standard RLVR signal. |
| **Point grounding** | Distance between predicted point and the gold point (from SAM3 mask). Best for counting tasks. |
| **Box grounding** | IoU between predicted box and the gold box. Best for spatial tasks. |

The composite reward is summed with task-tuned weights and plugged into the standard GRPO update. The dense grounding signal turns each reasoning step into a small RL gradient signal, not just the terminal answer.

## Why it matters

- **4B → 27B level on spatial reasoning** from grounding alone. Counting and spatial reasoning benchmarks see consistent gains over both the base and a non-grounded CoT baseline.
- **Grounding rewards are the multimodal analogue of process rewards** ([prm](../post-training/reasoning/prm.md)) — cheap to compute (SAM3 + IoU) and dense in step count.
- **Point ≠ box.** Point grounding is enough for counting; box grounding is needed for spatial relations. The right primitive depends on the task class.
- The data pipeline (CoT → SAM3 → aligned supervision) is reusable for any multimodal RL training that needs a visual process signal.

## Gotchas & tricks

- **SAM3 quality is the data ceiling.** SAM3 errors propagate into the gold points/boxes; ambiguous descriptions ("the small one") get noisy ground truth.
- **Box grounding without dense reward underperforms.** The paper isolates that pure outcome-reward training on grounded format doesn't transfer; the dense reward is doing real work.
- **Point grounding on spatial tasks is a noisy signal.** Spatial tasks need extents, not centroids. Use the right primitive per task.
- **Trace length grows.** Interleaving format tags inflates trace length by ~30–40%. KV-cache costs proportionally.
- **Tag escape is a parsing problem.** The model occasionally emits malformed tag boundaries; format reward (small but nonzero) cleans this up.

## Sources

- Paper: *Thinking with Visual Grounding* — Zhang, Deng, Chang, Wang (UCLA), 2026, arXiv 2606.16122.
- Paper: *Segment Anything 3 (SAM3)* — the grounding agent in the data pipeline.
- Paper: *Gemma 3* — base model family used in the experiments.
- Background: [grpo](../post-training/grpo.md) for the RL objective.

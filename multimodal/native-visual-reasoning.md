# Native Visual Reasoning
*Depth — treating visual generation as the reasoning substrate itself, not just input/output.*

**TL;DR:** Native visual reasoning generates images or videos *as the reasoning trace*, not merely as input to interpret or output to render. Progress has been stalled by the lack of scalable training tasks, reliable feedback, and controlled substrate comparisons. VBVR-Pro (Xu et al., 2026) provides all three: 300 procedurally generated tasks, deterministic per-task rule-based scorers that plug into RL, and a modality-ablation harness across 30+ image, video, and interleaved generators.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/rubric-based-rl.md](../post-training/rubric-based-rl.md), [../evaluation/vbvr-pro.md](../evaluation/vbvr-pro.md)

---

## What it is

In native visual reasoning, the model produces intermediate images or video frames as part of its problem-solving process — e.g. drawing a partial diagram, generating a hypothetical world state, rendering a sequence of physical evolutions — and those visual artifacts are the reasoning trace rather than a decorative side product. The contrast is with "vision-language reasoning" where the model outputs *text* about images: native visual reasoning generates *visuals*, and the visuals themselves carry the argument.

The framing is empirically motivated by evidence in VBVR-Pro that certain tasks admit vision-native trajectories the language-only model can't reproduce: video generation is strongest for tasks requiring persistent spatiotemporal state tracking, while interleaved image+text generation is a compute-efficient alternative.

## How it works

Three problems have blocked progress and VBVR-Pro maps directly to solving each:

1. **Task scaling.** A closed-loop testbed of 300 *procedurally generated* visual reasoning tasks — enough curriculum to train, evaluate, and control task difficulty programmatically. Models trained on VBVR-Pro transfer to seven external visual-reasoning benchmarks.
2. **Verifiable rewards.** Rule-based, task-grounded scorers replace the failure-prone "MLLM-as-judge" pattern. The paper shows systematic failure modes of VLM judges; deterministic scorers avoid them and act as reliable RL reward signals for large-scale multi-task RL, with stronger post-RL performance across visual reasoning tasks.
3. **Mechanism study.** Controlled ablations across 30+ image, video, and interleaved generators isolate which modality is best for each task. Video generation dominates on persistent-spatiotemporal-state tasks; interleaved generation is compute-efficient for shorter horizons.

## Why it matters

If native visual reasoning is going to be a real capability rather than a demo, it needs (a) a training curriculum, (b) rewards that don't reward-hack, and (c) a way to compare substrates fairly. VBVR-Pro is the first testbed to provide all three, which is exactly the shape needed to do RL on visual generation as a first-class reasoning modality. The mechanism ablations also give a concrete practical rule: pick video for persistent state, interleaved for compute-efficiency.

## Gotchas & tricks

- **Reward-hacking of MLLM judges is documented, not hypothetical.** Deterministic rule-based scorers exist precisely to avoid it. Any RL loop over visual reasoning should assume MLLM-as-judge as a fallback, not a default.
- **Not all "visual chain-of-thought" is native.** Text-first CoT with occasional generated diagrams is still primarily text reasoning. Native visual reasoning has generated visuals carrying inferential weight, not just illustrating text.
- **Substrate choice is task-conditioned.** The image-vs-video-vs-interleaved answer is not "always video"; VBVR-Pro's mechanism study makes this decision empirical.

## Sources

- Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning* — Xu et al., 2026 — [arXiv:2608.26105](https://arxiv.org/abs/2608.26105)

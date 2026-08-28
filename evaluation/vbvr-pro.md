# VBVR-Pro
*Depth — closed-loop testbed for native visual reasoning: procedural tasks, verifiable rewards, and substrate ablations.*

**TL;DR:** VBVR-Pro (Xu et al., 2026) is a benchmark and training substrate for *native visual reasoning* (see [../multimodal/native-visual-reasoning.md](../multimodal/native-visual-reasoning.md)). It provides 300 procedurally generated visual reasoning tasks, deterministic rule-based scorers that replace VLM-as-judge, and controlled ablations across 30+ image, video, and interleaved generators — so RL over visual reasoning gets a training curriculum and a reward signal that doesn't hack.

**Prereqs:** [../multimodal/native-visual-reasoning.md](../multimodal/native-visual-reasoning.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/rubric-based-rl.md](../post-training/rubric-based-rl.md)

---

## What it is

A closed-loop testbed for native visual reasoning, structured to be trainable, verifiable, optimizable, and experimentally controllable. Three pieces:

1. **Task scaling.** 300 procedurally generated tasks organized by a domain × skill taxonomy; procedural generation supports controlled difficulty and effectively unlimited fresh instances.
2. **Verifiable rewards.** Per-task rule-based scorers grounded in deterministic task rules — fine-grained alignment with human judgment, no VLM-as-judge failure modes, cleanly plug into RL as reward signals.
3. **Mechanism study.** 30+ image, video, and interleaved generators evaluated on the same task suite for controlled modality comparisons.

## How it works

**Procedural task generation.** Each task type has a generator that emits fresh instances at controllable difficulty — this both supports large-scale training curricula and defeats memorization (no fixed test set to overfit).

**Rule-based scorers.** For each task, a task-specific deterministic scorer checks the generated visuals against ground-truth structural rules — object counts, spatial relations, temporal ordering, quantitative attributes. These plug directly into RL objectives ([RLVR](../post-training/rlvr.md)-style) as verifiable rewards.

**Substrate ablations.** The same tasks are attempted by many models spanning image-only, video, and interleaved image+text generators. Findings:

- **Video generation** is strongest for tasks requiring persistent spatiotemporal state tracking.
- **Interleaved generation** is a compute-efficient alternative for shorter horizons.
- Probing and ablations suggest **vision-native trajectories** exist and are crucial to visual reasoning — language-only reasoning does not reproduce them.

**Transfer.** Models trained on VBVR-Pro tasks transfer to seven external visual reasoning benchmarks (RISE-Video, MME-CoF-Pro, BabyVision, and others named in the paper).

## Why it matters

VBVR-Pro fixes the two blockers for native-visual-reasoning research in one benchmark: procedural curriculum for training and deterministic scoring for reward. The mechanism study also produces the first controlled evidence for *when* video vs interleaved generation is the right substrate, giving practitioners an empirical rule instead of a preference.

## Gotchas & tricks

- **VLM-as-judge failure modes are documented.** The paper enumerates recurring failure modes of MLLMs used as judges; treat any RL reward that leans on an MLLM judge as suspect.
- **Deterministic scorers are per-task.** The upfront authoring cost of writing rule-based scorers is real; procedural task generators help amortize it because one scorer covers a whole task family.
- **300 tasks / 810 instances is a lot but not infinite.** The procedural generators are what actually scale training; the released 810-instance fixed set is the evaluation snapshot.

## Sources

- Paper: *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning* — Xu et al., 2026 — [arXiv:2608.26105](https://arxiv.org/abs/2608.26105)

# WROP — World-Model Object Permanence Benchmark
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for evaluating **object permanence** in video generation / world models: 150 hand-designed cognitive-science-inspired tasks across six categories (occlusion, containment, deletion, etc.), procedurally expanded to 10K+ samples per task, plus a 300-question multiple-choice exam scored by an LLM judge. Introduced together with **PWM-WROP**, a 16B continuation model trained specifically for object permanence that ranks first among continuation models and third overall on the exam.

**Prereqs:** [multimodal README](../multimodal/README.md)
**Related:** [evaluation README](README.md)

---

## What it is

Video generation models are increasingly framed as "world models" but there is no clean measurement of whether they track objects through occlusion, containment, and other permanence-breaking events. WROP frames the Piagetian object-permanence milestone as a generative video evaluation: given a prompt or reference frame(s), does the model preserve an object's identity, position, and count after a distractor event?

## How it works

- **Task taxonomy.** Six categories, each targeting one aspect of permanence (e.g. occlusion by a passing object, containment inside another object, deletion / replacement, count preservation across cuts).
- **Data engine.** 150 hand-designed scenes act as templates; a procedural generator produces 10K+ variants per task by randomizing textures, camera paths, distractor identity, and timing.
- **Exam.** 300 handpicked, human-verified questions used to score models; an LLM judge grades free-form continuations against a reference by asking multiple-choice permanence questions ("Is the red ball still behind the box?").
- **Model splits.** Continuation models (frame → future) evaluated separately from reference-to-video models (multi-frame → interpolation); scores reported per category and aggregate.
- **PWM-WROP.** 16B continuation model trained on procedurally-generated WROP data plus general video; RL-finetunes for permanence scoring. Ranks first among continuation models and third overall on the exam.

## Why it matters

- Puts a concrete number on the "video generators don't understand physics" critique — turns a philosophical debate into a leaderboard.
- Establishes a training target: WROP scores improve substantially with modest permanence-focused fine-tuning, suggesting the gap is trainable, not architectural.
- Bridges cognitive-science-style probing into modern generative video evaluation.

## Gotchas & tricks

- LLM-judge scoring can be gamed by models that produce ambiguous frames — human spot-checks on the 300-question exam are recommended.
- Continuation vs. reference-to-video splits matter: reference-to-video models exploit interpolation to appear more "permanent" than they are on true continuation.
- Procedural expansion helps training but the *exam* is fixed at 300 items — avoid leakage by holding out generator seeds.
- Category imbalance: containment and multi-object count are the hardest categories and dominate the top-of-leaderboard gap; a model with high aggregate can still fail these.
- Prompt sensitivity: continuation models score noticeably higher when the reference frame is cleaner — noisy first frames confound the object-permanence signal.

## Sources

- Paper: *Training Object Permanence in World Models* — Zhang et al. (multi-institution), 2026 — [arXiv:2609.28654](https://arxiv.org/abs/2609.28654).

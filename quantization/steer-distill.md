# SteerDistill (closed-loop teacher-steered quantization)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A quantization-aware distillation recipe that targets *agent* workloads. A full-precision teacher watches a quantized student run a multi-turn agent trajectory, intervenes only on failed decisions, writes corrections, and distills those corrections back into the student in a closed loop. Reported in *SteerDistill* (2611.13410): W4A16 agents end within ~1 point of BF16 teachers across 5 agentic benchmarks and 3 base models, where vanilla trajectory distillation and standard QAT leave a much larger gap.

**Prereqs:** [_number-formats.md](_number-formats.md)
**Related:** [fp8.md](fp8.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md), [../agents/README.md](../agents/README.md)

---

## What it is

Low-bit weight quantization (W4A16, W4A8) usually tanks agentic LLM performance even when it's near-lossless on static benchmarks. The reason is error compounding: a single wrong tool call derails a long-horizon episode, and small numeric drifts at each decoding step compound into those wrong tool calls.

SteerDistill treats quantized agent *alignment* as a post-training problem: the student is already close to the teacher on most tokens; the training signal should concentrate on the handful of decisions where it goes off-track.

## How it works

Four steps, repeated until convergence:

1. **Run the student.** Let the quantized model execute an agent trajectory end-to-end on training tasks.
2. **Detect failures.** Compare against a task-level success criterion (reward model, verifier, or final-answer check).
3. **Teacher steers.** For each failure, re-run the trajectory with the BF16 teacher; let the teacher produce a *corrective action* at the earliest divergence point.
4. **Distill corrections.** Fine-tune the student on the ⟨prefix, corrected action⟩ pairs. Only those pairs — not the teacher's full trajectory.

Over several rounds, the student learns to make those corrections unprompted, and the gap to BF16 closes.

## Why it matters

- Reframes low-bit agent quality as a distillation-alignment problem rather than a pure numerics problem. That reframing is why it closes the gap where QAT cannot.
- Recovers near-BF16 performance at W4A16 for long-horizon agents, which is the deployment regime where low-bit quantization used to be a non-starter.
- Composes with any QAT or post-training quant method — SteerDistill is the *supervision loop*, not a new numeric format.

## Gotchas & tricks

- Needs a reliable per-step failure detector. On trajectories without verifiable rewards (open-ended), use an LLM-judge; expect noisier training.
- Correcting only at the earliest divergence is important — correcting *every* failed step floods training with trivial fixes and destabilizes the student.
- Training is round-based; after each round, re-evaluate and gather new failure trajectories. Static datasets saturate quickly.
- Teacher BF16 inference cost dominates training; cache corrected trajectories aggressively.

## Sources

- Paper: *SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering* — Cui, Li, Hao, Ding, Liu, Cao (2026) — [arXiv:2611.13410](https://arxiv.org/abs/2611.13410)

# Failure Attribution for Agentic Systems
*Depth — identifying which step in a failing agent trajectory caused the failure, so the system can be debugged and improved.*

**TL;DR:** Agent runs fail across many steps; knowing *which* step is the root cause is the debugging bottleneck. Existing approaches either use expensive prompting pipelines or require step-level error annotations that are costly to collect. **Oat** (Yeh et al., 2026) reframes failure attribution as **unsupervised one-class learning**: train only on *successful* trajectories with neural controlled differential equations (NCDEs) to model the "success manifold" in latent space; at inference, score each step of a failure trajectory by its deviation from that manifold. With **100 successful trajectories**, Oat is **200–5000× faster** than prompting-based baselines and beats them by **+20% F1 in-domain, +7% F1 OOD**.

**Prereqs:** [agent-harness](agent-harness.md)
**Related:** [harness-handbook](harness-handbook.md), [../evaluation/agent-evaluation-infra.md](../evaluation/agent-evaluation-infra.md)

---

## What it is

Failure attribution is the diagnostic task of pointing at the step (or steps) in a multi-step agent trajectory that caused it to fail. It is separate from *detecting* failure (that the run failed at all) and from *fixing* failure (patching the harness or model). Attribution is what closes the debugging loop: without it, you know the run failed but not where to intervene.

Two existing approaches:

- **Prompting-based attribution.** Feed the trajectory to a large LLM and ask "which step went wrong?". Works but is per-trajectory expensive and slow.
- **Supervised attribution.** Train a classifier on failure trajectories with step-level error labels. Accurate but requires an annotation pipeline that scales badly.

**Oat** is the unsupervised alternative: learn what success *looks like* in latent space, then flag anything that doesn't look like it.

## How it works

1. **Encode trajectories into a latent representation.** Each trajectory becomes a sequence of latent states — one per step — via an encoder over prompt/tool-call/observation triples.
2. **Fit a neural controlled differential equation (NCDE) to the success manifold.** NCDEs model continuous dynamics over irregular time series and here capture the *smooth trajectory* successful runs trace through latent space. Only ~100 successful trajectories are needed to fit it usefully.
3. **At inference, score each step of a failure trajectory** by its deviation from the learned dynamics — an anomaly score per step. High-scoring steps are the candidate root causes.
4. **Return the ranked set of error steps** for downstream use (developer inspection, automated fix generation, or feedback to the harness).

The one-class-learning framing (train on success, detect at inference) sidesteps failure-annotation collection entirely.

## Why it matters

- **Cheap, per-agent deployment.** 100 successful runs is a bar most teams already clear. No annotation pipeline required.
- **Order-of-magnitude speedup.** 200–5000× vs. prompting-based baselines means attribution can run inline on every failure, not just spot-checked.
- **Generalizes OOD.** +7% F1 improvement on OOD data indicates the success-manifold view captures a real, transferable structure rather than memorizing distributional quirks.
- **Bridges agent research and mature ML.** NCDEs and one-class learning are established techniques; connecting them to agent debugging opens a productive cross-pollination.

## Gotchas & tricks

- **The success set must be representative.** If successful trajectories in your training data don't cover the operating envelope, novel-but-successful runs at inference will look like anomalies. Refresh the success set as the harness evolves.
- **Anomaly ≠ root cause.** A step downstream of the true root cause often looks abnormal too (the trajectory has already drifted). Prefer earliest high-scoring step or use additional heuristics on the score profile.
- **Latent encoder choice matters.** The whole approach is only as good as the trajectory encoder — poor encodings collapse the manifold.
- **Complementary with prompting-based attribution.** For high-value failures, run both: Oat as the cheap first pass, prompting for the ones Oat flags.

## Sources

- Paper: *Tracing Agentic Failure from the Flow of Success* — Yeh, Zhu, Deep, Li, 2026 — [arXiv 2607.12747](https://arxiv.org/abs/2607.12747). University of Wisconsin-Madison + Microsoft Research. Introduces Oat and the one-class NCDE formulation.

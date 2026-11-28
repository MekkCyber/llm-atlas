# Speculative streaming (interactive world models)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative streaming is the speculative-decoding idea applied to interactive video world models. During the gap before a user picks an action, the model pre-rolls 1–8 candidate action-conditioned futures; once the real action lands, **single-latent commitment** keeps only the matching candidate and drops the rest, with no visible tearing. Introduced in ShowNow (2026); runs 5B @ 480p @ 16 fps with 8 prepared futures on one H100.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

Interactive world models have to generate the next video frame **conditional on whatever action the user picks**. A naive implementation waits for the action, then renders — the user feels every ms of inference. Speculative streaming overlaps this: between actions, the model speculatively renders candidate futures for each plausible action in parallel; when the real action arrives, it commits to the matching candidate.

## How it works

Three pieces:

1. **Fine-grained preparation.** During the inter-action gap, generate K candidate latent streams, each conditioned on one of K anticipated actions. K is a configurable width (1–8) chosen by the compute budget.
2. **Single-latent commitment.** Each candidate is produced *as a single latent per step*, not as a fully-rendered frame. When the real action arrives, the system picks the matching latent and runs only its decoder, discarding the other K−1 latents without side effects. Because commitment is latent-level, there is no partial rendering to roll back.
3. **Playback-time scheduling.** The pipeline runs the preparer and the decoder on overlapping time slices, so decode latency hides inside the inter-action window.

Latency is bounded by *commitment* (one decoder pass) rather than by *generation* (preparation of all K futures).

## Why it matters

Previously the only way to get interactive world models was to shrink the generator until it ran in real time — sacrificing visual quality. Speculative streaming lets you use a 5B model at 480p 16 fps on a single H100 while still feeling instantaneous, because the user-perceived latency is the decoder time, not the full-model time. The pattern generalises beyond video: any interactive generative system with discrete-ish branching user input (games, dialogue UIs, interactive simulators) can pre-roll branches during the user's thinking time.

## Gotchas & tricks

- The speedup is bounded by the action distribution's entropy: if the user almost always picks one of 2 actions, K=2 is enough; a 20-way action space forces you to either widen K (more compute) or accept miss-penalty latency on rare branches.
- Single-latent commitment requires the latent space to be *action-separable* — otherwise picking one candidate's latent leaves artefacts of the discarded candidates in the decoder's temporal context.
- Candidate preparation must be wall-clock bounded by the inter-action gap or latency regresses — paper reports preparation width 1–8 as the useful band on a single H100.
- Miss-penalty is asymmetric: when the user picks an un-prepared action, the full decode has to run from scratch and the frame skips, so training should bias K toward high-entropy action regions.

## Sources

- Paper: ShowNow: Fine-Grained Speculative Streaming for Responsive World Models — Fang & Shou (Show Lab, NUS), 2026 — [arXiv:2611.34693](https://arxiv.org/abs/2611.34693)
- Related: speculative decoding (Leviathan et al., 2023) — original idea for language models

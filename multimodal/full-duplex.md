# Full-Duplex Audiovisual Interaction (Duo)
*Depth — intent/realization decomposition + causal chunked timeline for real-time audio+video assistants.*

**TL;DR:** Full-duplex means the model **perceives and generates at the same time** — it listens / watches while it speaks / renders, enabling natural interruptions and shared on-screen activities. Duo's Reciprocal World Model (RWM) decomposes this into two modules: **Intent** (autoregressive, decides what to do) and **Realization** (generative, produces audio and pixels). Both operate on a unified **causal chunked timeline**, so input and output overlap in real time, and the paper's **dual rollout distillation** reduces train/test distribution shift between the two modules.

**Prereqs:** *(none)*
**Related:** [../post-training/_post-training](../post-training/_post-training.md)

---

## What it is

A step past turn-taking. Turn-taking agents listen, process, then respond; full-duplex agents keep both streams open. For a voice + video assistant this means: it can start speaking before the user finishes, stop when interrupted, and co-watch a video stream with the user — things that fundamentally cannot be bolted on top of a serial ASR → LLM → TTS pipeline.

Duo is one of the first systems to train for full-duplex audio+video jointly rather than glue it together post-hoc.

## How it works

### Intent + Realization split

- **Intent** is an autoregressive model over an interleaved stream of **action tokens** (decisions: *speak*, *stay silent*, *gesture*) and condensed context tokens.
- **Realization** is a generative model (flow-matching / autoregressive hybrid) that renders the Intent's decisions into continuous audio and video chunks.

### Causal chunked timeline

The user's stream and the model's stream are both quantized into short time chunks on a single shared timeline. At every chunk boundary the Intent module makes a decision conditioned on all *prior* chunks (input + output), preserving causality while allowing overlap.

### Dual rollout distillation

Intent is trained on teacher-forced rollouts and deployed on its own rollouts; the distribution shift is nontrivial because realization-side outputs feed back into Intent's context. Dual rollout distillation trains Intent on a mix of teacher-forced and self-sampled timelines so it sees realistic context at train time.

## Why it matters

- **Trainable rather than engineered duplexing.** The industry pattern has been scripted interruption detection and TTS-cancellation; Duo makes duplexing a learned behavior.
- **Audio + video + text in one model.** Unifies modalities that normally sit in separate services, which is what frontier assistants are moving toward.
- **Causal-chunk framework is reusable.** The timeline structure is likely to recur in other duplex models independent of Duo's specific architecture.

## Gotchas & tricks

- Chunk size is a latency vs quality knob: smaller chunks let the model interrupt faster but shrink the context per decision.
- Train/test distribution shift from Intent's own realization outputs is the main failure mode; without something like dual rollout distillation the model drifts.
- "Full-duplex" in this paper is audio+video; many prior "duplex" works are audio-only turn-taking and are a different (easier) problem.

## Sources

- Paper: *Duo: A Reciprocal World Model for Full-Duplex Interaction* — 2026 — arXiv 2611.15131 — https://arxiv.org/abs/2611.15131

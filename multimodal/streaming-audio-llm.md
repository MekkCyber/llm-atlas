# Streaming audio LLM (SoundFlow / Audio Interaction Model)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A streaming audio LLM continuously perceives audio (sound, environment, instructions), decides when to act, and generates a response — all in a single always-on loop, replacing the offline "transcribe → reason → speak" stack used by today's Large Audio Language Models. The SoundFlow framework instantiates this perceive–decide–respond loop end to end: streaming-native data construction, comprehension-aware training, asynchronous low-latency inference. From Xie et al., 2026.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/streaming-multi-agent.md](../agents/streaming-multi-agent.md), [../agents/README.md](../agents/README.md)

---

## What it is

Existing Large Audio Language Models (LALMs) treat audio as a finished input: you feed a complete clip, the model returns a complete response. Streaming audio models exist for *single* tasks (streaming ASR, voice chat), but each handles one task and lacks general instruction-following.

A streaming audio LLM unifies both: a single model that listens continuously, *decides on its own* when to respond (or interrupt, or stay silent), and supports the full range of audio tasks — ASR, audio QA, voice chat, proactive intervention. The unit of operation is the *step* of audio (a short chunk), not the whole clip.

## How it works

The SoundFlow framework has three stages, each redesigned for the streaming regime:

1. **Streaming-native data construction.** Source clips are chunked along the same time axis the model will see at inference, with per-chunk annotations of *what the model should do* (wait, respond, interrupt). The StreamAudio-2M dataset is 2.6M chunked examples spanning 7 fundamental abilities and 28 sub-tasks.
2. **Comprehension-aware training.** Per-chunk supervision teaches the model to choose among (a) silence, (b) continue listening, (c) respond, (d) interrupt. The choice is made from the *semantics* of the incoming audio (does the user need help, did they finish a thought) rather than VAD-style heuristics on silence/energy.
3. **Asynchronous low-latency inference.** Perception (encoder) and generation (decoder) run on separate execution streams. The encoder continuously processes incoming audio; the decoder runs only when "respond" is chosen. Latency is bounded by chunk size, not full-clip length.

The model retains offline task execution (give it a full clip and it does the task) but adds online behaviors that pure offline models cannot express.

## Why it matters

- **Proactive turn-taking.** Today's voice assistants need wake words or button presses; SoundFlow models can decide *on semantic grounds* when to speak, much closer to human conversation.
- **Latency.** Audio↔text↔audio in production stacks today is multi-second round-trip; a unified streaming model halves at least one direction.
- **Benchmarks.** Proactive-Sound-Bench introduces an evaluation surface for proactive intervention, a behavior class the field has not measured before.
- **Reference recipe for any-modality streaming.** The perceive–decide–respond loop generalizes to streaming video and streaming agent inputs; SoundFlow is one of the first end-to-end demonstrations.

Reported across 8 benchmarks: competitive on conventional offline tasks while *uniquely* delivering streaming ASR, online instruction-following, and proactive help.

## Gotchas & tricks

- **Decide-step training data is scarce.** Most existing audio corpora are full-clip task data; the per-chunk "what to do" annotation has to be partially synthetic.
- **Encoder–decoder asynchrony.** Running them on separate streams complicates standard inference frameworks; engineering work is non-trivial.
- **Interruption hurts user trust if mis-timed.** False positives on "respond now" are conspicuous. Tuning the decide-head's threshold matters.
- **Generalizes only as far as the chunking generalizes.** Chunk size at train time is a strong inductive bias on inference behavior.

## Sources

- Paper: *Audio Interaction Model* — Xie et al., 2026 — [arXiv:2606.05121](https://arxiv.org/abs/2606.05121).
- Datasets: StreamAudio-2M (training), Proactive-Sound-Bench (eval).

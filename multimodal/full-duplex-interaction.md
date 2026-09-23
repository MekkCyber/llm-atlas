# Full-Duplex Interaction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An interaction pattern where an omni-modal model **listens and speaks simultaneously**, sharing a single causal timeline for user inputs, model outputs, and delegation events. Slow work (search, tool calls, deep reasoning) is offloaded to a second "harness" model running asynchronously; the foreground stays live and integrates results as they arrive. Enables voice/video assistants that stay conversational while thinking.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../agents/agent-harness.md](../agents/agent-harness.md) · [README.md](README.md)

---

## What it is

Traditional voice assistants are **half-duplex**: turn-based — you speak, the model listens; the model speaks, you listen. This breaks on the two interactions humans do without thinking:

- **Interruption.** Cutting the model off mid-sentence.
- **Overlap.** Backchannels ("mm-hmm"), overlapping speech, background chatter that the model should ignore but not treat as a turn-end.

**Full-duplex** systems treat perception and generation as concurrent processes on a single **causal timeline**. Every token — user audio, user video, model output, tool result, delegation event — has a timestamp on the same timeline. The model can be generating speech while attending to incoming audio, and can decide to yield, continue, or clarify at any moment.

Modern full-duplex omni-modal systems (Realtime-Venus, GPT-4o Realtime, Gemini Live) implement this pattern with a single trained model per modality set, no explicit turn-taking module.

## How it works

**Shared causal timeline.** All streams — user audio, user video, model tokens, and *delegation events* (start-tool, tool-result) — are interleaved in one causal sequence. The model attends over the full past and generates the next unit into its own stream. Turn-taking becomes an implicit decision (should I generate or stay silent?), learned rather than rule-based.

**Dual-loop runtime.** Two models cooperate:

- **Foreground model** — omni-modal (audio + optional video + speech generation). Runs in a low-latency loop: perceive, decide, generate. Small enough to be always-live (~9B in Realtime-Venus).
- **Harness / background model** — invoked asynchronously by the foreground model whenever a long-running task is needed (search, complex reasoning, code execution, deep tool chains). Results come back into the foreground timeline as delegation-result events.

The foreground never blocks on the background — it can talk about *anything else* while the tool call is in flight, then integrate the result naturally when it arrives.

**Post-training recipe.** SFT on labeled full-duplex trajectories that include (a) proactive utterances by the model, (b) mid-generation interruption handling, (c) delegation-and-integration patterns, then RL to sharpen turn-taking, interruption response, and cross-modal grounding.

## Why it matters

- **Latency is not the whole story.** Even with fast token generation, half-duplex systems *feel* stilted because they can't interject, backchannel, or gracefully handle interruption. Full-duplex closes the human-conversation gap.
- **Concurrent perception + generation** is what makes voice-chat-with-tools plausible: the model can research your question *while* asking a clarifying question about it.
- **Reusable pattern beyond voice.** The shared causal timeline + async delegation split works for any interactive agent (browser use, computer use) where perception and action need to stay coupled.
- **Benchmark evidence.** On **Full-Duplex-Bench v1.5** Realtime-Venus-Audio responds to **75%** of interruptions with **97 / 88 / 86%** continuation under backchannels / other-directed speech / background speech — above Gemini 3.1 Live and GPT-4o on all three continuation metrics.

## Gotchas & tricks

- **Turn-taking is learned, not rule-based.** Don't try to bolt a VAD (voice activity detector) on top; the model's own attention over recent audio decides when to speak. VAD as a hard gate destroys backchannel handling.
- **Delegation events must be first-class tokens.** If tool calls live outside the causal timeline (e.g. as async coroutines invisible to the model), the model can't reason about "I'm waiting on X" or integrate results in-context.
- **Latency budget is tight.** The foreground model must respond in <300 ms to feel live; keep it small and let the harness carry the heavy load.
- **Audio and video streams are asymmetric.** Audio needs low-latency continuous chunks (~80 ms); video can subsample far more aggressively. Model tokenization should reflect this.
- **Interruption ≠ end-of-turn.** The model must distinguish "the user interrupted with a real question" from "backchannel; keep going". This is data-collection-hard — needs annotated conversations with both patterns.
- **Post-training data is the hard part.** Full-duplex trajectories with mid-generation interruption and delegation are not naturally logged by half-duplex systems — collect them explicitly.

## Sources

- Paper: *Realtime-Venus: A full-duplex interaction system with asynchronous delegation* — Venus Team, Ant Group, 2026 — two 9B omni models + harness runtime; introduces the shared-timeline + dual-loop design and reports Full-Duplex-Bench v1.5 results. [arXiv:2609.13814](https://arxiv.org/abs/2609.13814)
- System: *GPT-4o Realtime* — OpenAI, 2024 — early production full-duplex voice system.
- System: *Gemini Live* — Google, 2024/2025 — audio+video full-duplex assistant.
- Benchmark: *Full-Duplex-Bench* — the reference eval for interruption handling and continuation.

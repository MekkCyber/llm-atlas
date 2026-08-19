# Streaming VLM (Perceive-While-Speak)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Turn-based VLMs process one image + one prompt at a time. A **streaming VLM** takes frames continuously *while* it is generating text — the model must simultaneously perceive incoming visual context, decide when to speak, when to stay silent, and when to *revise* what it has already said. MOSS-VL (Fudan, 2026) presents an end-to-end recipe: [gated cross-attention for vision](gated-cross-attention-vision.md), a synthesized "when to speak" corpus, and a staged curriculum concentrating all real-time-specific training in one light final stage over a strong offline foundation.

**Prereqs:** [../multimodal/README.md](README.md), [gated-cross-attention-vision.md](gated-cross-attention-vision.md)
**Related:** [../case-studies/moss-vl.md](../case-studies/moss-vl.md)

---

## What it is

Live voice+vision assistants (screen-share helpers, camera assistants, live-video coaches) can't wait for a full "user turn" before responding. The model must:

- **Perceive** continuously as frames arrive (typically 1–5 fps).
- **Speak** while frames keep coming — the text KV cache can't be invalidated per frame.
- **Stay silent** when the correct action is to wait for more information.
- **Revise** a prior utterance when new frames make the earlier claim wrong.

A streaming VLM is designed for this contract from architecture to training data, rather than bolting turn-taking onto an offline VLM at inference time.

## How it works

Three co-designed pieces (the MOSS-VL recipe):

1. **Architecture: separate pathways.** [Gated cross-attention](gated-cross-attention-vision.md) puts vision on a distinct attention pathway so new frames don't invalidate the text KV cache. The decoder can generate a token while the cross-attention updates against the newest frame.

2. **Data: a synthesized interaction corpus.** Real-time turn-taking labels are not naturally present in web data. The recipe *synthesizes* interaction traces annotating when-to-speak, when-to-stay-silent, and when-to-revise, and uses them as supervised targets. Turn-taking becomes a trained capability instead of an emergent one.

3. **Curriculum: staged, real-time-late.** Heavy offline VLM pretraining acquires visual understanding + language quality on standard data; a small final stage adds real-time controls on top. Concentrating all streaming-specific training in one light stage keeps the offline capability intact instead of forcing it to be relearned under a real-time reward.

Runtime: the harness pushes new vision features into the cross-attention key/value store at whatever rate the source provides; the decoder emits tokens (or the silence token) on its own schedule, gated by the trained when-to-speak controls.

## Why it matters

- Most open VLMs today are turn-based. Live voice+vision assistants are a fast-growing product surface where turn-based is the wrong abstraction.
- The pattern generalizes: *any* modality that streams (audio, sensor data, agent screen) benefits from a separate KV cache and a "when to speak" trained control, not just cameras.
- Sets a concrete evaluation axis for VLMs — not just "answer this VQA question," but "watch this video and interject appropriately."

## Gotchas & tricks

- **Silence must be an emitted token.** If silence is "predict nothing," the model has no way to learn *when* to be silent. Emitting an explicit silence token (or turn-holding token) that occupies decoder steps is the tractable design.
- **Revise is hard to supervise.** Synthesized traces have to contain plausible mistake-then-correction sequences without teaching the model to constantly waffle. Careful corpus design more than architecture.
- **Latency budgets are unforgiving.** Adding cross-attention layers costs decode-time compute; adding perceiver-style feature compression helps but has its own cost. The whole design lives inside a ~200ms budget.
- **Vision frame rate ≠ language token rate.** The two run on independent clocks. Reasoning about the coupling (how many frames per generated token, when to refresh the visual features) is where the streaming-serving stack differs most from turn-based serving.
- **Offline eval doesn't cover streaming.** A model that scores well on VQA can still be a poor streaming interlocutor; interactive benchmarks are the honest ones.

## Sources

- Paper: *MOSS-VL Technical Report* — Wang, Tan, Zhou et al. — arXiv:2608.15045 — 2026 (Fudan University / Shanghai Innovation Institute).
- See also: the [MOSS-VL case study](../case-studies/moss-vl.md) for the end-to-end system built from these primitives.

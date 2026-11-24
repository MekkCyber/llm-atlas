# Full-Duplex Agents (Duopra)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training recipe for **speech-first agents** that can both carry on a full-duplex conversation (speech in and out can overlap in time) **and** issue well-formed tool calls to a backend agent mid-turn. Duopra stacks protocol-aware SFT, RL rewarding timely + complete execution, and a projector-only repair stage. Lifts Full-Duplex-Bench v3 Pass@1 from 4.7% to 44.7% with the same backend, 49.3% with a stronger backend.

**Prereqs:** [_rl](../post-training/_rl.md), [ppo](../post-training/ppo.md)
**Related:** [_post-training](../post-training/_post-training.md)

---

## What it is

Classical voice assistants pipeline ASR → text LLM → TTS and lose the conversational timing properties that make speech natural (barge-in, turn-taking, mid-sentence correction). **Full-duplex speech models** keep listening and speaking simultaneously on separate channels, but have mostly been limited to chit-chat.

Duopra extends full-duplex to **agency**: the model treats a backend LLM/tool-using agent as an external worker. Mid-conversation, it emits a self-contained natural-language request to the backend on a side channel, continues conversing (or pauses gracefully) while the backend works, and resumes with the result.

---

## How it works

**Protocol-aware SFT.** The training corpus is labeled with speech + control tokens defining when to emit a backend request, what channel structure to use, and how to resume speech afterwards. SFT teaches the surface syntax of the full-duplex + dispatch protocol.

**Reinforcement learning on execution.** On top of SFT, the model is trained with RL whose reward combines:
1. **Task completion** (did the backend's work satisfy the user's request as spoken?)
2. **Timing** (was the request dispatched soon enough, and did the model keep the user engaged while waiting?)
3. **Conversational coherence** (does speech resume cleanly?)

The reward is sparse and trajectory-level; the authors use PPO-style updates.

**Projector-only repair stage.** After RL, a final stage freezes most of the model and fine-tunes only the speech-text projection layer to fix residual TTS/ASR interface bugs without disturbing the learned dispatch policy. This is cheap and prevents RL from degrading audio quality.

---

## Why it matters

The voice-agent stack has been bottlenecked by the ASR-LLM-TTS pipeline, which cannot support barge-in or parallel execution without heavy orchestration. Duopra shows a **native** full-duplex model can learn to dispatch to tools while staying conversational — a structural change, not more scaffolding.

The staged recipe (SFT → RL → projector repair) is also reusable: the projector-only repair stage is a general pattern for cheaply fixing audio-layer artifacts introduced by RL.

---

## Gotchas & tricks

- **Backend quality dominates headroom.** Same policy swapped to a DeepSeek backend takes Pass@1 from 44.7 → 49.3; the agent's cognitive work is largely offloaded.
- **Reward shape matters.** Pure completion rewards collapse timing; the three-term reward prevents the model from stalling conversation to wait for the backend.
- **Projector repair after RL** is the step most teams skip and most pay for — RL can leave audible artifacts without it.

---

## Sources

- Paper: *Duopra: From Full-Duplex Conversation to Full-Duplex Agency* — Chen et al., 2026 — [arXiv:2611.25773](https://arxiv.org/abs/2611.25773)

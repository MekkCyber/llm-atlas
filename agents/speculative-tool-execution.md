# Speculative Tool Execution

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Lift the speculative-decoding idiom from token-level inference up to **agent-level actions**. While the user's intent is still being formed (partial speech, partial query), the agent runs reasoning and tool calls on best-guess intents at **heartbeat intervals**. Only a separate Responder, gated on commit-time (end-of-utterance), decides which speculative work to use. Introduced for realtime voice agents in Heartbeat Thinking, with +16 pp speculative tool coverage and ~36% latency reduction on FDBench v3.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Standard agent loops wait for a full user turn before planning: ASR must complete, intent must be resolved, then reasoning + tools fire, then a response is generated. In latency-sensitive settings (voice agents especially) this creates a user-perceived gap between end-of-speech and start-of-response.

Speculative tool execution breaks the loop. The incoming stream (ASR hypothesis, partial query, in-progress document) is sampled at periodic "heartbeats." At each heartbeat, a Speculator runs reasoning + tool calls on the current partial as if it were the final input. When the final input commits, a Responder picks up whatever speculative work matches the committed intent and either uses it directly or discards it. Unused speculation is cancelled.

The analogy with [speculative decoding](../inference/README.md) is deliberate: a cheaper, parallel process guesses ahead; a verifier commits at the real boundary.

---

## How it works

### Heartbeat loop

```
   ASR stream  ──► partial_t at heartbeat intervals (e.g. 200ms)
                      │
                      ▼
            Speculator: reason + call tools on partial_t
                      │
                      ▼
            Speculated results pooled (keyed by hypothesized intent)
                      │
         (user stops speaking)   ───►   Responder:
                                           pick best speculative result
                                           for the committed intent,
                                           or re-run if none fits.
                                        cancel unused speculation.
```

### Three roles

- **Speculator.** The forward process. Takes a partial as if it were final input, runs the normal agent loop, caches results keyed by its hypothesized intent.
- **Responder.** Fires once at commit. Reads the committed input, retrieves matching speculative work from the pool, and produces the user-visible response.
- **Cancellation.** When new partial information arrives (ASR revises its hypothesis), stale speculative work is marked for cancellation. In-flight tool calls that are expensive-to-cancel may still complete; cheap-to-cancel ones (e.g. in-flight LLM reasoning) are preempted.

### Heartbeat granularity

Too fast (every token): wastes compute, every heartbeat redoes near-identical work.
Too slow (every few seconds): defeats the latency savings.
The paper uses heartbeats at ASR event boundaries — new words, pauses — which naturally paces the speculation to meaningful intent changes.

### Metrics the paper tracks

- **Speculative tool coverage.** Fraction of final-intent tool calls that were pre-executed speculatively (so the Responder has their results in cache). FDBench v3: 34.67% → 50.67% vs a windowed baseline.
- **Final-response latency.** Mean seconds from end-of-utterance to start-of-response. FDBench v3: 6.90s → 4.39s.
- **Quality on downstream benchmarks.** Improved on FDBench; slight regression on SpeechFCEval and AudioCRAG — the speed/precision tradeoff is real.

---

## Why it matters

- **Breaks the voice-agent latency wall.** The dominant cost in production voice agents is "wait for end of speech" before any reasoning starts. Speculative execution recovers that time.
- **Reusable template beyond voice.** Any agent that can act on a partial input — a browser-use agent on a half-typed query, a code assistant on an incomplete prompt — can apply the same pattern.
- **Measurable tradeoff.** Unlike many agent optimizations, the paper exposes the speed-quality tradeoff explicitly (quality rose on some benchmarks, dropped on others). This gives operators a knob rather than a magic win.
- **Composable with speculative decoding.** The two operate at different levels of the stack and should stack cleanly.

---

## Gotchas & tricks

- **Wasted compute is real.** Most speculative tool calls will be cancelled or discarded. Pricing a system this way requires accepting a multiplier on tool-call cost; it is a latency-for-cost trade.
- **Idempotent tool calls only (or expensive compensation).** Speculating a *side-effect*-ful tool (send email, make purchase) is dangerous — commit boundary must gate real actions, not just reads. The paper's deployment is on reads.
- **Cancellation protocol.** Needs tool clients that support cancel semantics. For HTTP calls, in-flight requests may still complete at the server; measure cost not just latency savings.
- **ASR revision cost.** When ASR revises its hypothesis, the pool must be invalidated carefully. Over-eager invalidation loses the cache; lazy invalidation serves stale results.
- **Not for text chat.** Text input arrives at commit (user hits send). The speculation window doesn't exist until the user starts typing *and you can read partial input* — which is possible but adds UX complexity.

---

## Sources

- Paper: *Heartbeat Thinking: Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents* — Zhang, Fu, Tong, Zheng, Wu, Yang, Li, Ant International, 2026 — [arXiv:2610.26233](https://arxiv.org/abs/2610.26233).
- Analogy: *Speculative Decoding* — Leviathan et al. 2023; Chen et al. 2023 — the inference-level pattern this generalizes.

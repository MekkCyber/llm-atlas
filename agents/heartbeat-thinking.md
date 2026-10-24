# Heartbeat Thinking
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Realtime voice agents waste latency waiting for the user to finish speaking before thinking or firing tools. Heartbeat Thinking speculatively reasons and executes tools **during** the user's utterance, on a short periodic "heartbeat" tick keyed to incremental ASR output. If the final transcript matches the speculated prefix, the reply (and its tool results) is already in hand when the user stops talking (Ant International, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

A speculative-execution pattern for voice agents, applied at the agent/tool-call level rather than at token decoding. The user is still speaking; the ASR is producing partial transcripts; the agent runs reasoning and tool calls on each tick against the current partial transcript, maintaining a shadow execution that it either accepts (if the user's final transcript matches the speculated prefix enough) or rolls back and re-runs.

## How it works

- **Heartbeat tick.** A short periodic interval (small fraction of a second) that fires while the user is speaking. Each tick: snapshot the current partial transcript, submit it to the reasoning stage and any dispatched tools, and record the speculation state.
- **Speculation state.** The agent tracks (a) which tool invocations were made, (b) what reasoning path they were conditioned on, and (c) a confidence score against future transcript extensions.
- **Commit / rollback.** When the user's turn ends (ASR confirms end-of-utterance), the system compares the final transcript to the last-committed speculation. If they match, the pre-computed answer is streamed back immediately. If they diverge, the speculation is rolled back and a fresh pass runs on the full final input.
- **Tool safety.** Tools with side effects are only allowed if idempotent or if the speculation framework can revert them; read-only tools (search, lookup) run freely each tick.

## Why it matters

- **Overlapping compute with user speech is the biggest latency lever for voice.** Turn-end latency is what humans perceive as "the AI is slow"; shrinking it is the user-facing win.
- **Generalises past voice.** The same shape works for any streamed input: live code review, co-piloted editing, streaming search. The ingredient is "ASR" with "stream of partial inputs."
- **Reusable speculation infra.** The commit/rollback machinery is similar to speculative decoding and transactional systems; the paper formalises it at the agent level.

## Gotchas & tricks

- **Side-effect tools need gating.** Firing a bank-transfer tool speculatively is unacceptable; a tool-allowlist for speculation vs commit is required.
- **Confidence scoring is hard.** Without a good match-probability estimator, you either over-speculate (wasted compute) or under-speculate (no latency win).
- **ASR errors propagate.** A hallucinated word in a partial transcript sends the agent down the wrong reasoning path for several ticks before the next partial corrects it.
- **Cost model is non-trivial.** You pay for every speculation tick; the net saving depends on cancellation rate, average turn length, and tool cost. The paper evaluates on realtime voice workloads.

## Sources

- Paper: *Heartbeat Thinking: Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents* — Zhang et al., Ant International, 2026 — [arXiv:2610.26233](https://arxiv.org/abs/2610.26233).

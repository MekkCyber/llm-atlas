# HABIT-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Benchmark testing whether an LLM agent's memory system can **induce user habits from scattered, incomplete evidence** across hundreds of past sessions, and apply them correctly to a new request. Four-choice tasks across four domains (food, finance, software, travel), with distractors for temporary exceptions, changed habits, and rejected suggestions. Best of 8 memory methods tops out at **33.6% macro accuracy**; even supplying ground-truth evidence directly barely helps — the bottleneck is reasoning over the evidence, not retrieving it.

**Prereqs:** [../agents/_memory.md](../agents/_memory.md)
**Related:** [../agents/_memory.md](../agents/_memory.md) · [../agents/memory-tax.md](../agents/memory-tax.md)

---

## What it is

Agent-memory benchmarks have mostly tested *fact recall* ("what did the user say about X in session 42?"). HABIT-Bench tests *habit induction*: inferring the user's recurring preferences from weak, repeated evidence spread across many sessions, then correctly applying the inferred habit — or correctly *not* applying it (temporary exception, changed habit, suggested-but-rejected pattern).

Synthetic user histories built from task-oriented dialogue data across four domains: **food, finance, software, travel**. Sampled tasks are independently reviewed by humans to confirm the ground truth. Evaluated with Qwen3-235B-A22B and GLM-5.3-Flash as the agent models.

## How it works

Each task:

- **Longitudinal history**: hundreds of past user sessions. Some repeated patterns are genuine habits; some are noise; some were rejected by the user when previously suggested.
- **Current request**: a user turn where the right response depends on the inferred habit (apply it, override it, or ignore it).
- **Four choices**: typically {habit-aligned, habit-ignoring, over-applied habit (ignoring the exception), rejected-pattern application}. The distractors probe specific failure modes.

Scoring is macro-accuracy across four task types:

1. **Apply habit** — the usual case.
2. **Temporary exception** — habit applies in general but not today.
3. **Changed habit** — recent sessions show the habit has shifted.
4. **Rejected suggestion** — the pattern appeared because *the system* suggested it; the user never confirmed it.

## Why it matters

- **Memory ≠ habit induction.** All 8 tested memory methods — RAG over history, hierarchical summaries, graph memory, learned recall — perform similarly poorly. **33.6% macro accuracy** is close to the 25% random baseline on 4-choice tasks.
- **Not a retrieval problem.** When the ground-truth evidence is handed to the agent directly, performance stays limited. The bottleneck is *reasoning over weak repeated evidence*, not retrieving it.
- **Pairs with [memory-tax.md](../agents/memory-tax.md).** Memory Tax shows current systems can be worse than nothing; HABIT-Bench shows that even perfect retrieval doesn't unlock habit-level reasoning. Together: agent memory is both over-sold and bottlenecked at the reasoning layer.
- **Realistic distractors.** Temporary exceptions and rejected suggestions mirror actual personal-assistant failure modes.

## Gotchas & tricks

- **Synthetic histories.** Real user histories have messier signal. HABIT-Bench's synthetic structure makes evaluation possible but may understate noise.
- **Backbone matters.** Scores differ meaningfully between Qwen3-235B-A22B and GLM-5.3-Flash; report both when citing.
- **Open-book baseline.** Running the agent with the full history in-context (no memory abstraction) is informative — HABIT-Bench reports this and it still doesn't close the gap, which strengthens the "reasoning is the bottleneck" conclusion.
- **Not a benchmark where RAG-over-history wins by default.** Unlike many agent-memory evals, retrieval alone isn't enough here — see the "Pairs with" note above.

## Sources

- Paper: *HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory* — Zhang, Wei, Cheng, Sun, Wang, Cheng, Wang, Wu, Chen, Lin, 2026 — [arXiv:2610.24520](https://arxiv.org/abs/2610.24520)

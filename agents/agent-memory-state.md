# Agent memory with relational + reasoning state
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-running personal agents accumulate thousands of facts over months; naïve retrieval breaks because facts become stale, interdependent, and context-dependent. EvoMemState (2026) separates stored facts from two *state* layers: a **Relational State** that tracks validity and inter-fact relations, and an **RL-trained Reasoning State** that decides which facts actually apply to the current request.

**Prereqs:** [grpo](../post-training/grpo.md)
**Related:** [rlvr](../post-training/rlvr.md), [ppo](../post-training/ppo.md)

---

## What it is

Standard agent memory is a set of (fact, timestamp, embedding) tuples with vector retrieval. This breaks on three patterns: (1) facts contradict each other after an update ("user lives in NYC" → "user lives in SF"); (2) facts have implicit relations the retriever cannot express ("the user's dog" is linked to "the user's new apartment is pet-friendly"); (3) retrieval returns too much — many stored facts are *irrelevant* to the current intent even though they match on embedding.

The paper augments the raw fact store with two learned layers that handle these patterns.

## How it works

**Relational State** — for each stored fact, maintain a small structured record: validity (current / superseded / contested), timestamps for first/last seen, and typed edges to related facts. New facts can invalidate old ones; edits propagate along edges. This is *not* a learned component — it's a bookkeeping layer that lets retrieval reason about freshness and conflicts.

**Reasoning State** — a learned policy trained with RL. At each request, it takes (request, candidate facts from retrieval, relational-state metadata) and outputs the subset of facts that should actually enter the LLM's context. Reward comes from downstream task success (user got the right answer / completed the action correctly).

Training the reasoning state is set up as a short-horizon RL problem: a trajectory is (retrieve, select, answer, score), and the policy is updated with grouped baselines over N candidate selections per request (similar in flavour to GRPO).

## Why it matters

Personal agents have been bottlenecked by either (a) oversized context windows and hallucinated retrieval, or (b) hand-coded memory heuristics. A learned usage policy on top of structured memory is a principled replacement and the paper reports both retrieval-completeness and answer-accuracy gains over baseline memory stacks (standard retrieval, MemGPT-style).

## Gotchas & tricks

- **Relational-edge noise.** Automatically extracted edges are noisy; the paper uses a confidence threshold and discards low-confidence edges rather than acting on them.
- **Reward hacking.** Downstream task reward is sparse and noisy. The paper adds a shaping term that penalises selecting too many or too few facts.
- **Memory growth.** Relational-state bookkeeping is more expensive per fact than vector embedding. Compact "cold" facts periodically (merge, dedup, summarise).

## Sources

- Paper: EvoMemState: Evolving Memory State for Personal Agents — Xiang et al., 2026 — [arXiv:2611.03743](https://arxiv.org/abs/2611.03743)

# Referential dangling
*Depth — a paradigm-level failure of hard prompt compression on multi-hop reference chains.*

**TL;DR:** Hard prompt compression scores tokens / sentences / chunks *independently* and keeps the top under a budget. When a bridge entity is defined in chunk A and used in chunk B, independent scoring can keep A but drop B (or vice versa), leaving the answer path with a dangling reference. On three multi-hop QA benchmarks at compression ratio 0.30, Beaver — a Qwen3-0.6B-embedding chunk ranker — leaves the answer path incomplete in **34–54% of bridge examples**. This isn't a bug in any one compressor; it's structural to independent-scoring compression.

**Prereqs:** [prompt-compression.md](prompt-compression.md), [README.md](README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Hard-compression scorers evaluate each unit in isolation. The score $s(u_i \mid q)$ ignores whether $u_i$ is *referentially dependent* on any $u_j$. When the answer to query $q$ requires resolving a chain "entity defined in $u_j$, used in $u_i$," dropping either breaks the chain — but the scorer has no visibility into the dependence.

"Referential dangling" is the paper's term for the resulting failure mode: the compressed prompt contains the answer *span* but not the *reference* it depends on.

## How it works

The failure pattern:

```
Full prompt:
  u_j:  "Marie Curie was born in Warsaw in 1867."     (defines entity)
  u_i:  "She won the Nobel Prize in Physics in 1903."  (uses entity)
  Query: "In what year did the person born in Warsaw in 1867 win the Nobel Prize in Physics?"

Independent scoring under query q:
  u_i scored high — contains "Nobel Prize" + "1903"
  u_j scored low — doesn't mention "Nobel" or "1903"

At ratio 0.30, keeper picks u_i. Compressed prompt keeps:
  "She won the Nobel Prize in Physics in 1903."

Downstream LLM: cannot resolve "She" ⇒ dangling reference.
```

The measurement recipe:
1. Take a multi-hop QA dataset (HotpotQA / MuSiQue / 2WikiMQA all fit).
2. For each bridge example (2+ hops), identify the *bridge sentence* that defines the entity used in the answer sentence.
3. Compress at target ratio.
4. Flag as *dangling* any example where the answer sentence is kept but the bridge sentence is dropped.

Reported dangling rates for Beaver at $\rho = 0.30$: 34% to 54% across three benchmarks.

## Why it matters

- **Structural, not fixable per-compressor.** Every independent-scoring compressor has this failure; better scorers don't remove it.
- **Silently craters multi-hop QA quality.** Downstream metrics drop without an obvious cause because the prompt looks fine — the referent is just missing.
- **Motivates dependency-aware compression** as the next paradigm — score pairs / graphs of units, not units in isolation.
- **Reproducible measurement.** The paper's bridge-example labeling gives a concrete unit test for future compressors.

## Gotchas & tricks

- **Rate is compression-ratio-sensitive.** Higher $\rho$ (more units kept) reduces dangling but also reduces the compression benefit.
- **Coreference resolution as a preprocessing step** helps but doesn't eliminate — new references arise at inference time.
- **Not the same as "the compressor missed the answer."** Referential dangling *keeps* the answer chunk; the failure is upstream in the reference chain.
- **Applies to soft compression too, in principle.** A soft summarizer that summarizes each chunk independently and concatenates has the same problem, just less measurable.

## Sources

- Paper: *Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression* — Hu, Li, Fu, Zou et al., 2026 — arXiv:2608.04569.
- Baseline compressor: Beaver (chunk ranking with Qwen3-0.6B embeddings) — analyzed in the same paper.

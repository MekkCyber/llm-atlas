# Deterministic-Computation-Grounded Verifiable Tool Tasks
*Depth — build tool-use SFT and RL tasks whose ground truth is computable by executable code against real data sources.*

**TL;DR:** The constructor in Axion 1.0 Stage 2 builds tool-use training tasks by taking a real-world data source (API, database, dataset), expressing the ground-truth answer as a **deterministic computation** over that source (SQL, Python, calculator), and generating natural-language prompts that require tools to recover the answer. The task's correctness check is the deterministic computation itself — no human labels, no learned critic, exact rewards. Feeds Stage 2 SFT of [Axion 1.0](../case-studies/axion-1-0.md), where the recipe excludes DPO and distillation.

**Prereqs:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [README.md](README.md) · [../case-studies/axion-1-0.md](../case-studies/axion-1-0.md)

---

## What it is

Tool-use training data is hard to label. A prompt like "what was Acme's Q3 revenue growth?" needs a verifiable ground-truth answer — which only exists if a canonical data source contains the answer *and* if the path through tools to retrieve it is fixed. Hand-authoring such tasks is slow; LLM-generated tool-use tasks tend to have ambiguous or wrong ground truths.

Deterministic-computation grounding reverses the direction: start from the executable computation, then frame the natural-language task around it. Because the answer is computed by code against a real source, correctness is a code check — exact, reproducible, and verifiable at scale.

---

## How it works

Three-step construction, repeated per task:

1. **Pick a data source + computation.** Choose a source (SQL database dump, public dataset, documentation corpus, calculator tool) and write a deterministic computation over it — e.g. a SQL query, a Python snippet over a dataframe, a sequence of API calls. The computation's output is the ground-truth answer.

2. **Author a natural-language prompt.** Write a prompt whose correct answer is exactly the deterministic computation's output. The prompt is phrased so that solving it *requires* tool use (the model can't guess from pretraining knowledge alone).

3. **Define the verifier.** The computation itself is the verifier. At training time, the agent's final answer is checked against the computation's output — string match, numeric tolerance, or set equality depending on type.

Multi-step tasks compose: chain two or more computations, with intermediate results feeding the next tool call. The verifier still checks only the final output.

### Properties

- **Exact.** No judge model, no label noise.
- **Reproducible.** Re-running the computation gives the same ground truth; no drift.
- **Scalable.** New tasks cost as much as writing a new query or script; no human-in-the-loop.
- **Compatible with RL.** The verifier returns a scalar reward — plug straight into [RLVR](../post-training/rlvr.md).

---

## Why it matters

- **Removes human labeling from tool-use training.** The bottleneck becomes writing computations, not writing labels.
- **Makes tool-use RL possible at scale.** Without verifiable rewards, long-horizon tool agents train on fragile judge-LLM signals. Deterministic verifiers eliminate that failure mode.
- **Aligns with real deployment.** Production tool agents hit exactly this pattern: an underlying data source, a code path to the answer, a natural-language user query. Training on this shape is a transfer-friendly signal.
- **Composable with trace mining.** Pipelines that mine real agent-tool traces to discover tool dependencies pair with this one: trace mining gives the environment shape, deterministic-computation grounding supplies the terminal-state verifier that turns a dependency subgraph into a trainable task.

---

## Gotchas & tricks

- **Prompt phrasing dictates whether tools are actually needed.** If the prompt leaks the intermediate values, the model short-circuits to pattern matching. Blind the prompt to specific numbers and dates.
- **Deterministic ≠ trivial.** Over-constrained computations (closed-form answers) lead to tasks where a strong model can guess without tools. Favor computations whose intermediate steps touch private data.
- **Source freshness matters.** Data sources that drift (live APIs, external databases) change ground truth over time. Snapshot the source at task-creation time or pin it to a versioned release.
- **Verifier type must match answer type.** Numeric tolerance for floats, canonical order for sets, normalized strings for text. A too-strict verifier false-negatives; a too-loose one false-positives.
- **Not a replacement for user-intent evaluation.** The pipeline tests "did the model recover the computed answer" — not "did the model satisfy the user's intent in a friendly way." Pair with human-authored eval sets for the latter.
- **Distribution shift from real users.** Computations are easier to author than realistic user queries. The resulting task distribution skews toward well-formed questions. Add realism-targeted augmentation (ambiguity, follow-ups, incomplete prompts).

---

## Sources

- Paper: *Axion 1.0: Open Agent Foundation Models, Simplified* — 2026, arXiv 2611.03233 — primary source. Stage 2 data construction, §3.
- Related: [../post-training/rlvr.md](../post-training/rlvr.md) — the RLVR pattern this fits inside.

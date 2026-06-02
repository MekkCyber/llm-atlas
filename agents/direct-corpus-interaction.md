# Direct Corpus Interaction (DCI)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A retrieval paradigm in which a search agent interacts with the raw corpus through shell commands (`grep`, `awk`, file reads) instead of querying a precomputed embedding index. The corpus *is* the environment; the agent reasons about filenames, line numbers, and pattern matches directly. Trained with a two-stage pipeline — verified-trajectory SFT cold start followed by GRPO RL — DCI agents reach SOTA token-level F₁ on open-domain QA while sidestepping the encoding loss of dense retrievers.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [_harness](_harness.md) · [../safety/persistent-control-attack.md](../safety/persistent-control-attack.md)

---

## What it is

Conventional retrieval-augmented generation (RAG) couples the LLM to a *retriever*: a separate component that takes a query, produces a sparse or dense representation, and returns ranked documents. The retriever is opaque to the LLM — the LLM can re-issue queries, but it cannot inspect the index.

Direct Corpus Interaction inverts the relationship. The retriever is replaced by the corpus itself, exposed through a shell-like interface (`grep`, `awk`, line-bounded `cat`, etc.). The agent issues commands, observes output, and iterates. Evidence composition (which lines, which files, in what order) becomes part of the agent's policy.

DCI is positioned as **complementary** to dense retrieval — it wins on queries where surface form matters (codebases, identifiers, named entities) and loses on heavy-paraphrase queries where embeddings shine.

---

## How it works

### Two-stage training

**Stage 1 — verified cold start.** A *Tutor* (answer-aware) and *Planner* (answer-blind) jointly generate candidate trajectories of shell commands that solve QA examples. Trajectories where the final answer is correct *and* the reasoning is causally grounded (the executed commands actually produced the evidence cited) become the SFT corpus. This avoids the unstable cold-start regime where an untrained policy issues bad commands that get zero reward.

**Stage 2 — GRPO RL.** The cold-started policy is refined with [GRPO](../post-training/grpo.md): for each query, sample G trajectories from the current policy, execute each against the corpus, score by final-answer correctness, and update with the group-relative advantage. Reward is verifiable (correct vs incorrect final answer), so this is an instance of [RLVR](../post-training/rlvr.md).

### Sharded-parallel execution

Each shell command can take seconds on a large corpus. To make multi-rollout RL practical, the executor shards the corpus and runs the command in parallel across shards. A consistency guarantee ensures **byte-exact equivalence** with sequential execution — needed because the policy's next command may depend on exact byte offsets. Up to 7.6× speedup over sequential.

### Action space

Actions are full shell-command strings. The vocabulary is the LLM's normal tokenizer; no special tool-call schema. This keeps DCI compatible with any agent harness that supports text-tool actions.

---

## Why it matters

- **Removes the encoding bottleneck.** Dense retrievers compress documents into a few hundred dimensions before the LLM ever sees them. Anything the encoder discarded is gone. DCI's information loss is bounded by what `grep` can express, which is usually less than what an encoder discards.
- **Composable with dense retrieval.** Run DCI and dense in parallel; merge candidate evidence. Each catches what the other misses (lexical vs semantic).
- **Stable RL recipe for shell-tool agents.** The Tutor/Planner cold-start trick generalizes — any tool-using agent with a large jagged action space can use the same pattern to bootstrap GRPO RL.

---

## Gotchas & tricks

- **Lexical ceiling.** Queries with substantial paraphrase or implicit reference defeat `grep`-style retrieval. Pair with a dense retriever as a fallback channel, not as a sole interface.
- **Trajectory length blow-up.** Sloppy `grep` patterns can return megabytes; the policy must learn to constrain output (use `-l`, `head -n`, line ranges) or context budget collapses.
- **Sharded executor must preserve order semantics.** A naive shard-and-concat breaks commands that depend on cross-document order. The paper's sharded executor is engineered for byte-exact sequential equivalence — don't substitute a generic map-reduce.
- **Reward hacking risk.** If the verifier scores token-level F₁, the policy can learn to dump suspicious-looking strings into its answer to chase F₁. Use exact-match alongside F₁ during training.

---

## Sources

- Paper: *GrepSeek: Training Search Agents for Direct Corpus Interaction* — Salemi, Zeng, Nijasure, Chung, Rahimi, Diaz, Zamani, 2026 — UMass Amherst / Princeton / CMU. Two-stage SFT-then-GRPO recipe and the sharded executor.

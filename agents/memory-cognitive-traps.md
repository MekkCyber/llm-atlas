# Memory-Induced Cognitive Traps (MemTrapBench + AdaptiveMem)
*Depth — retrieved memories can distort LLM reasoning on the current task even when the memory is faithfully stored and semantically relevant.*

**TL;DR:** Standard memory benchmarks ask whether the right memory was stored and retrieved. **MemTrapBench** asks a different question: given that retrieval was correct, does the memory *hurt* the current task? Two failure modes emerge: **Reasoning Fixation** (memory locks the model into a stale reasoning path) and **Belief Distortion** (retrieved memory shifts the model's answer despite being irrelevant to the current query). Across two model families and five representative memory frameworks, **every** memory strategy underperforms the no-memory baseline; even the strongest lose **>10 percentage points**. The paper proposes **AdaptiveMem**, an inference-time prompt-side intervention that instructs the LLM to explicitly watch for and discount trap patterns — recovering most of the loss without regressing standard memory benchmarks.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A benchmark and mitigation for a specific class of memory-usage bugs:

- **Reasoning Fixation.** The model, upon seeing a retrieved memory, latches onto the reasoning trajectory implied by the memory even when the current problem is a superficially-similar-but-substantively-different variant. It reasons as if solving the remembered problem.
- **Belief Distortion.** A semantically-related but *irrelevant* memory shifts the model's beliefs or priors — e.g. a stored fact about a related entity biases the answer for a different entity in the current query.

MemTrapBench includes controlled test cases where a "correct + relevant" memory is provably harmful.

## How it works

**Evaluation.** For each item, an oracle memory (faithful, semantically related) is provided to the memory framework; the model is asked to answer with and without memory. Delta accuracy quantifies trap severity. Frameworks tested include representative retrieval, summary, episodic, and hybrid memory stacks across two frontier model families.

**AdaptiveMem.** A lightweight inference-time prompt that:

1. Directs the model to *first* check whether the retrieved memory's structure matches the current problem.
2. Explicitly instructs the model to treat memory as evidence, not as answer.
3. Adds a self-check step for reasoning fixation ("would you solve this the same way absent the memory?").

No retrieval-side change, no fine-tuning — pure prompt-level mitigation, so it composes with existing memory frameworks.

## Why it matters

Persistent memory (long-term stores, episodic recall, agent scratchpads) is now default agent infrastructure. If faithful, relevant memory can *degrade* current-task performance, memory eval must measure the *interaction* with reasoning, not just retrieval quality. MemTrapBench stakes that ground and shows the trap effect is universal across current memory frameworks — not a bug in any one implementation.

## Gotchas & tricks

- Trap severity is model-dependent — larger / more capable models don't automatically avoid the traps, and can be *more* fixation-prone because they trust the memory more.
- AdaptiveMem is a prompt intervention; it can regress on tasks where the "just use the memory" heuristic is actually correct. The paper shows this trade-off is favorable but domain-specific.
- Benchmark construction requires per-item ground truth about *irrelevance* — expensive to scale beyond the paper's coverage.

## Sources

- Paper: *MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use* — Wang, Luo, Xu et al. (Zhejiang U.), 2026 — [arXiv:2608.20202](https://arxiv.org/abs/2608.20202)

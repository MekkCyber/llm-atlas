# Thought-Level Beam Search (Gambit)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A test-time reasoning algorithm that runs **beam search at the granularity of thoughts** rather than tokens. Periodically prunes unpromising trajectories, immediately branches from high-quality prefixes, and uses a lightweight hidden-state scorer instead of a full verifier call. Keeps GPUs saturated where naïve parallel sampling fragments memory and subtractive pruning starves compute.

**Prereqs:** [mcts](mcts.md), [prm](prm.md), [long-cot-rl](long-cot-rl.md)
**Related:** [length-penalty](length-penalty.md), [orm](orm.md), [_post-training](../_post-training.md)

---

## What it is

Test-time compute for large reasoning models (LRMs) is a **compute allocation problem over partial trajectories** under a fixed hardware budget:

- **Parallel sampling** treats traces independently. Memory bandwidth blows up (KV caches for $N$ traces), and traces waste steps on already-lost branches.
- **Subtractive pruning** kills low-scoring traces early. Once too many die, hardware sits idle — throughput collapses.

Gambit rejects the dichotomy: keep a beam of *thoughts*, periodically prune bad prefixes, and **immediately branch** from surviving high-quality prefixes so the batch stays full.

## How it works

Rollouts are segmented into **thoughts** — natural units of reasoning (sentence, step, whatever the decoder emits between delimiters). At each thought boundary:

1. **Score each active trace** using a lightweight scorer that probes the model's hidden state (not a separate verifier call — cheap enough to run every step).
2. **Prune** the bottom-$k$ traces below a threshold.
3. **Branch** from top-scoring prefixes to fill the freed batch slots, sharing KV cache prefixes where possible.
4. **Continue** decoding to the next thought boundary.

Because branching reuses KV prefixes and refills the batch, hardware utilization stays high throughout the run. Because pruning is aggressive but immediate rebranching is aggressive too, low-value trajectories die fast and high-value ones deepen fast.

The hidden-state scorer is a small MLP over intermediate residual features — it plays a PRM-like role but avoids a second model forward pass per step. Training the scorer is optional; the paper uses a lightweight version trained on model rollouts.

## Why it matters

Under identical hardware constraints on reasoning benchmarks:

- **+6.7% absolute** on HMMT-24 over pruning baselines.
- **+3.3%** on AIME-25.
- **>2× higher throughput** on trace completion.
- **-68.5% total tokens** vs. standard parallel sampling.

More generally: the test-time-compute conversation has been shifting from "how much compute" to "where to allocate it". Gambit gives a concrete answer that respects real hardware constraints (memory, batch size) rather than treating trace count as a free variable, and slots into the inference stack alongside speculative decoding.

## Gotchas & tricks

- **Thought boundaries matter.** Coarse boundaries (paragraph) waste pruning opportunities; too-fine boundaries (per-token) burn scorer calls. The paper uses model-emitted delimiters as the natural unit.
- **Scorer needs to be cheap.** A full verifier per step is the trap — it destroys the throughput win. A single-MLP hidden-state probe is the right budget.
- **KV-cache-aware branching is required.** If branches don't share prefixes, memory blows up and the beam collapses to naïve parallel sampling.
- **Pruning threshold interacts with beam width.** Aggressive pruning + narrow beam = greedy under a mask; loose pruning + wide beam = expensive parallel sampling. There's a sweet spot per benchmark.
- **Complementary to speculative decoding, not competitive.** Speculative decoding accelerates each step; Gambit chooses which trajectories to spend those steps on.

## Sources

- Paper: *Thought-Level Beam Search for Reasoning* — Lijie Yang, Hongyin Luo, Jiawei Zhao, Tri Dao, Ravi Netravali (Princeton / MIT CSAIL / Meta AI), 2026 — [arXiv:2608.08020](https://arxiv.org/abs/2608.08020).

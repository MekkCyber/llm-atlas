# Visual KV pruning (TRACE)
*Depth — a training-free, cache-reuse-safe policy for admitting and evicting visual tokens in a GUI-agent's growing KV cache.*

**TL;DR:** GUI agents accumulate high-resolution screenshots as the trajectory unfolds; the visual KV cache balloons and inference latency and memory suffer. Naive Top-K token pruning either loses coverage of operable regions or forces re-encoding when a discarded region matters later. TRACE ranks visual evidence by a combination of **layout-derived interaction priors**, **instruction relevance**, and **feature novelty**; reserves part of the budget for **native-resolution tokens spread across the screen**; and produces a **nested order** so retained evidence shrinks monotonically across budgets and stays reusable throughout the trajectory. Retired frames contract into compact session state without re-encoding.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** *(none in-graph yet)*

---

## What it is

A GUI agent takes a screenshot each step, encodes it into visual tokens, and appends them to the KV cache. Cache size grows linearly with the number of steps, hitting three walls:

1. Attention cost per new token scales with the total context length.
2. GPU memory caps concurrent trajectories.
3. Re-encoding a discarded region costs the entire vision-encoder forward pass.

Standard visual-token pruners (image-side Top-K over the current frame) either (a) discard tokens with no way to bring them back if a later step needs them, or (b) re-encode from scratch. TRACE targets the **irreversible admission** problem: once a token is dropped from KV, its visual evidence is gone, so admission decisions must remain useful across the unknown-future rest of the trajectory.

## How it works

### Three ranking signals per token

- **Layout-derived interaction prior.** Query-independent: which regions of the screen carry high a-priori interaction density (buttons, form fields, headers)? Derived from layout analysis, not per-instruction.
- **Instruction relevance.** Standard query-token dot-product against the current instruction.
- **Feature novelty.** How different is this token's feature from tokens already retained? Rewards tokens that add information beyond what the cache already has.

The three signals combine into a single per-token score.

### Spatial-coverage reserve

Reserve part of the budget for **native-resolution tokens distributed across the screen** — a sparse grid that prevents any region from dropping out entirely. Without this reserve, the ranker can concentrate the budget in one dense area (e.g. a form) and leave the rest of the screen unrepresented; the reserve pays a small quality cost to keep global spatial coverage.

### Nested ordering — monotone contraction

Sort all tokens by the combined score, but preserve a **nested** structure: the top-$b$ set is a subset of the top-$b'$ set for every $b' > b$. This means shrinking the budget doesn't require re-scoring; the ranker simply keeps a prefix of the same order. Concretely, retained visual evidence **shrinks monotonically across budgets**, and cross-step reuse of the KV cache remains valid — dropping the last token of a nested sequence is safe; dropping an arbitrary token in the middle would break the cache invariant.

### Monotone KV contraction for retired frames

Old frames retire when the trajectory advances. TRACE's contraction pass folds the retired frame's KV into **compact session state** — a small residual summary — instead of dropping it entirely or re-encoding on demand. Because contraction is monotone (it can only remove information, not add), it never needs a re-encoding trip.

## Why it matters

- **Training-free.** No model changes, no fine-tuning; ships as an inference-time policy on top of an existing GUI-agent VLM stack.
- **Trajectory-safe pruning.** The nested-order invariant is the key insight: it turns pruning from an irreversible one-shot admission into a reversible budget knob. The same ordering serves any budget, cache reuse survives, and later frames don't need to know what earlier frames dropped.
- **Coverage-and-relevance combined.** Layout priors + relevance + novelty covers the failure modes of single-signal rankers (relevance alone concentrates; novelty alone drifts; layout alone ignores the instruction).
- **Fits computer-use workloads.** GUI agents are the dominant "long-visual-trajectory" workload; TRACE targets their bottleneck directly.

## Gotchas & tricks

- **Layout prior needs a real layout parser.** Feeding pixels directly to compute the prior misses the point — the prior should encode UI structure, not just visual salience.
- **Novelty ties can drift.** Two tokens with identical scores can end up in different positions across re-scorings; pin ties by a stable secondary key (spatial index) to preserve the nested invariant.
- **Session state summary is a lossy compression.** If a retired frame's information is genuinely needed later, contraction can't fully recover it. For agents that revisit past screens, keep those frames in a rehydration cache.
- **Compatible with, not a replacement for, KV-storage compression.** Compose with [MLA](../architectures/mla.md)-style KV latents or GQA — TRACE selects *what* to keep, storage compression shrinks *how much* each retained token costs.

## Sources

- Paper: *TRACE: Trajectory-robust Admission with Evidence Ordering for Efficient GUI Agents* — Wang, Qiao, Zhang, Zhuge, Zhang, Lu, 2026 — [arXiv:2609.10297](https://arxiv.org/abs/2609.10297).

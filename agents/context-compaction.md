# Context Compaction with Behavior-Equivalence RL (RECAP)
*Depth — an RL-trained summarizer for long-horizon agents, rewarded for preserving the behavior the full history would produce.*

**TL;DR:** Long-horizon LLM agents blow their context budget on stale history. Sliding windows and naive summaries lose crucial state. RECAP trains a compactor with an RL objective where a *good* summary is one that preserves the behavior the original history would produce — a behavior-distortion loss plus the task reward. Beats task-reward-only RL baselines by +4.6% on BrowseComp-Plus and +4.3% on SWE-bench Verified, with Qwen3.5-4B.

**Prereqs:** [rlvr.md](../post-training/rlvr.md), [grpo.md](../post-training/grpo.md)
**Related:** [long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [agents/README.md](README.md)

---

## What it is

Agents accumulate history: tool calls, observations, intermediate reasoning. At some point it won't fit in the context window and has to be summarized. Any lossy summary changes the next-action distribution. RECAP frames the problem as *keep the action distribution*: a summary `C(h)` is good iff `π(· | C(h)) ≈ π(· | h)` on the relevant subsequent actions. That equivalence is the training signal.

## How it works

Two RL training signals work in parallel on the compactor `C`:

1. **Task reward.** Standard: compacted-agent completes the task (SWE-bench, BrowseComp-Plus) → +1. Standard GRPO-style update.
2. **Behavior-distortion reward.** Sample action distributions from the full-history policy `π(· | h)` and the compacted-history policy `π(· | C(h))`. Penalize the agent for divergence (KL or a top-k agreement measure) on the next k actions.

Both rewards feed into the same policy-gradient update. Ablations show that supervising both summary *generation* (which tokens the compactor emits) and post-compaction *behavior* (how the agent acts on the summary) is what matters — either alone underperforms.

```
Full history h ──► Full-history policy ──► action distribution π_full
                                                        ▲
                                                        │  KL divergence
Full history h ──► Compactor C ──► summary C(h) ──► Compacted-history policy ──► action distribution π_comp
                                                        │
                                                        └── task reward from rollout
```

## Why it matters

- **Right objective for long-horizon agents.** Previous compaction methods optimize surrogates (summary ROUGE, hand-chosen features). Behavior equivalence is the *actual* downstream objective; optimizing it directly is a cleaner approach.
- **Scales to real benchmarks.** +4.6% on BrowseComp-Plus, +4.3% on SWE-bench Verified with Qwen3.5-4B. Not toy numbers — those benchmarks are where agent research is centered.
- **Composable with any base policy.** The compactor is a separate component; the base agent isn't retrained. Drop into an existing agent stack.
- **Fits the trend.** As context lengths plateau and agent trajectories grow, compaction-not-expansion is where the serving cost goes. RECAP is the first well-scaled instance.

## Gotchas & tricks

- **Need the full-history policy at training time.** You can only compute `π(· | h)` if you have a model that fits `h` in its context. The paper uses a longer-context teacher during training; the compactor and the short-context student are trained together.
- **Behavior divergence metric matters.** KL on action logits is stable; top-k agreement is noisier. The paper tries both; KL wins.
- **Compactor granularity.** Compact per-tool-call vs. per-k-tokens vs. per-episode. The paper compacts episodically; too-fine compactions thrash.
- **Task reward dominates if too hot.** If the task reward weight is large, the compactor learns to drop everything and let the agent guess. Keep behavior-distortion weight competitive.
- **Not a replacement for long context.** When all history fits, use it. Compaction is specifically for when it doesn't.
- **Transfer across tasks.** A compactor trained on SWE-bench-like traces transfers partially to browsing traces; the task-reward term is the leak. Train compactors per task family where the trajectory structure differs.

## Sources

- Paper: *RECAP: Training Long-horizon LLM Agents with Context Compaction* — correspondence: EPFL and ETH Zürich, 2026 — [arXiv:2611.31258](https://arxiv.org/abs/2611.31258).

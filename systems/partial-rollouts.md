# Partial Rollouts (Long-Context RL Infrastructure)
*Depth — Kimi k1.5's system-level trick for efficient long-context RL.*

**TL;DR:** In long-CoT RL at 128k context, a few very long trajectories monopolize each rollout step — all other rollouts block waiting. **Partial rollouts** cap each iteration's output budget; trajectories that hit the cap are **saved to a replay buffer and continued in the next iteration**. Rollout workers run asynchronously — fresh prompts start while long trajectories continue from the buffer. Responses assemble from segments across iterations. **Only the final segment is on-policy**; earlier segments are masked out of loss. Plus repeat detection → early termination → repetition penalty. Not a paper benchmark gain (no numeric speedup disclosed) but a named system-level contribution of Kimi k1.5.

**Prereqs:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md), [online-policy-mirror-descent](../post-training/reasoning/online-policy-mirror-descent.md)
**Related:** [dualpipe](dualpipe.md) · [ray](ray.md) · [kimi-k1-5 case study](../case-studies/kimi-k1-5.md)

---

## What it is

A rollout infrastructure for long-context RL that decouples "one trajectory's length" from "one iteration's duration." Introduced in Kimi k1.5 (Moonshot AI, 2025, Sec. 2.6.2) as part of the system support for 128k-context RL training.

The problem it solves: under pure synchronous-rollout RL (the default for PPO, GRPO, long-CoT RL), a rollout step ends when the longest trajectory in the batch finishes. With context length pushed to 128k and long-CoT policies emitting 20k+-token responses, **one trajectory's tail keeps the whole GPU fleet idle**. This is a long-tail latency problem: median rollout is short, p99 is very long, and the step duration is gated by p99.

Partial rollouts convert the blocking pattern into a streaming-continuation pattern. Each iteration covers *up to `T` tokens per worker* regardless of where that lands inside any given trajectory. The next iteration continues from wherever the last one stopped.

---

## How it works

### The mechanism

```
per_iteration_token_budget = T              # hard cap per iteration
replay_buffer = {}                          # (prompt, prefix) → state

for iteration i = 0, 1, 2, ...:
    for rollout_worker in workers:          # run asynchronously
        if replay_buffer has a trajectory to continue:
            (x, prefix) = pop from replay_buffer
            continue generating from prefix under π_{θ_i} up to T tokens more
        else:
            sample new prompt x
            generate from π_{θ_i} up to T tokens

        if generation hit EOS or correctness reward → done:
            trajectory is complete; pass to trainer
        else:
            save (x, current prefix) to replay_buffer

    # trainer collects all completed trajectories this iteration
    # runs gradient updates, produces θ_{i+1}
```

### Segment assembly

A complete response may span **`n-m, n-m+1, …, n`** iterations. When a trajectory finishes at iteration `n`, its segments are concatenated:

- Segment `n-m` was generated under policy `π_{θ_{n-m}}`.
- Segment `n-m+1` was generated under `π_{θ_{n-m+1}}`.
- …
- Segment `n` (the final one) was generated under the current policy `π_{θ_n}`.

**Only the final segment is on-policy** under the reference `π_{θ_n}`; all earlier segments come from stale references.

### Iteration scheduling — mixing prompts and closing groups

The per-iteration token budget is **per-worker**, not per-prompt or per-iteration-total. Within one iteration, each async worker processes one or more trajectories up to `T` tokens of output; a worker that finishes a short trajectory picks up another from the queue (a fresh prompt or a buffer resumption). So **one iteration mixes many prompts** across workers, at different stages of completion.

What the trainer consumes per iteration is **group-complete prompts** — those for which all `k` rollouts have hit EOS (or correctness termination). This matters because the mirror-descent baseline `r̄ = mean(r_1, …, r_k)` and the group-relative length reward both need all `k` rollouts' rewards and lengths. A prompt with 7 of 8 rollouts done cannot be consumed yet — the group isn't closed.

```
prompt x → k = 8 rollouts distributed across async workers
  ├─ rollouts 1–7 finish in iteration i       → sit in buffer with rewards computed
  └─ rollout 8 keeps running                   → saved to replay buffer at end of i,
                                                 continues in i+1, i+2, … finishes in i+5

At iteration i+5:
  all 8 rollouts of x complete
  → compute r̄, advantages, length reward
  → prompt x contributes to the gradient step at iteration i+5 only
```

Prompt `x` contributes nothing to iterations `i` through `i+4` — it just waits. **The trainer never starves** because many prompts are in flight simultaneously; while `x` waits, prompts `y, z, w, …` close their groups at their own pace and feed each iteration's gradient step.

The subtle cost is **reference drift across the group**: when `x`'s group closes at iteration `i+5`, rollouts 1–7 were generated under the much-staler reference `π_{θ_i}` while rollout 8's final segment is under `π_{θ_{i+5}}`. The rewards still compose into `r̄`, but the stale tokens are **masked out of the gradient** per the rule below — only on-policy segments receive gradient, even though all segments contribute their outcome reward to the baseline.

**Per-iteration throughput** is bounded by `worker_count × T`, independent of any individual prompt's completion. **Per-prompt latency** is bounded by the slowest rollout in its group. Partial rollouts decouple these — the two metrics that synchronous rollouts were forced to couple.

**What the paper leaves ambiguous:**
- Whether the trainer strictly requires all `k` rollouts or allows degenerate groups (`k' < k`) with a rescaled baseline.
- How aggressive the stale-segment masking is — how many iterations of reference drift before a segment is masked.
- Whether there's a hard timeout on individual trajectories; at some point a runaway rollout may need to be force-terminated and its group closed with whatever has finished.

These are implementation choices the paper does not commit to.

### Staleness handling: loss masking

The paper (Sec. 2.6.2) says: *"certain segments can be excluded from loss computation."* Implementation-wise, stale segments (from iterations too far back) are **masked out** — their tokens contribute to the response (they're part of the trajectory used for reward computation) but **not to the policy gradient**. Only the on-policy segment's tokens receive gradient.

The paper doesn't specify a cutoff for "too stale." A reasonable rule: mask segments from iterations more than 1–2 back, in keeping with PPO's usual "off-policy by at most a few steps" tolerance.

### Repeat detection → early termination → repetition penalty

Orthogonal to the core mechanism but ships with the same system (Sec. 2.6.2):
- The rollout worker monitors generated content for **repeated sub-sequences**.
- If a repeat is detected, generation is **terminated early** — no point burning tokens on a degenerate loop.
- Detected repetitions can carry **an additional penalty** in the reward — discouraging degenerate policies.

This is a degeneracy-guard, not an optimization signal. It's specific to long-context RL where repeat-loops are common failure modes (the model finds a local optimum that says "keep repeating the same step, it maximizes verifier acceptance probability"). Without this detection, such loops can monopolize entire rollout budgets.

### Reported efficiency

The paper does **not** give a standalone speedup number for partial rollouts. The system-level outcome is expressed through the hybrid train↔inference transition cost (Sec. 2.6.3): **<1 minute from training to inference phase, ~10 seconds back to training**. Good numbers for a 128k-context setup, but credit is shared across partial rollouts + the hybrid Megatron↔vLLM deployment + the Mooncake RDMA transfer.

---

## Why it matters

- **Converts p99 latency into a streaming problem.** Long-tail rollouts no longer block short rollouts. GPU utilization stays flat instead of collapsing to zero during long-trajectory tails.
- **Enables 128k-context RL.** Without partial rollouts, the wall-clock cost of long-context RL grows superlinearly (you wait for the longest trajectory every step). With partial rollouts, wall-clock stays close to the median-trajectory cost.
- **Composable with async workers.** Fits naturally on top of a Ray-style async rollout cluster. Each worker makes local progress; global synchronization only at the trainer step.
- **Doesn't require changing the RL algorithm.** Drop-in on top of GRPO, mirror descent, PPO. Just need to handle segment assembly and loss masking.
- **A rare named system-level contribution.** Most RL-for-LLM infrastructure is papers' appendices. Partial rollouts is explicit, reusable, and useful beyond Kimi — worth internalizing as a general long-context RL pattern.

---

## Gotchas & tricks

- **Staleness bias from loss masking.** Masking earlier segments is safe but leaves signal on the table — you're training only on the last-iteration tokens, not the whole trajectory. If trajectories span many iterations, most of the token-budget worth of rollouts contributes zero gradient. There's an engineering trade-off in how aggressively to mask.
- **Replay buffer is mutable state.** The replay buffer holds in-flight prefixes. Crashes / worker restarts can orphan trajectories; recovery logic must handle this.
- **EOS detection per segment.** If a trajectory ends inside an iteration, that segment is shorter than `T` tokens. The scheduler needs to decide: fill the remaining budget with a new prompt, or batch differently. Straightforward but non-trivial.
- **Reward computation is deferred.** Outcome reward (answer correctness) can only be computed when the *full* response is assembled — which is at the end of the last segment. Partial-rollout systems need to defer reward computation until EOS, which complicates reward-sharing across iterations.
- **Length penalty interacts.** [length-penalty](../post-training/reasoning/length-penalty.md) is computed per-response using `len(i)` — the full response length. Partial rollouts preserve this (length is known once the response completes) but require the length reward to be applied retroactively when the trajectory closes, not per-iteration.
- **Repetition penalty calibration.** Too aggressive → early termination of legitimate repetition (e.g., summing a list). Too permissive → degenerate loops. The paper doesn't specify the detection threshold.
- **Compatibility with speculative decoding / draft models.** Partial rollouts work with vanilla autoregressive generation. Speculative decoding (draft + verify) complicates the "where did this segment come from" bookkeeping. The paper uses vLLM for rollouts; vLLM supports vanilla AR decoding cleanly.
- **GPU memory for the replay buffer.** Each pending prefix's KV cache is state that must live somewhere. If rollout workers use a shared vLLM instance, the cache is on GPU; if workers share across machines, you need to serialize prefixes. Mooncake's RDMA transfer handles the training-to-inference direction; the partial-rollout inter-worker direction is a separate concern.
- **Hard to ablate.** The paper reports partial rollouts' benefit qualitatively. Quantifying the standalone speedup requires a controlled comparison (128k-context RL with vs without partial rollouts); no paper has published that number.

---

## When it's unnecessary

- **Short-context RL (≤ 8k).** The long-tail problem barely exists; synchronous rollouts are fine.
- **Small response-length variance.** If all rollouts finish within 2× of the median, blocking on the longest is cheap.
- **Batch-level rather than worker-level parallelism.** If you can fit many full-length rollouts in one iteration (huge GPU memory, small model), the p99 problem dissolves.

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot 
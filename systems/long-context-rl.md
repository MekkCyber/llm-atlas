# Long-Context RL Execution
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL-step execution stack that lets you train with **million-token contexts** on a fixed GPU budget by separating the shared prompt pass from per-response replay. Evaluate the prompt once *without* autograd; retain only the per-model state that later tokens will need; then replay each short response branch one at a time. Peak memory becomes $O(\text{prompt} + \text{one response})$ instead of $O(\text{prompt} + G \cdot \text{response})$, closing the gap between inference-time context lengths (already at 1M+) and RL post-training (still stuck at 256K on most stacks).

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [partial-rollouts.md](./partial-rollouts.md) · [dualpipe.md](./dualpipe.md)

---

## What it is

Group-relative RL (GRPO and siblings) samples $G$ responses per prompt and runs an autograd-enabled forward+backward over the concatenation. When the prompt is long (agent tool traces, long documents, accumulated observations), the compute of the prompt-side tokens dominates but the *gradients* are only needed for the response-side tokens.

Long-Context RL execution exploits this split. The prompt pass runs detached — no autograd graph, no activations kept for backprop. Only the per-layer state that later tokens will actually read (KV cache, moe expert routing state, etc.) is retained. Response branches are replayed *serially*, each carrying its own small autograd graph.

## How it works

Given a batch of prompts $\{q_j\}$ each with $G$ sampled responses $\{o_{j,i}\}$:

1. **Prompt pass (no grad).** Forward $\pi_\theta(q_j)$ under `torch.no_grad()`. Capture the per-layer state that response tokens read — for a standard decoder, this is the KV cache; for MoE, add the router's routing decisions and expert-selection metadata.
2. **Response replay (one at a time).** For each response $o_{j,i}$: rebuild an autograd context, seed it with the retained state from step 1, run the response forward+backward. Accumulate gradients into $\theta$.
3. **Optimizer step.** After all $G \cdot |\text{batch}|$ replays finish, step the optimizer once.

The peak live autograd graph is now bounded by *one* response's activations plus the retained per-layer state — not by the group size $G$ and not by the full prompt.

## Why it matters

- **Closes the inference↔RL context gap.** Inference systems have crossed the million-token threshold; RL post-training generally hasn't. Long-Context RL execution lets researchers train against the same trajectory lengths deployed agents already see.
- **Fixed GPU budget.** No new hardware. The trade is wall-clock (serial response replays) for peak memory (only one response live).
- **Agent trajectories were the bottleneck.** Multi-tool traces easily exceed 256K when observations, tool outputs, and prior decisions accumulate. Being able to RL against these directly (rather than truncating and hoping length-generalization holds) is the main practical win.
- **Composes with GRPO.** Nothing about the technique is GRPO-specific — the shape (one shared prompt, $G$ short responses) is what the trick exploits.

## Gotchas & tricks

- **Retained state must be enumerated per architecture.** For MoE, forgetting the router's routing state produces silently-wrong replays: the response reads a different set of experts than it did at sampling time.
- **Wall-clock cost is real.** Serial replay adds latency proportional to $G$. When $G$ is small (2–4) the trade is favorable; when $G$ is very large (64+) other partitioning schemes may win.
- **Determinism required across the split.** The prompt pass in eval-mode and the response pass in train-mode must produce identical intermediate states. Dropout, stochastic MoE routing, and non-deterministic attention kernels all violate this — pin them.
- **KV state has to be reconstructable, not just cached.** If any layer's forward has side effects that mutate global state (some FlashAttention variants do), the replay diverges. Prefer kernels with pure-functional forwards.
- **Not a replacement for sequence parallelism.** If one response is itself long enough to OOM (rare, but happens in long-CoT RL), stack this with tensor / sequence parallelism on the response pass.

## Sources

- Paper: *LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget* — Zhou et al., Mind Lab, 2026 — introduces the detached-prompt + serial-replay execution stack, instantiated on GRPO.
- See also: [partial-rollouts.md](./partial-rollouts.md) for a related "don't materialize the full trajectory" strategy in a different regime.

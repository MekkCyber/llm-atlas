# Parallel Reasoning (Test-Time Compute)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sample multiple reasoning branches in parallel and aggregate. Test-time-compute scaling for reasoning models; the wall-clock and token cost is the practical bottleneck. **ParaTempo** (Zhang et al., 2026) introduces *temporal confidence* — a branch-local, training-free signal that periodically probes each branch's tentative answer distribution and drives asynchronous control (prune, retire, fork, global stop) from that single signal.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md) · [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md) · [../evaluation/aime.md](../evaluation/aime.md)

---

## What it is

For a reasoning prompt, run `k` independent CoT branches with the same policy and combine — via majority vote (self-consistency), weighted vote by a reward model, or best-of-`k` by a verifier. This is the dominant test-time-compute knob for reasoning. Cost grows linearly in `k · depth`; the operational question is which branches to keep spending on.

## How it works

**Self-consistency baseline:** sample `k` branches to completion, majority-vote on answers. Fixed cost `k × avg_depth`.

**ParaTempo control loop** — training-free, asynchronous:

```
for each active branch b:
    while not done(b):
        continue generating tokens...
        if reached_probe_interval:
            probe answer distribution p_b_t = P(answer | branch state)
            temporal_confidence(b) = concentration(recent probes)
            if temporal_confidence(b) < prune_threshold:
                kill(b)
            elif temporal_confidence(b) > retire_threshold and stable:
                retire(b, keep_answer=argmax(p_b_t))
            # freed compute → fork new branches from divergent states
    if confidence_weighted_vote(alive ∪ retired) > global_stop_threshold:
        return vote_answer()
```

The probe is a lightweight forward call that induces an answer distribution (e.g. by conditioning the model to emit a summary answer, then reading top-token logits). Temporal confidence is the sharpness of recent probes — smoothed across the last few probes rather than instantaneous.

## Why it matters

- **Cuts inference cost without training.** On math/science reasoning, ParaTempo reports 21.8–32.2% latency reduction and 18.1–30.3% total-token reduction vs standard parallel-reasoning baselines at competitive accuracy.
- **No sync required.** All decisions are branch-local, so parallelism efficiency stays high — no barrier or majority-vote step across branches until the global-stop check.
- **Better signal than the alternatives.** Final-answer consensus is *delayed*; per-token confidence is *noisy*; isolated intermediate probes are *unstable*. Temporal confidence (repeated probes, smoothed) is more predictive of future branch convergence than either.

## Gotchas & tricks

- **Probe overhead.** Each probe is one forward pass to induce an answer distribution. Too-frequent probes eat the savings. Typical interval: every N generated tokens (paper-dependent).
- **Fork policy matters.** Naive fork = duplicate a random alive branch; better = fork from a *divergent* state where multiple continuations are plausible, otherwise the fork adds no diversity.
- **Doesn't help if `k=1`.** Parallel reasoning has no lever without multiple branches; ParaTempo assumes a parallel budget to allocate.
- **Applies to any reasoning policy.** Training-free — works on any RL-trained reasoning model (DeepSeek-R1, Qwen-QwQ, o-family).
- **Interacts with length penalty.** A length-penalized model already produces shorter branches — the ParaTempo savings on top are smaller. Length-penalty is train-time; ParaTempo is inference-time; they compose but with diminishing returns.

## Sources

- Paper: *ParaTempo: Efficient Parallel Reasoning via Temporal Confidence* — Zhang et al., SJTU, 2026 — introduces temporal confidence.
- Paper: *Self-Consistency Improves Chain of Thought Reasoning* — Wang et al., 2022 — majority-vote baseline.
- Paper: *Let's Verify Step by Step* — Lightman et al., 2023 — verifier-weighted best-of-`k`.

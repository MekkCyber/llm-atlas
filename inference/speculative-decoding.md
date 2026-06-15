# Speculative Decoding

*Depth — a draft-and-verify decoding family that breaks autoregressive serial cost by predicting several tokens in parallel.*

**TL;DR:** A small, fast **drafter** proposes a sequence of $k$ candidate tokens; the large **verifier** model evaluates them in *one* forward pass and accepts the longest prefix that matches what it would have sampled itself. Rejected tokens force a recomputation from the divergence point. The output distribution is provably identical to plain greedy / sampled decoding from the verifier — speed comes for free, no quality loss. Modern variants (EAGLE, Medusa, lookahead, multi-tier) differ in *how* the drafter is built and *how* verification is structured.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Autoregressive decoding is bottlenecked by serial dependency: token $t+1$ needs token $t$. Each step is memory-bandwidth-bound on a large model (weights barely change per step; compute is under-used). Speculative decoding amortizes that by **packing $k$ candidate tokens into one verifier forward** — the verifier's per-step cost is dominated by weight movement, so verifying $k$ tokens in parallel is nearly the same cost as verifying one.

Two roles:

- **Drafter** — small, fast model (or routed sub-path of the big one). Proposes $\hat{x}_{t+1}, \ldots, \hat{x}_{t+k}$.
- **Verifier** — the actual model being served. Computes its own distributions $p(x_{t+1}), \ldots, p(x_{t+k})$ in parallel given the draft sequence.

## How it works

```
1. Drafter generates k candidate tokens
2. Verifier evaluates them in 1 forward pass → distributions p_t..p_{t+k}
3. For each draft token i in order:
   - Accept with probability min(1, p_verifier(x_i) / p_drafter(x_i))
4. On rejection at position j:
   - Sample x_j from a corrected distribution (p_verifier - p_drafter)+
   - Discard x_{j+1..k}
5. Slide window forward; repeat
```

The acceptance rule is what makes the output **distribution-equivalent** to direct verifier sampling — proven by Leviathan et al. (2022) and Chen et al. (2023).

## Why it matters

- **2–4× wall-clock speedup** at large batch=1 inference on commodity LLMs — the regime that dominates chat and agent serving.
- **No quality loss.** Output distribution is identical to direct verifier decoding.
- **Composes with batching / paged attention / continuous batching.** SD is orthogonal to the rest of the inference stack.
- **Modern default** for low-latency LLM serving (vLLM, SGLang, TGI all ship speculative variants).

## Gotchas & tricks

- **Drafter quality dominates speedup.** Acceptance rate $\alpha$ → speedup $\approx \frac{1-\alpha^{k+1}}{1-\alpha}$. A bad drafter wastes verifier compute; aim for $\alpha \geq 0.6$.
- **Tree drafting beats linear drafting.** Drafting *several* candidate continuations as a tree and verifying them jointly (Medusa, EAGLE-2) raises effective acceptance.
- **Binary verify-or-recompute is brittle.** Multi-tier schemes (e.g. [VIA-SD](via-sd.md)) add a slim middle-cost verifier for medium-confidence tokens — recoverable rejections become cheap rather than discarded.
- **Sampling temperature interacts.** Higher temperature → wider verifier distribution → higher acceptance, but also more drift; SD is most powerful at low-temperature greedy decoding.
- **Memory cost.** The drafter (or its KV cache) lives alongside the verifier. With self-drafting variants (EAGLE shares the verifier's transformer trunk), this overhead is small.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, 2022.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023.
- Paper: *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *VIA-SD: Verification via Intra-Model Routing for Speculative Decoding* — Xian et al., 2026 — see [via-sd.md](via-sd.md).

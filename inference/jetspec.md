# JetSpec

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A head-on-target speculative-decoding scheme whose draft head emits an entire **causal tree of candidate tokens in one forward pass**. Each branch is causally conditioned on its parent path, so the tree's joint score aligns with the target model's autoregressive factorization — fixing the "individually plausible but mutually inconsistent" failure of bidirectional block-diffusion drafters. Reaches up to **9.64× speedup on MATH-500** and 4.58× on chat with Qwen3 dense and MoE targets on H100.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md), [kv-cache-compression](kv-cache-compression.md)

---

## What it is

The frontier of speculative decoding has been stuck on a **causality-efficiency dilemma**:

- **Autoregressive drafters (EAGLE-style):** path-conditioned candidates → high acceptance, but drafting cost grows with tree depth.
- **Bidirectional block-diffusion drafters:** emit all positions in one pass → cheap drafting, but the per-position marginals don't condition on each other, producing trees where individually-plausible siblings are mutually inconsistent. The verifier wastes budget on impossible branches.

JetSpec is the first head architecture to break both horns simultaneously: **one forward pass** *and* **branch-causal conditioning**.

## How it works

1. **Hidden-state fusion.** The frozen target model's hidden states from the most recent step are fused (concatenation + projection) into a compact input feature for the draft head.
2. **Causal parallel head.** A small transformer head consumes that feature and emits the full draft tree. Each branch attends causally to its ancestor positions inside the tree, so a child token is scored conditional on its parent path — *but all branches are produced in the same forward pass*. This is the part bidirectional drafters can't do.
3. **Aligned verification.** Because each branch's joint score factorizes the same way the target model would factorize the path, the verifier can accept the *longest* matching prefix per branch under the standard exact-match SD criterion. No additional rejection of "phantom" branches is needed.
4. **vLLM integration.** Plugs into the vLLM serving stack so realistic batched-load benefits also materialize end-to-end, not just on single-stream microbenchmarks.

Each draft step then costs **one** draft-head forward + **one** target forward, regardless of tree depth — flat in the draft budget — while accepted prefixes grow longer as the budget grows.

## Why it matters

Reasoning workloads (long CoT, code) have very high local predictability, which means SD acceptance can be near 1 when the drafter is good. Prior heads couldn't *use* big tree budgets profitably; JetSpec converts those budgets into accepted prefixes that scale roughly linearly. On long-form tasks (MATH-500, code) it widens the gap between speculative and standard decoding by another 2–3× over EAGLE-style baselines.

It also has practical deployment leverage: a single small head on top of the existing target, no second hosted model, and direct vLLM support.

## Gotchas & tricks

- **The "free lunch" is bounded by acceptance.** Open-ended chat workloads — where the policy distribution is wider — top out around 4–5×, not 9×. The huge MATH-500 number is a high-acceptance regime.
- **Head training matters.** The head is trained on the frozen target's distribution; mismatches degrade acceptance quickly. Re-train per target model family.
- **Tree shape is a hyperparameter.** Deeper trees raise potential accepted length but increase verifier cost per step; the gain plateaus before the cost does. The paper sweeps a sensible default.
- **Composes with cache compression.** Memory pressure during prefill+SD is real; combine with KV compression schemes for long-context workloads.

## Sources

- Paper: *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting* — Hu, Feng, Wu, Yuan, Zhao, Qian, Wang, Zhao, Jiang, Zhu, Rosing, Zhang, 2026 — [arXiv:2606.18394](https://arxiv.org/abs/2606.18394).
- Code: [github.com/hao-ai-lab/JetSpec](https://github.com/hao-ai-lab/JetSpec)

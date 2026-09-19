# Instance-level Difficulty-Aware Control (IDAC)
*Depth — reward shaping that lets a hybrid reasoning model decide per-instance whether to think.*

**TL;DR:** Uniform length penalties trade accuracy on hard problems for token savings on easy ones. IDAC replaces the uniform penalty with a **per-instance** shaping term keyed on pre-computed reference statistics (e.g. baseline-model accuracy/length on this specific problem): easy instances are pushed toward direct answering, hard instances are allowed extended CoT. Introduced in **When2Think** (Shim et al. 2026); on AIME24, Pass@3 rises by 10.0% while token usage drops by 27.9%; AIME25 Pass@3 reaches 40.0%.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md), [../grpo](../grpo.md)
**Related:** [long2short](long2short.md), [../rlvr](../rlvr.md), [../../evaluation/aime](../../evaluation/aime.md)

---

## What it is

A **reward-shaping mechanism** used during hybrid-reasoning post-training. Where [length-penalty](length-penalty.md) uses a *group-relative* linear penalty within each set of $k$ rollouts, IDAC precomputes a per-problem *reference statistic* (e.g. baseline model's Pass@k and typical CoT length on this exact instance) and shapes the reward so the model is *encouraged to skip the CoT* on problems where thinking doesn't help and *preserve extended CoT* on problems where it does.

Hybrid reasoning models emit either "think and answer" or "answer directly" depending on a toggle in their prompt or a first-token router. IDAC trains the router.

---

## How it works

Two ingredients:

1. **Reference statistics per problem.** Before RL, run a baseline pass to compute per-instance quantities that identify difficulty:
   - Accuracy of a baseline model.
   - Typical CoT length for successful runs.
   - Optionally: solve rate at short CoT vs. long CoT.

2. **Difficulty-conditioned reward shaping.** During RL, the total reward for rollout $i$ on problem $x$ is:

$$
r(i) = r_{\text{outcome}}(i) + w_D \cdot s(x) \cdot \phi(\text{length}(i))
$$

where $s(x)$ is a per-instance shaping coefficient derived from the reference stats (positive on easy problems, near-zero on hard ones), and $\phi$ rewards shortness. Signs and thresholds are set so:

- Easy problem + short correct answer → boosted.
- Easy problem + long answer → penalized (unnecessary thinking).
- Hard problem + long correct answer → unchanged (thinking earned).
- Hard problem + wrong short answer → unchanged (no reward for premature answering).

The specific $s(x)$ function and reward coefficients are tuned per paper; the pattern is what matters.

---

## Why it matters

- **Beats uniform length penalties on Pareto.** On AIME24, IDAC hits +10.0% Pass@3 at −27.9% tokens — both metrics improve, not a tradeoff.
- **Fits the hybrid-reasoning wave.** Modern reasoning models (Kimi-K1.5, R1, K2) increasingly ship with a "think or don't" toggle. IDAC is a template for training that toggle.
- **Cheap to instrument.** Reference statistics are a one-time offline computation; no online reward-model calls.

---

## Gotchas & tricks

- **Reference statistics must generalize.** If you compute $s(x)$ from a weaker baseline, hard problems may look artificially hard and easy problems easy — the shaping direction can invert. Use the same base model.
- **The problem distribution matters.** IDAC assumes a heterogeneous mix of easy and hard problems. A dataset that is uniformly hard collapses the shaping term.
- **Coefficient tuning is per-benchmark.** AIME numbers depend on the specific $w_D$ and $\phi$; expect to re-tune when moving to code or open-domain reasoning.
- **Interacts with EOS alignment.** If length inflation from EOS mismatch is present (see [../eos-alignment](../eos-alignment.md)), IDAC's length signal is confounded — fix the tokenizer first.
- **Doesn't replace length-penalty.** For pure long-CoT (non-hybrid) models, the [length-penalty](length-penalty.md) group-relative penalty remains simpler.

---

## Sources

- Paper: *When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models* — Shim et al., 2026 (Microsoft) — [arXiv:2609.19671](https://arxiv.org/abs/2609.19671).

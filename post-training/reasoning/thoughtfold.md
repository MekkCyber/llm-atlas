# ThoughtFold
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** ThoughtFold reduces over-thinking in long-CoT reasoning models by treating *redundant exploration* inside correct trajectories as the negative half of a preference pair. Introspection identifies which sub-segments could be elided while still arriving at the right answer; a masked DPO-style objective then trains the model to take the shorter, folded path. Cuts DeepSeek-R1-Distill-Qwen-7B token usage by ~56% at parity accuracy.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../dpo.md](../dpo.md)
**Related:** [length-penalty.md](length-penalty.md), [long2short.md](long2short.md)

---

## What it is

Long-CoT models trained with RLVR routinely emit reasoning traces of 5–20k tokens, much of which is exploratory dead-ends ("let me try X… that doesn't work… let me try Y…"). Length penalties shrink traces uniformly, which can erase necessary deliberation alongside redundancy. ThoughtFold is *surgical*: it identifies which spans inside a *correct* trajectory are redundant and trains the model to skip them.

## How it works

1. **Introspective span labeling.** For each correct trajectory, prompt the model itself to identify redundant exploration spans — sub-sequences whose removal would still leave a coherent solution.
2. **Sub-trajectory spectrum.** Construct a family of candidate trajectories by masking subsets of labeled redundant spans. The original is the "least folded"; the fully-masked version is the "most folded."
3. **Preference pairs.** Within the spectrum, pair each more-folded trajectory $o^+$ against a less-folded one $o^-$. Both reach the correct answer, so the preference is purely on concision.
4. **Masked preference optimization.** Apply a DPO-style loss over these pairs, but with token-level masking that focuses gradient on the boundary tokens where the model would decide to enter or exit a redundant span:
   $$L_{\text{TF}} = -\log \sigma\!\big(\beta \log \tfrac{\pi_\theta(o^+ \mid q)}{\pi_{\text{ref}}(o^+ \mid q)} - \beta \log \tfrac{\pi_\theta(o^- \mid q)}{\pi_{\text{ref}}(o^- \mid q)}\big)$$
   computed only over the tokens that differ between $o^+$ and $o^-$.

The objective explicitly penalizes the model for taking exploratory detours and rewards bridging directly between essential reasoning segments.

## Why it matters

- **Surgical vs blunt.** Reward shaping with a length penalty is blunt — every trace gets shorter. ThoughtFold only shortens the parts the model itself flags as redundant.
- **No new reward model needed.** The objective rides on existing preference-optimization infrastructure (DPO loss, reference model). Drop-in for any long-CoT model.
- **Big practical win.** ~56% token reduction at SOTA accuracy on DeepSeek-R1-Distill-Qwen-7B. At inference time, that is roughly a 2× latency and cost reduction.

## Gotchas & tricks

- **Introspection quality is the bottleneck.** If the model mislabels essential reasoning as redundant, training will *hurt* accuracy. The paper relies on a strong distilled reasoner that is reasonably honest about which parts of its trace it actually used.
- **Masking matters.** Vanilla DPO over full trajectories tends to wash out the signal because most tokens are identical between $o^+$ and $o^-$. Masking to the differing tokens concentrates gradient.
- **Stack with length-penalty RL?** Probably orthogonal — ThoughtFold removes redundant exploration; a length penalty further compresses essential reasoning. The authors do not stack the two; combinations are open.
- **Generalization.** Reductions transfer across math/code reasoning benchmarks; behavior on open-ended tasks (where redundant exploration is harder to define) is not characterized.

## Sources

- Paper: *ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning* — Liu et al., 2026 — [arXiv:2606.03503](https://arxiv.org/abs/2606.03503).
- Related: DPO (Rafailov et al., 2023); long2short and length-penalty literature.

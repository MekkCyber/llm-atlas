# Predictive Divergence Mask
*Depth — masking off-policy LLM RL tokens by the sign of the *predicted* first-order change in behavior-vs-training divergence, not by the sampled importance ratio.*

**TL;DR:** PPO-style trust-region masks in LLM RL have two silent components: a **proximity criterion** ("are we already outside the trust region?") and a **direction criterion** ("would this update push us further out?"). Ratio-based methods use the sampled importance ratio for both, which is a poor proxy for how the full softmax distribution actually moves. Divergence-mask methods (DPPO) upgraded the proximity criterion to a real distributional divergence but kept the ratio-based direction criterion — a mismatch. **Predictive divergence masks** derive a closed-form first-order predictor of the divergence change under softmax policies and mask by its *sign*, making direction and proximity coherent.

**Prereqs:** [ppo](ppo.md), [_rl](_rl.md)
**Related:** [grpo](grpo.md), [rlvr](rlvr.md), [../systems/partial-rollouts](../systems/partial-rollouts.md)

---

## What it is

An asymmetric mask on the policy gradient update for off-policy LLM RL. A token is masked out (its gradient zeroed) when both:

- **proximity fails**: the behavior-vs-training divergence at that state already exceeds a trust-region radius, and
- **direction fails**: the next gradient step would further *increase* that divergence.

Predictive divergence masks change how "direction" is computed. Instead of reading the sampled importance ratio $\pi_\theta(y|s) / \pi_{\text{behavior}}(y|s)$, they compute a closed-form first-order predictor of how the divergence changes under the next gradient step.

## How it works

For a softmax policy, differentiate the KL / divergence between behavior and training policies with respect to the parameters. The predicted first-order change decomposes cleanly into two terms:

$$\Delta D \;\approx\; \underbrace{c_{\text{local}}(y_s, p_\theta(y_s))}_{\text{sampled-token contribution}} \;+\; \underbrace{c_{\text{global}}\!\big(p_\theta(\cdot)\big)}_{\text{softmax-normalization coupling}}.$$

The **local term** at the sampled token exactly matches the quantity the ratio-based direction criterion reads. The **global term** captures how probability mass moves over the *rest* of the vocabulary because of softmax renormalization — invisible to a single-sample ratio. Keeping only the local term recovers PPO's direction check; keeping both lets the sign of $\Delta D$ track the true divergence change.

Practical rollout engines expose only top-$K$ logprobs. The paper closes the tail gap with two lightweight estimators — an **aggregated-tail** approximation (treat the tail as one lump) and a **uniform-tail** approximation (spread the remaining mass uniformly). Both are simple reductions over the retained probabilities and add negligible overhead.

## Why it matters

Practical LLM RL is inherently off-policy: the rollout engine (vLLM/SGLang, low precision, different kernels) is not bitwise identical to the training stack, and rollouts are reused across several minibatch updates. This makes the sampled importance ratio a lossy summary of the actual shift. Predictive divergence masks give a *coherent* mask — the proximity and direction criteria now both refer to the same divergence — and empirically improve training stability across model scales and precision settings.

## Gotchas & tricks

- **Tail estimator matters at low top-K.** With top-K=20 and highly peaked distributions the tail is a lot of mass; the uniform-tail estimator systematically over-corrects on very sharp distributions.
- **Sign, not magnitude.** The mask uses only the sign of the predicted change. Using the magnitude leaks the divergence into the effective learning rate.
- **Compatible with existing PPO/DPPO stacks.** No new hyperparameter — swap the direction check.
- **Doesn't fix reward hacking.** Better off-policy correction preserves signal from the reward, whatever the reward is. Verify rewards separately.

## Sources

- Paper: *Predictive Divergence Masks for LLM RL* — Zhou, Yao, Qi, Ping, Tang, Wang, Pang (Tencent Hunyuan / UIUC / NUS), 2026 — [arXiv:2607.10848](https://arxiv.org/abs/2607.10848).
- DPPO (divergence-mask baseline the paper builds on) — cited within.

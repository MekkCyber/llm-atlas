# Latent Reasoning (Continuous-Space CoT)
*Depth — reasoning in a continuous hidden state instead of discrete tokens, trained with intrinsic rewards for multimodal generation.*

**TL;DR:** Classical chain-of-thought reasoning emits discrete reasoning tokens before producing the answer. For **unified multimodal models** that generate continuous outputs (images, video), this discrete-to-continuous bridge is where test-time scaling breaks — longer text CoT stops helping. LARC (Latent Reasoning in Continuous Space) replaces text reasoning steps with **iterated updates to a continuous reasoning latent** that lives in the same manifold as the generator's input. Trained in two stages: a *curriculum* fine-tune that gradually replaces text CoT with latent CoT, then RL with **intrinsic rewards** (information gain + diversity), no ground-truth reasoning chain required. On GenEval, posts SOTA unified-model results and *monotonic test-time scaling* as the number of latent steps grows.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../_rl.md](../_rl.md), [../rlvr.md](../rlvr.md)
**Related:** [length-penalty.md](length-penalty.md) · [long2short.md](long2short.md) · [../grpo.md](../grpo.md) · [../../multimodal/README.md](../../multimodal/README.md)

---

## What it is

Chain-of-thought RL (R1-style) assumes a reasoning trace is a sequence of **discrete tokens** that the model can condition on when producing the final answer. This works when the answer is also discrete (math digits, code tokens). For **image / video / audio generation** inside a unified multimodal model, the answer is continuous — a sequence of latent tokens the generator decodes into pixels. Discrete reasoning tokens never enter that latent space, so most of the "think longer, image gets better" signal is lost at the bridge.

Latent reasoning removes the bridge: the reasoning trace *is* a continuous hidden state, iteratively refined, and the generator reads it directly.

## How it works

### Representation

Let the generator consume a conditioning latent $z \in \mathbb{R}^d$. A standard VLM/MLLM would produce $z$ from a text prefix in one shot. LARC instead maintains a reasoning latent $h_t$ and updates it for $T$ steps:

$$
h_{t+1} = f_\theta(h_t, x) \qquad z = g_\theta(h_T)
$$

where $x$ is the user prompt and $f_\theta$ is a transformer block applied iteratively. The number of steps $T$ is the test-time compute knob.

### Stage 1 — Curriculum fine-tune

Start from a backbone that already does text CoT (BAGEL in the paper). Progressively replace text reasoning *steps* with latent updates, keeping the final answer loss. The schedule anneals from "all text" to "all latent" over fine-tuning, so the latent trace inherits the text trace's structure before the text is removed.

### Stage 2 — Intrinsic-reward RL

With no ground-truth reasoning chain, external supervision is impossible. Two intrinsic rewards instead:

- **Information gain.** Reward the latent step for *reducing* the generator's uncertainty about the eventual output (measured via a surrogate, e.g. negative conditional entropy).
- **Diversity.** Reward latent trajectories that don't collapse to a single mode across prompts.

These combine into a per-rollout scalar that feeds standard policy-gradient RL (GRPO-shape in the paper's implementation).

### Why no ground-truth chain is needed

The reward is a property of the latent trajectory itself — "does it reduce the generator's uncertainty?" — not a comparison to a reference. So the usual bottleneck for CoT RL (where do we get the reasoning chains?) is removed.

## Why it matters

- **Test-time scaling for image generation.** LARC is the first unified-model recipe to show *monotonic* GenEval improvement as reasoning compute grows, matching the LLM "think-longer" curve that was previously absent from image generation.
- **No reasoning-chain supervision.** Removes the biggest cost of CoT RL for modalities where chains aren't natural (image generation, video, audio).
- **Reusable primitive.** The curriculum fine-tune ("replace token steps with latent steps") is a generic bridge between discrete-token CoT and continuous-latent generation.

## Gotchas & tricks

- **The curriculum is load-bearing.** Jumping straight to pure latent reasoning gives a cold-start that doesn't converge. The gradual replacement inherits the text trace's structure as scaffolding.
- **Intrinsic reward is a surrogate.** Information gain depends on a conditional-entropy surrogate; different surrogates give different optima. The paper's details on exactly which surrogate matter for reproduction.
- **Diversity reward prevents mode collapse** but can push the latent toward noise; coefficient tuning is non-trivial.
- **Decoupled from a verifier.** This is the signal and the limit: no verifier, no ability to catch catastrophic trajectories. Hallucination-shaped failures survive.
- **Backbone dependence.** Reported on BAGEL. Whether the curriculum transfers to backbones that didn't do text CoT first is open.

## Sources

- Paper: *Latent Reasoning in Continuous Space for Unified Multimodal Models (LARC)* — anonymous, 2026 — arXiv 2610.42149.
- Related: *Chain of Continuous Thought (Coconut)* — Hao et al., 2024 — text-only ancestor of continuous-latent reasoning.
- Related: *BAGEL* — the multimodal backbone LARC fine-tunes.
- Related: [long-cot-rl.md](long-cot-rl.md) — the discrete-token CoT counterpart.

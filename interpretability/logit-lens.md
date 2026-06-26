# Logit Lens
*Depth — read intermediate-layer hidden states through the model's own unembedding to see what each layer "predicts."*

**TL;DR:** A frozen LLM is a stack of residual updates. The logit lens projects the residual stream at *any* intermediate layer through the final unembedding (LM head) — without any further computation — and treats the result as the model's "current best guess" of the next token. Cheap, training-free, and surprisingly informative: most LLMs decide their answer well before the final layer, intermediate layers carry semantically meaningful predictions, and many downstream interpretability methods (probing, steering, jailbreak detection) build on top of this primitive.

**Prereqs:** [transformer-block](../architectures/transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md), [refusal-suppression](../safety/refusal-suppression.md)

---

## What it is

Given a frozen Transformer LM with $L$ layers, residual stream $h_\ell \in \mathbb{R}^d$ at layer $\ell$, output norm $\mathrm{LN}_f$, and unembedding $W_U \in \mathbb{R}^{V \times d}$:

```
lens(ℓ, t) = softmax( W_U · LN_f( h_ℓ[t] ) )
```

This produces a distribution over the vocabulary "as if layer ℓ were the last layer." No retraining, no auxiliary head — just borrow the existing unembedding.

Two close cousins:

- **Tuned lens** (Belrose et al., 2023): replace the bare unembedding with a *learned* affine map per layer that approximates the model's own final residual. Much less noisy at early layers.
- **Patchscopes / activation patching** (Ghandeharioun et al., 2024): same projection, used as a causal probe by patching one model's residual into another forward pass.

## How it works

The logit lens exploits two facts about Transformer LMs:

1. **Residual additivity.** The final hidden state is $h_L = h_0 + \sum_\ell \Delta h_\ell$, where each block contributes a residual update. Each $h_\ell$ already "lives in" the output-token space, just incompletely.
2. **Shared unembedding.** The same $W_U$ is used for the final prediction, so projecting any $h_\ell$ through it produces tokens — not "concepts in some intermediate space."

Per-layer entropy of the lens distribution typically:

- **Starts high** (uniform-ish) at the embedding layer.
- **Drops** through middle layers as the model commits to a prediction.
- **Drops further but plateaus** at the final layers.

The shape of this entropy trajectory turns out to be diagnostic — see *What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics* (Nikolenko et al., 2026), which uses monotonic trend statistics of the per-position entropy across layers to detect jailbreak prompts.

## Why it matters

- **Free interpretability.** Zero training, zero extra parameters. A scaffolding for almost every "what does layer X know" experiment.
- **Layer-by-layer decision tracing.** Tells you *when* the model commits to its answer. Often surprisingly early — see *Do Thinking Tokens Help with Safety?* (Ri et al., 2026): safety decisions are largely set after ~20% of the thinking chain.
- **Substrate for safety monitoring.** Mid-network entropy is more discriminative than final-layer entropy for many safety signals — the unembedding tends to "wash out" intermediate information.
- **Substrate for activation steering.** Logit-lens deltas are how steering vectors are interpreted ("this direction increases token X's probability at layer Y").

## Gotchas & tricks

- **Early-layer noise.** Bare logit lens is noisy at early layers because $h_\ell$ hasn't been projected to the output space yet. Use the *tuned lens* if you care about early layers.
- **Pre-norm vs post-norm.** Most modern LLMs are pre-norm; apply $\mathrm{LN}_f$ before the unembedding even at intermediate layers. Forgetting the norm produces garbage.
- **Per-position vs per-prompt aggregates.** A single per-prompt statistic (mean entropy) often carries little signal; trajectories over positions / layers are far more discriminative.
- **Don't confuse with attention probing.** Logit lens reads the residual stream. Attention probes read the attention weights. Different signal.
- **Quantization changes the picture.** With INT4-quantized models, the logit-lens distribution is noisier; tuned lens recovers a usable signal.
- **Open-source unembeddings only.** API-only models (closed weights) preclude the lens unless you can reach the logits at intermediate layers, which is rarely exposed.

## Sources

- Blog post: *interpreting GPT: the logit lens* — nostalgebraist, 2020 — [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
- Paper: *Eliciting Latent Predictions from Transformers with the Tuned Lens* — Belrose et al., 2023 — [arXiv 2303.08112](https://arxiv.org/abs/2303.08112).
- Paper: *What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics* — Nikolenko et al., 2026 — [arXiv 2606.25182](https://arxiv.org/abs/2606.25182).
- Paper: *Patchscopes* — Ghandeharioun et al., 2024 — extends the lens into a causal probe.

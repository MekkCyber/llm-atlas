# Logit Lens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Take the final unembedding matrix and apply it to the residual stream at every intermediate layer — as if that layer's activation were the model's final output. The resulting per-layer distribution over the vocabulary is the **logit lens**, and it lets you watch a prediction *form* through the depth of the network without training any probe. Standard tool for mechanistic interpretability; recently used to show that speech LMs latently "translate to text" mid-network.

**Prereqs:** [README.md](./README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [../multimodal/implicit-transcription.md](../multimodal/implicit-transcription.md)

---

## What it is

A trained transformer LM ends with an unembedding matrix `W_U` mapping the residual stream to vocabulary logits. The forward pass usually applies `W_U` only to the *final* layer's residual. The logit lens applies it to *every* intermediate layer:

```
logits_ℓ = LayerNorm(residual_ℓ) · W_U
```

Reading out `argmax(logits_ℓ)` at layer `ℓ` gives the model's "current guess" at depth `ℓ`. Sometimes the guess stabilises early; sometimes it flips between candidates; sometimes it converges to the answer only in the last few layers.

## How it works

**Practical recipe.**
1. Take a trained model, freeze it.
2. For each layer `ℓ`, take the residual stream after that layer's addition.
3. Optionally apply the model's final LayerNorm (some variants skip it; results are qualitatively similar).
4. Multiply by `W_U` to get vocabulary logits.
5. Read out top-k tokens; compute entropy; compare to final-layer output.

**No training.** The lens uses the model's own unembedding. There is no probe to fit — every result comes from the model itself.

**Variants.**
- **Tuned Lens** (nostalgebraist et al.): fit a small per-layer affine transform before applying `W_U`, correcting for representation drift between intermediate layers and the final residual. More accurate reads at earlier layers, but requires training.
- **Direct Logit Attribution:** decompose the final logit into per-head, per-position contributions using linearity of the residual stream. A more surgical version of the same idea.
- **Cross-modality lens:** apply the *text* unembedding to a *speech* or *image* residual stream to test whether the model has implicitly rendered its state into the text vocabulary. See [../multimodal/implicit-transcription.md](../multimodal/implicit-transcription.md).

## Why it matters

- **Zero-training interpretability.** Any pretrained transformer with an unembedding admits a logit lens — no annotations, no probe fitting.
- **Depth-resolved prediction dynamics.** Reveals where in the network a decision is *made*, distinct from where the final output is *produced*.
- **Multimodal probe.** Applying the lens across modalities is a lightweight test for whether a joint model uses a shared latent code (e.g. text) — the interleaved-SLM paper is a canonical recent example.

## Gotchas & tricks

- **Early layers are noisy.** Residual streams in the first few layers of a large model are not yet aligned with the unembedding — expect uninformative distributions. Tuned Lens is designed for this.
- **Skipping final LayerNorm shifts the read.** Applying `W_U` directly vs after LayerNorm gives measurably different distributions; pick one and be consistent.
- **Not a causal claim.** A token being at the top of the intermediate distribution doesn't mean the model *uses* that token in later layers. Pair with activation patching or causal scrubbing for causal claims.
- **Sensitive to tokenisation.** Multi-token words look worse than single-token words; longer BPE tokens dominate intermediate top-k reads.
- **Cross-modality reads are indicative, not proof.** The interleaved-SLM finding uses top-candidate rate as evidence of latent transcription — a strong hint, not a mechanistic guarantee.

## Sources

- Blog: *interpreting GPT: the logit lens* — nostalgebraist, 2020 — original writeup.
- Paper: *Eliciting Latent Predictions from Transformers with the Tuned Lens* — Belrose et al., 2023.
- Paper: *Interleaved Speech Language Models Latently Work In Text* — Sternberg, Maimon, Adi, 2026 — [arXiv:2606.22473](https://arxiv.org/abs/2606.22473) — cross-modality application that motivated extracting this depth file.

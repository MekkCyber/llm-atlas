# LLM Watermarking — and Its Ensemble Fragility
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distributional-perturbation watermarks (KGW family, SynthID-style) embed a statistical signature in LLM output by perturbing token logits in a way only the provider can verify. The dominant family is fragile in a *multi-provider world*: averaging output distributions across 3–5 competing models cancels each provider's perturbation, recovering the unwatermarked distribution up to second-order error. WASH (Watermark Attenuation via Statistical Hybridisation) demonstrates the attack practically, handling vocabulary misalignment and tokenization differences across heterogeneous models.

**Prereqs:** [_attacks](_attacks.md), [README](README.md)
**Related:** [../safety/_attacks.md](_attacks.md)

---

## What it is

LLM watermarking attaches a detectable statistical signature to generated text so a verifier with the secret key can later determine whether the text came from a watermarked model. The dominant **distributional-perturbation** family (Kirchenbauer et al. 2023, SynthID-Text, Aaronson scheme, etc.) works by partitioning the vocabulary into a pseudo-random "green list" at each step and biasing the model's logits toward green tokens. Detection counts green-list hits and runs a z-test.

The WASH paper proves a structural vulnerability: when a user can query multiple LLMs (today's reality), simply averaging the providers' output probability distributions cancels the watermarks — because each provider's perturbation is approximately independent of the others.

---

## How it works

### Distributional-perturbation watermarks (background)

At each step, given context $c$:

- Partition vocabulary into green/red sets via PRF on a context window.
- Add a bias $\delta$ to green logits before softmax: $\tilde p(v|c) \propto p(v|c) \cdot \exp(\delta \cdot \mathbb{1}[v \in G(c)])$.
- Detection: count green tokens over a passage; large positive z-score → watermarked.

The text-quality cost is small if $\delta$ is modest; detection power grows with passage length.

### The ensemble attack

Suppose $K$ providers each watermark with independent green-list partitions. Their watermarked distributions are $\tilde p_k(v|c) = p_k(v|c) \cdot w_k(v|c)$ where $w_k$ is provider-$k$'s perturbation. With matched tokenizers and similar base distributions $p_k \approx p$:

$$\bar p(v|c) = \tfrac{1}{K} \sum_k \tilde p_k(v|c) \approx p(v|c) \cdot \tfrac{1}{K} \sum_k w_k(v|c)$$

Because the green sets $G_k(c)$ are independent across providers, $\tfrac{1}{K}\sum_k w_k(v|c) \to \mathbb{E}[w(v|c)]$ for every $v$ — a *context-independent constant*. The watermark perturbation cancels out, leaving a constant rescaling that's absorbed by re-normalization. The paper proves the residual error is second-order in the perturbation magnitude.

### Practical issues WASH handles

- **Vocabulary misalignment:** providers use different tokenizers. WASH aligns distributions to a common vocabulary using byte-level normalization.
- **Heterogeneous base quality:** averaging weak + strong models hurts text quality. A **fluency-aware router** weights distributions by confidence.

### Attack budget

3–5 models are enough to drive detection z-scores from 5–300 down to <2 (below the threshold of 4) and TPR@5%FPR below 50%. WASH runs 6× faster than a paraphrase-attack baseline of comparable detection-evasion effectiveness, and quality improves by 27.5% (the ensemble outperforms any single watermarked model on fluency).

---

## Why it matters

- **Refutes the implicit single-provider threat model.** Most watermarking papers assume the attacker can only query the watermarked model. WASH shows that *any user* with API access to multiple competing models can defeat distributional watermarks for free, with no adversarial expertise.
- **Theoretical bound is tight.** The second-order residual means provider-specific defenses (e.g. correlated green lists across providers) would need coordination — currently antitrust-infeasible.
- **Provenance ≠ watermarking.** WASH undermines distributional watermarks but not cryptographic provenance approaches (C2PA-style signatures over model outputs at API time). The right defense layer moves up the stack from text to API metadata.

---

## Gotchas & tricks

- **Doesn't apply to non-distributional watermarks** (e.g. sampling-based schemes that select tokens via shared randomness across providers — though those have their own coordination problem).
- **Tokenizer alignment is the practical bottleneck.** Vocabulary differences mean averaging happens after expensive normalization; WASH includes specific tricks (BPE-equivalent merging) that other ensemble attackers will need to replicate.
- **Quality bonus.** Ensemble averaging usually *improves* fluency vs any individual model — so the "attack" is also a free quality boost, which makes the attack-cost calculation even more lopsided in the attacker's favor.
- **Single-provider settings remain safe.** If watermarking is deployed only by one provider and the threat model is "is this text from us?", distributional watermarks still work; WASH only breaks the multi-provider attribution case.
- **Composability with paraphrase attacks.** WASH + paraphrase compose multiplicatively — the WASH-attacked text can be further paraphrased, but typically WASH alone is already below the detection threshold.

---

## Sources

- Paper: *Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs* — Wu, Gong, Zhu, Chen, Zhao, 2026 — [arXiv:2605.30501](https://arxiv.org/abs/2605.30501) — introduces WASH and the second-order cancellation proof.
- Background: *A Watermark for Large Language Models* — Kirchenbauer et al., ICML 2023 — the green-list distributional-perturbation scheme WASH targets.
- Background: *SynthID-Text* — Dathathri et al., Nature 2024 — Google's deployed variant.
- Background: paraphrase-attack literature on watermark robustness — Krishna et al. 2023 — the comparison baseline WASH beats on both compute and effectiveness.

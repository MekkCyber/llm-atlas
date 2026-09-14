# Next Concept Prediction (NCP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretraining objective that predicts multi-token *concepts* alongside next tokens. Concepts are discrete codes from a product-quantized (PQ) codebook built directly from the model's own hidden states, so no external segmenter is needed. Layered on top of standard NTP, NCP shifts the pretraining horizon from single tokens to learned concept trajectories and improves token efficiency and downstream reasoning at 8B+ scale.

**Prereqs:** [mtp](mtp.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [mtp](mtp.md), [_tokenization](../fundamentals/_tokenization.md), [dolma](../data/dolma.md)

---

## What it is

Standard next-token prediction (NTP) supervises the model one token at a time. Next Concept Prediction (NCP) adds a *concept-level* objective: at each position, the model must also predict the next discrete concept code representing the next multi-token span. The concept vocabulary is not hand-designed — it is built by product quantization directly over the backbone's hidden states, so the model *invents* its own concept space and then supervises itself on it.

Concepts, once predicted, are fed back into the token stream to condition subsequent generation. This means the same backbone learns (a) a token-level generative model, (b) a concept-level generative model over its own latent space, and (c) a token-level model conditioned on those predicted concepts — jointly, end-to-end.

## How it works

- **Concept vocabulary via PQ.** Split each hidden state into $M$ sub-vectors, quantize each with its own small codebook of size $K$. The resulting code is a tuple in $[K]^M$, giving up to $K^M$ concepts without materializing them all.
- **Concept Module.** A small transformer head consumes the current backbone state and predicts the *next* concept code (all $M$ sub-codes jointly or factorized). Its loss is cross-entropy over the PQ codes.
- **Injection back into tokens.** Predicted concept embeddings are concatenated / added to the token stream so subsequent NTP steps generate under the concept's guidance.
- **Joint objective.** $L = L_{\text{NTP}} + \lambda \, L_{\text{NCP}}$, trained end-to-end. The PQ codebook is updated online (EMA / straight-through).

## Why it matters

At 8.9B params / 5.73T Dolma-3 tokens, NCP-ArchPreview reached OLMo-3-7B's pretraining loss using only 51.3% of the tokens and beat it by +2.45 macro (+5.99 GSM8K). NCP also produces a *reusable interface*: retraining just the 17M-param VQ module cheaply adapts the model to a new domain, and feeding NCP-predicted concepts into a speculative-decoding drafter raises accepted-length by 4.17%. It is the strongest 8B-scale evidence that latent-space objectives justify their complexity.

## Gotchas & tricks

- **PQ over hidden states, not inputs.** The codebook is built over the model's own representations so concepts co-evolve with the model — no frozen tokenizer bottleneck.
- **The Concept Module is small.** ~17M params in the published run. Bigger heads did not help and slowed training.
- **Don't drop NTP.** NCP alone under-specifies token-level generation. The two losses are complementary; ablations show NTP-only < joint, NCP-only < joint.
- **Concept prediction is easier than token prediction late in training** — schedule $\lambda$ so NCP does not dominate once the model is well-fit at the token level.
- **Sharing the head matters.** The concept module reads the backbone's *final* hidden state (analogous to MTP's design), so the concept space is anchored to the model's own best representation rather than an intermediate layer.

## Sources

- Paper: *NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction* — Intern-NCP Team, 2026 — https://arxiv.org/abs/2609.10715
- Related: *Multi-token Prediction* — Gloeckle et al., 2024 — the token-horizon predecessor.

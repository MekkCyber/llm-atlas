# Latent Memory Tokens
*Depth — compressing context into continuous embedding-space tokens read directly by a frozen decoder.*

**TL;DR:** Instead of compressing context into shorter *text* (or a rendered *image*) that the decoder must re-decode, compress it into continuous **memory tokens** injected via the decoder's normal input-embedding path. LatentPress (Sang, ASU, 2026) trains an encoder to produce such tokens for both text and vision inputs and shows a frozen decoder can consume them without any text reconstruction at inference — a machine-facing context channel that skips a decode round-trip.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [kv-cache-eviction](kv-cache-eviction.md)

---

## What it is

A context-compression scheme with an unusual output type. A trainable encoder maps a long context (chat history, document, image) to a fixed budget $N \ll L$ of continuous vectors $z_1, \ldots, z_N \in \mathbb{R}^d$ with $d$ matching the target decoder's embedding dimension. These vectors are then spliced into the decoder's input embedding stream, at reserved positions, alongside the actual (unchanged) query tokens. The decoder is **frozen**; only the encoder is trained.

Contrast with:
- **Text-based compression** (summarizers, LLMLingua) — outputs shorter text; decoder still decodes text.
- **Rendered-image compression** — encodes context as a rendered image consumed by a VLM; requires a vision encoder round-trip per inference.
- **Soft prompts / prefix tuning** — related mechanism, but historically task-specific and trained per downstream task; latent memory tokens are trained as a general context channel.

## How it works

Training the encoder:

1. Sample a document / dialog $c$ and a downstream query / continuation $q$.
2. Encoder produces $z = E_\phi(c) \in \mathbb{R}^{N \times d}$.
3. Frozen decoder consumes `[z; embed(q)]` and predicts the target continuation.
4. Loss: cross-entropy of the decoder's predictions on the target, or a distillation loss against the decoder's predictions when given the full raw context.

At inference the encoder runs once per context, its output is cached, and the decoder consumes the tokens through its normal input-embedding interface. There is no text reconstruction step.

## Why it matters

- **Skips a decode round-trip.** Text-based compressors decode to text and then re-encode into embeddings; latent tokens go straight into the embedding path.
- **Works across modalities.** Because the output is embedding-shaped, the same interface serves text-context and vision-context compression; LatentPress is validated on both.
- **Decoder stays frozen.** Any deployed decoder can be adapted with a compressed-context front-end without touching decoder weights.

## Gotchas & tricks

- **Encoder tied to a specific decoder.** The output tokens must lie in the target decoder's embedding space; changing decoder means re-training the encoder.
- **Fixed budget vs adaptive.** Fixed $N$ per context is simplest; adaptive-budget variants (longer contexts → more tokens) recover some accuracy at the cost of caching complexity.
- **Not human-inspectable.** Latent tokens carry no textual form — debugging and safety review lose a surface they had with text compression.
- **Position ids.** Reserve a stable positional slot for the memory-token block so the decoder's positional encoding does not drift with query length.
- **Composability.** Can be combined with KV eviction on the *query* side; the memory-token block itself is short and typically fully retained.

## Sources

- Paper: *LatentPress: Context Compression Beyond Text and Vision* — Hejian Sang, Arizona State University, 2026 — https://arxiv.org/abs/2609.01507
- Related: soft prompts / prefix tuning — Li & Liang, 2021 — a task-specific ancestor of the mechanism.
- Related: LLMLingua — Jiang et al., 2023 — the text-based compression baseline.

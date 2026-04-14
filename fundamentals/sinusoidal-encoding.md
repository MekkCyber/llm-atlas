# Sinusoidal Positional Encoding
*Depth — the fixed-frequency sin/cos encoding from the original Transformer.*

**TL;DR:** A deterministic, non-learned positional signal built from sinusoids at exponentially spaced frequencies. Added directly to the token embeddings before the first layer. Gives each position a unique vector and — because the frequencies are geometric — encodes relative offsets as linear functions, which attention can pick up on.

**Prereqs:** [attention](attention.md)
**Related:** [positional-encoding](positional-encoding.md)

---

## What it is

A function `PE : ℕ → ℝ^d` that maps a position index to a `d`-dimensional vector, using pairs of (sin, cos) at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- `pos` is the token's position in the sequence (0, 1, 2, …).
- `i` indexes into the embedding dimension (0 … d/2 − 1).
- `10000` is the base wavelength — small `i` → high frequency, large `i` → low frequency.

The resulting vector has the same dimension as the token embedding. It is **added** to the token embedding before the first Transformer layer. Nothing is learned — the encoding is fixed.

## How it works

### The frequency spectrum

For dimension pair `i`, the angular frequency is `1 / 10000^(2i/d)`. This spans from `1` (when `i = 0`) to roughly `10000^{-1}` (when `i = d/2 − 1`). The lowest dimensions oscillate fast (give fine-grained position info), the highest dimensions oscillate slowly (give coarse "which half of the sequence" info).

Analogy: a binary counter has low bits that flip every step and high bits that flip rarely. Sinusoidal encoding is the continuous, dense version of that.

### Why it works with attention

The key property: for any fixed offset `k`, the encoding `PE(pos + k)` can be written as a **linear function of `PE(pos)`**. Concretely, because

```
sin(a + b) = sin a cos b + cos a sin b
cos(a + b) = cos a cos b − sin a sin b
```

the shift from `pos` to `pos + k` is a rotation in each `(sin, cos)` pair by a fixed angle. Attention can, in principle, learn a linear projection that turns this rotation into a relative-position signal. So although the encoding itself is absolute, relative offsets are linearly recoverable from it.

### Where it's injected

```
x₀ = TokenEmbedding(tokens) + PE(positions)
```

Just once, at the input. All Transformer layers see this shifted embedding, and whatever positional reasoning they do is built on top of this signal propagating through residual streams and attention.

## Why it matters

- **First solution to the problem**. Every later positional encoding (learned absolute, T5 relative, RoPE, ALiBi) is responding to something sinusoidal encoding did or didn't do well.
- **No parameters**, no training instability, no vocabulary-size issue at inference. Works at any position you can compute the formula for.
- **Partial length extrapolation**. Because it's a deterministic function, you can evaluate it at positions never seen during training. The model may or may not use those positions well, but the encoding itself is defined.

## Gotchas & tricks

- **Added, not concatenated.** The encoding shares dimensions with the token embedding. This can look strange — why isn't it a concat? Adding works because the embedding space is large and the two signals live in (mostly) different subspaces that the model learns to disentangle.
- **Choice of base (`10000`) is a hyperparameter.** It controls the longest wavelength and therefore the maximum sequence length the encoding can distinguish. Too small → positions far apart look similar; too large → most dimensions barely move across the full sequence.
- **Extrapolation is *not* guaranteed.** Sinusoidal encoding extrapolates the *encoding*, not the *behavior*. Attention patterns learned at length 512 may not transfer to length 8192 even though the encoding is well-defined there.
- **Not invariant to scale.** Scaling up `d_model` without adjusting the base wavelength changes the effective frequency spectrum.
- **Modern LLMs don't use it.** Decoder-only LLMs have moved to RoPE (which encodes relative position through rotation of Q/K rather than additive input signal). Sinusoidal encoding's main role today is historical and as a teaching baseline.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762 (Section 3.5).

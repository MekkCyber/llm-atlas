# Tied Embeddings
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Share parameters between the input token embedding matrix and the output unembedding (classifier) matrix. The standard trick in LM design since GPT-2: halves the parameter count of a model's largest table (vocab × $d_\text{model}$), with minor quality impact. onóna (2026) extends this with a **two-table tied-pair + gated mixture**: two tied embedding tables, averaged on input, mixed with a per-token gate on output.

**Prereqs:** [./_tokenization.md](./_tokenization.md), [./bpe.md](./bpe.md)
**Related:** [../architectures/transformer-block.md](../architectures/transformer-block.md)

---

## What it is

A transformer has two large lookup-shaped matrices:

- **Embedding** $E \in \mathbb{R}^{V \times d}$ — maps token id → vector.
- **Unembedding** $U \in \mathbb{R}^{V \times d}$ — maps final hidden → logits over vocab.

For vocabularies in the 128k–256k range and $d \in \{2048, 4096\}$, each table is 0.3B–1B parameters. **Tied embeddings** set $U = E$ (or $U = E^\top$), cutting that in half.

The intuition is that embedding and unembedding geometries should align: the vector that writes "token $t$ means X" into the residual stream should be the same as the vector that reads "the model wants to emit token $t$". Empirically this is close enough to hold that tied embeddings are nearly free in quality.

## How it works

**Classical tie** (weight sharing):
$$
\text{logits}(h_t) = h_t \cdot E^\top, \qquad x_t = E[\text{id}_t]
$$

Both terms share $E$. Gradient from the logit loss and from the embedding lookup co-train $E$.

**onóna extension** (2026): two tied pairs $(E_a, E_b)$, with
$$
x_t = \tfrac{1}{2}(E_a[\text{id}_t] + E_b[\text{id}_t])
$$
on input, and output
$$
P(\cdot \mid h_t) = g(x_t) \cdot \mathrm{softmax}(h_t E_a^\top) + (1 - g(x_t)) \cdot \mathrm{softmax}(h_t E_b^\top)
$$

where $g$ is a learned gate depending only on the current token. This decouples the input and output roles slightly — the model can route tokens whose "write" and "read" geometry disagree through different tables — while keeping parameter count close to a single tied pair.

## Why it matters

- The embedding tables are often the single largest parameter group in a modern LM; a technique that saves tying cost or improves quality at that spot compounds across every pretrain.
- **Empirical:** at 142M, 330M, and 1B across dense, Gated-DeltaNet hybrid, MoE, and attention-residual architectures, onóna lowers validation loss and external perplexity and raises downstream average accuracy on most settings (1B MoE is the main exception on loss).
- Classical tied-embeddings (single tie) is still the default for most open models; onóna is a drop-in replacement.

## Gotchas & tricks

- **Not the same as "weight-tied LMs"** (encoder-decoder architectures where whole blocks are shared) — tied embeddings share only the vocab table.
- **Interacts with output norm.** Many architectures put a LayerNorm just before the unembedding; tying works fine with this but the norm's scale parameter can drift large to compensate — worth watching.
- **MoE caveat.** onóna's 1B-MoE exception on loss suggests the router's token-dependent dynamics already provide some of the decoupling the second tied pair offers.
- **Gate collapse.** If $g$ learns to 0 or 1 everywhere, onóna reduces to a single tied pair; a small entropy regularizer on the gate helps.
- **Doesn't compose trivially with word-level tokenizers.** Tied embeddings assume the vocab is a learned set of pieces; char-level or byte-level LMs mostly don't need the trick because the table is already small.

## Sources

- Paper: *onóna: Twin Embeddings with Tied Unembeddings* — Dai, Zhou, Cai, Zhu, Zhang, Hu, 2026 — [arXiv:2611.10556](https://arxiv.org/abs/2611.10556)
- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — introduces the tied-embedding setup.
- Blog: *Using the Output Embedding to Improve Language Models* — Press & Wolf, 2017.

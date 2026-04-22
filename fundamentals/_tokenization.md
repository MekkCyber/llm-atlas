# Tokenization

*Taxonomy — how raw text is turned into the sequence of integer tokens a model actually sees.*

**TL;DR:** Models don't read text; they read token IDs. Tokenization is the frozen, up-front choice of *what counts as a unit*. Modern LLMs have converged on **learned subword tokenization** — most commonly byte-level BPE, sometimes Unigram or WordPiece — trained once on a representative corpus and shared by every downstream stage.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [bpe](bpe.md)

---

## The problem

A language model produces a distribution over a fixed vocabulary at every step. That forces a decision before any training starts: what is in the vocabulary? The options span a spectrum from characters (tiny vocab, very long sequences, word structure re-derived every forward pass) to words (short sequences, but vocab explodes, `<UNK>` problem, morphology invisible) to subwords (common words stay whole, rare ones decompose).

Whatever you pick becomes part of the model forever: tokenization is **frozen at pretraining time** and can't be changed without retraining. The choice has to be good at scale, across domains, across languages, and robust to weirdness (code, emoji, Unicode edge cases, binary data).

## The shared pattern

Every learned subword tokenizer shares the same three-stage recipe:

1. **Pre-tokenization** — a regex splits the raw string into pieces (on whitespace, digits, punctuation) before the algorithm sees anything. This is what keeps `"1234"` from merging into one bizarre token.
2. **Learning phase** — run the algorithm's objective (merge frequency, likelihood gain, pruning) over a representative corpus to produce a vocabulary.
3. **Apply phase** — a fast deterministic procedure (greedy matching, BPE merge replay) that converts any input string into its token sequence.

The variants differ in step 2's objective and in whether step 3 is greedy or allows sampling, but the shape is always the same.

Beyond the core algorithm, every tokenizer also has to decide:

- **Byte-level vs. Unicode-level** — does the base alphabet span bytes (any string representable, no `<UNK>`) or Unicode codepoints (cleaner, but needs a fallback)?
- **Normalization** — NFC/NFKC, case, whitespace collapsing.
- **Special tokens** — `<bos>`, `<eos>`, `<pad>`, chat templates (`<|im_start|>`), tool-call delimiters.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [BPE](bpe.md) | Iteratively merge the most frequent adjacent pair | Purely frequency-driven, no probabilistic model | General-purpose default; GPT-2/3/4, LLaMA, most modern decoder-only LLMs |
| Byte-level BPE | BPE over raw bytes instead of Unicode codepoints | 256-symbol base vocabulary is noisy for non-Latin scripts | Guaranteed coverage of any input (no `<UNK>`); GPT-2 onward |
| WordPiece | Merge-based, pick merges that maximize data likelihood | Close cousin of BPE, mostly historical now | BERT-family encoder models |
| Unigram LM | Probabilistic subword model; prune vocab to maximize data likelihood | Slower train/apply; supports subword sampling | T5, ALBERT, XLNet; SentencePiece's default |
| Character | One token per character | Very long sequences | Specialized settings (DNA, music), not general LLMs |
| Word | One token per word | Vocab blow-up, `<UNK>` problem | Extinct for LLMs |

## How to choose

**For a modern LLM from scratch, use byte-level BPE with a large vocabulary** (~100k–200k merges). That's where the field has converged. The reasons:

- Byte-level means *no `<UNK>` token ever* — any input, any language, any binary data roundtrips through the tokenizer.
- BPE is fast to train, fast to apply, and well-understood across a decade of production deployments.
- A large vocab keeps sequences short (cheaper FLOPs, more context per token) at the cost of a bigger embedding table and softmax.

**Unigram LM (SentencePiece)** is the defensible alternative. It gives you principled subword sampling for regularization and tends to produce slightly more "linguistic" segmentations. Used in a minority of modern models.

**WordPiece** is mostly historical. BERT's influence means the literature is full of it, but new decoder-only LLMs almost always pick BPE.

**Character and word tokenization** are essentially extinct for LLMs. Character-level survives in specialized settings (byte-level models, music/DNA) but not for general text.

### Other knobs that matter more than they look

- **Vocab size.** Bigger vocab → shorter sequences (cheaper compute per token) but bigger embeddings + softmax (more memory, more rare-token data required to learn each entry well). Typical range: 32k (LLaMA 2) → 128k (LLaMA 3) → 200k+ (recent frontier models).
- **Pre-tokenization regex.** GPT-2's regex splits on whitespace, digits, and punctuation before BPE sees anything. Every modern tokenizer has a carefully tuned pre-tokenization pattern.
- **Digit handling.** Splitting numbers into single digits (LLaMA) dramatically improves arithmetic ability at the cost of slightly longer sequences.
- **Whitespace convention.** GPT-style "leading-space-as-part-of-token" (`" the"` is one token) vs. BERT-style "word-internal `##`" markers — small-looking choice with big downstream effects on how the model treats formatting.

## Adjacent but distinct

- **Embedding layer.** Tokenization produces integer IDs; the embedding layer maps IDs to dense vectors. Separate concern, handled inside the model.
- **Pure byte models** (no merges, each byte is a token). Different design point — much longer sequences, no learned vocab. Mostly research-only at LLM scale.

## Sources

- Paper: *Neural Machine Translation of Rare Words with Subword Units* — Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909 — BPE for NLP.
- Paper: *SentencePiece: A simple and language independent subword tokenizer* — Kudo & Richardson, 2018 — https://arxiv.org/abs/1808.06226
- Paper: *Subword Regularization* — Kudo, 2018 — https://arxiv.org/abs/1804.10959 — Unigram LM.
- Paper: *Google's Neural Machine Translation System* — Wu et al., 2016 — WordPiece.
- Library: tiktoken — OpenAI's fast BPE implementation — https://github.com/openai/tiktoken

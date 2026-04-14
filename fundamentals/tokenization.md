# Tokenization
*Taxonomy — how raw text is turned into the sequence of integer tokens a model actually sees.*

**TL;DR:** Models don't read text; they read token IDs. Tokenization is the frozen, up-front choice of *what counts as a unit*. Modern LLMs have converged on **learned subword tokenization** — most commonly BPE (or a byte-level variant), sometimes Unigram or WordPiece — trained once on a representative corpus and shared by every downstream stage.

**Related:** [attention](attention.md)

---

## The problem

A language model produces a distribution over a fixed vocabulary at every step. That forces a decision before any training starts: what is in the vocabulary? The options span a spectrum:

- **Characters** — tiny vocab, but sequences become very long, and the model has to learn word-level structure from scratch every forward pass.
- **Words** — short sequences, but the vocab explodes (languages have millions of word forms), rare words become `<UNK>`, morphology is invisible, and cross-lingual transfer is poor.
- **Subwords** — a middle ground: common words stay whole, rare ones decompose into frequent pieces.

Whatever you pick becomes part of the model forever: tokenization is **frozen at pretraining time** and can't be changed without retraining. So the choice has to be good at scale, across domains, across languages, and robust to weirdness (code, emoji, Unicode edge cases).

Beyond unit size, tokenization also decides:
- **Byte-level vs. unicode-level** — does the base alphabet span bytes (so *any* string is representable) or Unicode codepoints (cleaner, but needs a fallback)?
- **Pre-tokenization** — is text first split on whitespace/punctuation/digits via a regex, or does the algorithm see the raw string?
- **Normalization** — NFC/NFKC, case, whitespace collapsing.
- **Special tokens** — `<bos>`, `<eos>`, `<pad>`, chat templates (`<|im_start|>`), tool-call delimiters.

## Variants at a glance

| Technique | Key idea | When it wins |
| --- | --- | --- |
| [BPE](bpe.md) | iteratively merge the most frequent adjacent pair | general-purpose default; used by GPT-2/3/4, LLaMA (via tiktoken / SentencePiece) |
| Byte-level BPE | BPE over raw bytes instead of unicode codepoints | guaranteed coverage of any input (no `<UNK>`), used by GPT-2 onward |
| WordPiece | merge-based, but pick the merge that maximizes likelihood gain | used in BERT-family; close cousin of BPE |
| Unigram LM (SentencePiece) | probabilistic model over subwords, prune the vocab to maximize data likelihood | used in T5, ALBERT, XLNet; supports sampling multiple segmentations (regularization) |
| Character | one token per character | simple, language-agnostic, but very long sequences |
| Word | one token per word | extinct for LLMs — vocab blow-up and `<UNK>` problem |

## How to choose

**For a modern LLM from scratch, use byte-level BPE with a large vocabulary** (~100k–200k merges). That's where the field has converged. The reasons:

- Byte-level means *no `<UNK>` token ever* — any input, any language, any binary data roundtrips through the tokenizer.
- BPE is fast to train, fast to apply, and well-understood across a decade of production deployments.
- A large vocab keeps sequences short (cheaper FLOPs, more context per token) at the cost of a bigger embedding table and softmax.

**Unigram LM (SentencePiece)** is the defensible alternative. It gives you principled subword sampling for regularization and tends to produce slightly more "linguistic" segmentations. Used in a minority of modern models.

**WordPiece** is mostly historical at this point — BERT's influence means the literature is full of it, but new decoder-only LLMs almost always pick BPE.

**Character and word tokenization** are essentially extinct for LLMs. Character-level survives in specialized settings (some byte-level models, music/DNA) but not for general text.

### Other knobs that matter more than they look

- **Vocab size**: bigger vocab → shorter sequences (cheaper compute per token) but bigger embeddings + softmax (more memory, more rare-token data required to learn each entry well). Typical range: 32k (LLaMA 2) → 128k (LLaMA 3) → 200k+ (recent frontier models).
- **Pre-tokenization regex**: GPT-2's regex splits on whitespace, digits, and punctuation before BPE sees anything — this is what keeps "1234" from merging into a single bizarre token. Every modern tokenizer has a carefully tuned pre-tokenization pattern.
- **Digit handling**: splitting numbers into single digits (LLaMA's choice) dramatically improves arithmetic ability at the cost of slightly longer sequences.
- **Whitespace convention**: GPT-style "leading-space-as-part-of-token" (`" the"` is one token) vs. BERT-style "word-internal ##" markers — small-looking choice with big downstream effects on how the model treats formatting.

## Why the design choice matters

- **Tokenization is frozen at pretraining.** Changing it means retraining. This is why lab-internal tokenizer decisions are taken extremely seriously.
- **Tokenization affects measured capability**, not just efficiency. Arithmetic, coding, multilingual performance, and long-context cost are all sensitive to tokenizer design.
- **Cross-tokenizer comparisons are treacherous.** Two models with the same "context length" of 128k tokens may have very different actual text capacity depending on how their tokenizers compress.
- **Tokens are not words.** An average English word is 1.3–1.5 tokens in modern BPE; code is often 2–4× denser; Chinese/Japanese/Korean can be much worse without careful tokenizer design.

## Sources

- Paper: *Neural Machine Translation of Rare Words with Subword Units* — Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909 (BPE for NLP).
- Paper: *SentencePiece: A simple and language independent subword tokenizer* — Kudo & Richardson, 2018 — https://arxiv.org/abs/1808.06226
- Paper: *Subword Regularization* — Kudo, 2018 — https://arxiv.org/abs/1804.10959 (Unigram LM).
- Paper: *Google's Neural Machine Translation System* — Wu et al., 2016 — WordPiece.
- Library: tiktoken — OpenAI's fast BPE implementation — https://github.com/openai/tiktoken

# Byte-Pair Encoding (BPE)
*Depth — the iterative-merge subword algorithm that became the default LLM tokenizer.*

**TL;DR:** Start from a character (or byte) vocabulary. Count every adjacent pair in the training corpus, merge the most frequent pair into a new token, and repeat. After `N` merges you have a vocabulary of `|chars| + N` subword units where common words stay whole and rare words decompose into frequent pieces. Originally a 1994 data-compression algorithm, repurposed for NLP in 2016, now the tokenization scheme used by essentially every modern LLM.

**Prereqs:** none (foundational)
**Related:** [tokenization](_tokenization.md)

---

## What it is

A learned, deterministic mapping from strings to sequences of subword tokens, defined by:

1. A **base alphabet** — typically the 256 byte values (byte-level BPE) or the set of unicode characters in the corpus.
2. An **ordered list of merges** — pairs `(a, b) → ab`, learned from a training corpus.
3. A **pre-tokenization step** — a regex that splits raw text into candidate words before BPE runs.

Given new text, you pre-tokenize it into words, convert each word to its base symbols, then greedily apply the learned merges in order. The output is a sequence of integer IDs.

## How it works

### Training (learning the merges)

```
vocab = set of all base symbols in the corpus
merges = []

for step in 1..N:
    pairs = count frequency of every adjacent pair across the corpus
    (a, b) = the pair with highest frequency
    merges.append((a, b))
    vocab.add(a + b)
    replace every occurrence of (a, b) in the corpus with the new symbol (a+b)
```

The corpus is usually represented as a bag of words with counts, not as raw text, so the pair-counting doesn't cross word boundaries. That's what **pre-tokenization** enforces: split the input on whitespace / punctuation / digits (using a regex) so BPE only merges *within* candidate words.

### Inference (applying the merges)

For each pre-tokenized word:
1. Split into base symbols.
2. For each merge `(a, b) → ab` in the order learned, replace adjacent occurrences of `a b` with `ab`.
3. Emit the resulting sequence of symbols (look up integer IDs).

The order matters. A merge learned at step 1 is applied before a merge learned at step 100.

### End-of-word marker

To prevent merges from crossing word boundaries in subtle ways, the original paper appended a special symbol (`·` or `</w>`) to the last character of each word during training. GPT-2 replaced this with a cleaner convention: **a leading space is part of the token**. So `" the"` (space-the) and `"the"` (sentence-initial) are different tokens, which lets the tokenizer recover spacing exactly during decode.

### Byte-level BPE

The 2019 innovation (GPT-2): run BPE over **bytes**, not unicode characters. The base alphabet is always the 256 byte values. This means:

- **No `<UNK>` token ever** — every possible input is representable as a sequence of bytes.
- **Language-agnostic** — works for any script.
- **Byte-pair is still learned**, so common unicode characters and words still end up as single tokens.

GPT-2 through GPT-4, LLaMA (some variants), and most recent LLMs use byte-level BPE.

### Joint BPE

For bilingual / multilingual training: learn merges on the **concatenation** of all languages' corpora. This produces a shared vocabulary. Benefits:

- Names, numbers, and loanwords tokenize consistently across languages.
- Models can learn to copy unknown tokens from source to target in translation.
- A single model head works for all languages.

Joint BPE is the default now for multilingual models.

## Why it matters

- **Open vocabulary with bounded size.** Any string can be encoded; rare strings just become longer token sequences. No `<UNK>` problem, no hand-crafted dictionaries.
- **Frequency-adaptive.** "the", "and", "function" each get one token; "antidisestablishmentarianism" gets many. Token count tracks information content, which is what you want for a fixed-compute model.
- **Deterministic and reversible.** Encode → decode roundtrips exactly (provided byte-level or careful unicode handling). Reproducibility across systems is trivial.
- **Fast.** Applied as a cached lookup per pre-token, with the merge graph precompiled. tiktoken runs at ~1M tokens/sec.
- **Became the de facto standard.** Every major frontier lab ships a BPE-family tokenizer. Changes to the tokenizer are a significant lab decision (LLaMA 2 → LLaMA 3 was a tokenizer upgrade).

## Gotchas & tricks

- **Pre-tokenization regex is load-bearing.** Without it, BPE will happily merge across word boundaries and produce bizarre tokens (`"the quick"` → one token). GPT-2's regex is famously copied and extended across the ecosystem.
- **Digit handling.** If you let BPE merge digits, you get tokens like `"1234"` or `"2023"` as single units, and the model's arithmetic is terrible. LLaMA explicitly splits digits (each digit = one token). GPT-4 uses a compromise. This is one of the highest-leverage tokenizer decisions.
- **Vocab size tradeoffs.** Bigger vocab → shorter sequences (cheaper compute) but bigger embedding + output layer (more memory, more data needed to learn each rare token). Typical: 32k → 128k → 200k for frontier models.
- **"Glitch tokens."** Rare tokens that ended up in the vocabulary but barely appeared in training can have garbage embeddings — feeding one causes bizarre behavior. SolidGoldMagikarp is the famous example. This is a hazard of large vocabularies + insufficient data for rare merges.
- **Training corpus matters.** If you train BPE on web text, you'll get different merges than if you train on code. Modern tokenizers are trained on a carefully balanced mix (text + code + math + multilingual).
- **Leading space semantics are brittle.** "word" and " word" are different tokens in GPT-2-style tokenizers; getting prompt formatting wrong by one space changes what the model sees.
- **Applying merges naïvely is O(N · L)** per word. Real implementations use priority queues or precompiled lookup structures (tiktoken).
- **Tokenization is not bijective with text at the boundary.** Trailing partial characters in streamed decoding need to be buffered — a token may map to half a unicode codepoint in the middle of a multi-byte character.

## Sources

- Paper: *Neural Machine Translation of Rare Words with Subword Units* — Sennrich, Haddow, Birch, 2016 — https://arxiv.org/abs/1508.07909
- Original algorithm: Gage, Philip. *A New Algorithm for Data Compression*, C Users Journal, 1994.
- Byte-level BPE: *Language Models are Unsupervised Multitask Learners (GPT-2)* — Radford et al., 2019 — Section 2.2.
- Library: tiktoken — OpenAI's BPE implementation — https://github.com/openai/tiktoken
- Library: Hugging Face tokenizers — https://github.com/huggingface/tokenizers

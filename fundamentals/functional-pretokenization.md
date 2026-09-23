# Functional Pre-tokenization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A lossless *pre-tokenizer* that factors orthographic variation (case, diacritics, character repetition) out of a token as a compositional stream of **opcodes** applied to a canonical **operand**. `hello`, `Hello`, `HELLO`, `Héllo` all share one vocabulary entry (`hello`) prefixed by different opcodes. Reduces vocabulary slot requirements by up to 19.7% at equal coverage, without lossy normalization.

**Prereqs:** [bpe.md](bpe.md), [_tokenization.md](_tokenization.md)
**Related:** *(none in-repo yet)*

---

## What it is

Every subword tokenizer (BPE, Unigram, WordPiece) faces the same choice for orthographic variants of the same word:

1. **Treat as distinct tokens** — fragments the embedding space; `hello` and `HELLO` learn separately.
2. **Normalize away** — apply lowercasing or NFC and strip diacritics; loses information the model needs (`WHO` vs. `who`, `resume` vs. `résumé`).

Functional pre-tokenization takes a third path: **factor** the variation into a separate stream of reversible transformation operators, encoded as **opcode tokens** in the Unicode Private Use Area, before running the base subword tokenizer on the *canonical* form.

## How it works

Pipeline (input string → tokens):

```
input  →  factor(orthographic vars)  →  opcode stream + canonical operand  →  BPE/Unigram
```

**Opcode families defined by the source paper:**

| Family | Operators | Example |
| --- | --- | --- |
| Casing | `CAPITALIZE`, `UPPER` | `Hello` → `[CAPITALIZE] hello` |
| Diacritics | 13 dedicated opcodes | `résumé` → `[ACUTE E] resume [ACUTE E]` |
| Repetition | `REPEAT`, `MULTIREPEAT` | `hellooo` → `[REPEAT 2 o] hello` |

Opcodes are encoded as single characters in the Unicode Private Use Area so they carry through the base tokenizer as regular vocabulary entries.

**Reversibility** is guaranteed by design — every opcode has a well-defined inverse, and applying opcodes to the operand recovers the original string exactly. There's no lossy normalization; case, diacritics, and repetition are preserved bit-for-bit.

**Base tokenizer stays the same.** BPE, Unigram, WordPiece all work unchanged — they now see a shorter canonical vocabulary + a small opcode alphabet.

## Why it matters

- **Shared embeddings for orthographic variants.** `hello` / `Hello` / `HELLO` / `Héllo` all reuse the same operand embedding, letting a token benefit from all its casings' training signal.
- **Smaller vocabularies at equal coverage.** Up to **-19.7%** vocabulary slot requirements under unconstrained-exhaustion conditions. Smaller embedding table → cheaper memory and faster LM head.
- **Better downstream generation.** On 98M-param GPT-2 runs: **Python syntax validity 9.12% vs. 7.70%** for baseline; lower duplicate n-gram repetition in prose — plausibly because the operand vocabulary is denser and the opcodes are structured.
- **Multilingual leverage.** Languages with rich case/diacritic systems (French, Turkish, German, Vietnamese) collapse into a small opcode alphabet — no dedicated Cyrillic/Latin duplicates, no case explosion.

## Gotchas & tricks

- **PUA collisions.** Using Unicode Private Use Area for opcodes assumes no upstream corpus uses those code points. Validate this on your dataset — some corpora (Klingon, private fonts) do use PUA glyphs.
- **Opcode inflation for pathological inputs.** A string that is nothing but repeated capitalized diacritics will have as many opcode tokens as characters. In practice this is rare; the paper reports net token-count reduction across natural language and code.
- **Not all normalization is orthographic.** Unicode compatibility (NFC/NFKC) collapses fullwidth vs. halfwidth digits, ligatures, etc. — an orthogonal concern the Functionalizer doesn't address.
- **BPE merges over opcodes.** If you let BPE learn merges *including* opcodes, common opcode+operand pairs (e.g. `[CAPITALIZE] the`) may re-fuse into single tokens. This is fine — the factorization is only used at pre-tokenization time; downstream merges are the tokenizer's business.
- **Reproducibility depends on the operator set.** Since the trick is a compositional operator alphabet, two teams can implement compatible tokenizers only if they agree on the opcode set. Not yet standardized.

## Sources

- Paper: *The Functionalizer: Lossless Functional Decomposition for Subword Tokenization* — Makowski, Guter, 2026 — introduces the opcode/operand pre-tokenization scheme. [arXiv:2609.15991](https://arxiv.org/abs/2609.15991)
- Background: Sennrich et al. (2016) — BPE for NMT (the base tokenizer this wraps).
- Background: Unicode normalization forms (NFC/NFKC) — the classical lossy alternative this replaces.

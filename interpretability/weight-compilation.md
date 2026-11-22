# Weight Compilation

*Depth — write deterministic exact circuits directly into a pretrained model's weights, with no gradients and no training data.*

**TL;DR:** A pretrained LM can be augmented with capabilities by *compiling* a circuit into its weights, in closed form. No gradient descent, no held-out data, no runtime hooks: an automated compiler synthesizes a sparse weight patch that implements, say, signed multi-digit arithmetic, and the patched model answers those operations deterministically while retaining its natural-language ability. A capability-insertion mechanism orthogonal to SFT, LoRA, and RL.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

The dual of mechanistic interpretability. Mech-interp *recovers* circuits from trained weights; weight compilation *writes* a chosen circuit into them. The target capability is specified as a symbolic program (arithmetic, Boolean / majority / parity circuits, lookup tables), and a compiler produces a small sparse update that implements the program on the model's native representation.

The result is a **deterministic** add-on: given valid inputs, the compiled capability produces the exact correct output. Unlike fine-tuning, there is no distribution over outputs — it's a hardcoded circuit co-located with learned weights.

## How it works

Three pieces:

1. **Encoding scheme.** Choose how operands and results are represented in token embeddings and residual stream (digit-wise one-hot, bit-serial, place-value).
2. **Compilation.** The compiler expresses the target operation as linear/affine maps and attention patterns, then writes those maps into specific sparse locations in the target model's weight tensors.
3. **Sparse sidecar.** The patch is stored sparsely — only the non-zero weight deltas — and applied additively to the base model at serving time. Memory footprint is small.

The claim about preserving NLP ability rests on the patch being **sparse** and placed in locations that minimally overlap with the model's dense learned pathways; the compiler is designed to leave existing circuits structurally intact.

The 2026 paper evaluates this on IBM Granite 4.2 3B and Alibaba Qwen 3.5 4B, reporting high arithmetic accuracy on compiled operations and reports said to beat LoRA on the same tasks, with base NLP behavior intact.

## Why it matters

Three distinct consequences:

- **A gradient-free SFT alternative** for capabilities that can be expressed symbolically. The model becomes reliable on exact arithmetic without a tool interpreter.
- **A new story for interpretability.** Compiled circuits co-exist with learned ones inside the same weights — useful for benchmarking mech-interp methods (does the method find the planted circuit?) and for testing circuit-extraction tools.
- **A path to hybrid symbolic-neural systems** that keep one model for both: deterministic operations compiled in, fuzzy natural-language operations learned.

## Gotchas & tricks

- Works only for **capabilities expressible as finite deterministic circuits.** Statistical patterns, open-ended tasks, generalization beyond the compiled program — not covered.
- The encoding scheme has to match the model's tokenizer and representation. Compiling for a different tokenizer requires re-deriving the encoding.
- "Does not disturb NLP ability" is an empirical claim; measure it on your own evals before deployment, since patch locality is approximate.
- The sparse sidecar composes additively with LoRA and with other compiled patches — multi-capability stacks are possible in principle.
- A planted circuit is a distribution-shift stress test for safety tools: if mech-interp or circuit audits miss a compiled circuit, they will miss adversarial ones too.

## Sources

- Paper: *A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights* — 2026 — [arXiv:2611.26450](https://arxiv.org/abs/2611.26450). Evaluated on IBM Granite 4.2 3B and Alibaba Qwen 3.5 4B.

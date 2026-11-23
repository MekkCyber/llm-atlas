# Circuit Compilation
*Depth — compile deterministic algorithms directly into a pretrained LLM's weights with no gradients and no training examples.*

**TL;DR:** Treat a pretrained transformer's forward pass as a programmable substrate and *compile* a target algorithm (exact integer arithmetic, Boolean parity) into its weights in closed form. The resulting model executes the algorithm on prompts it formerly got wrong — no training data, no backprop, no runtime hooks. On 6-digit arithmetic, closed-form compilation beats a 12.8K-sample LoRA.

**Prereqs:** [README](README.md)
**Related:** [fine-tuning README](../post-training/fine-tuning/README.md) · [activation-steering](activation-steering.md)

---

## What it is

The inverse of circuit *discovery*. Mechanistic interpretability has spent years reading circuits out of models; circuit compilation writes them in. Given:

- a pretrained LLM (e.g. IBM Granite 4.2 3B, Qwen 3.5 4B),
- an algorithm expressible as a bounded-depth arithmetic/Boolean circuit (signed multi-digit $+$, $-$, $\times$; AND/OR/XOR/parity),

produce a **closed-form weight edit** that makes the model execute the circuit exactly on inputs presented in a fixed prompt template. No gradient optimization. No training set.

## How it works

Three ingredients:

1. **Target circuit in canonical form.** Express the algorithm as a layered Boolean / arithmetic circuit with known depth and fan-in. The compiler needs an explicit computation graph, not just a specification.
2. **Weight-space construction.** Each gate is instantiated as a hand-designed sub-block of MLP (and optionally attention) weights. Standard pieces: digit-wise one-hot embeddings read from token embeddings, carry propagation implemented with MLP ReLUs, final digit-wise logits added to the unembedding.
3. **Factorized sparse sidecar.** The circuit lives in a low-rank, sparse add-on to the existing weights so the base model's capabilities stay intact. Only the sidecar's forward is new; the host forward pass is unchanged elsewhere.

Signed multi-digit $+, -, \times$ and arbitrary Boolean / parity on IBM Granite 4.2 3B and Alibaba Qwen 3.5 4B are reported to compile this way. The sidecar's memory cost is measured and bounded.

## Why it matters

- **Sidesteps the "LLMs can't do arithmetic" plateau.** LoRA / SFT closes some of this gap, but needs data and leaves tail bugs. Compilation produces *exact* behavior by construction.
- **Capability insertion as a design discipline.** Treating the model as a substrate for deterministic subprograms points toward a hybrid stack where symbolic components live inside the weights rather than as external tools.
- **A new baseline for fine-tuning papers.** When a deterministic task admits circuit compilation, that is the ceiling fine-tuning is trying to approach from below.
- **A fresh handle for safety.** Since the sidecar is interpretable by construction (every weight has a gate-level meaning), circuit-compiled subroutines are a candidate for high-assurance components inside otherwise opaque LLMs.

## Gotchas & tricks

- **Input format is part of the compile target.** The compiler assumes a fixed prompt template (digit layout, separators). Off-template inputs degrade to the base model's behavior.
- **Sidecar adds parameters.** It is not literally gradient-free *because weights are the same* — it modifies the model's parameter set. The "zero-gradient" claim is about the training procedure, not about weight count.
- **Does not generalize for free to list algorithms.** Depth-bounded straight-line circuits compile cleanly; algorithms needing dynamic iteration or data-dependent loops may require token-level loop unrolling and may not fit the forward pass.
- **Interacts with existing LoRAs.** A sidecar and a LoRA both perturb the forward pass; the paper doesn't exhaustively characterize their interaction. Treat as additive and test.
- **Tokenization fragility.** Digit tokenization (single digits vs multi-digit tokens) changes the circuit design. Models with digit-splitting tokenizers are easier targets.

## Sources

- Paper: *A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights* — 2026.
- Related background: *Circuit discovery* papers (Olah et al., Nanda & Lieberum) — the reverse direction.

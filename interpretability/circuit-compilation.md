# Circuit Compilation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hand-design an exact Boolean or arithmetic circuit (adder, multiplier, lookup table) and **compile it directly into pretrained LLM weights** as a closed-form edit, without gradient training or external tools. The patched model executes the circuit with high accuracy — e.g. 6-digit arithmetic — keeps its natural-language benchmark scores, and seamlessly resumes free-form generation after producing the answer.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [self-editing-llms.md](self-editing-llms.md), [activation-steering.md](activation-steering.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

A compiler that takes an exact digital circuit (gates, wires, registers) and emits a set of weight edits — specific attention-head queries/keys, MLP up/down projections, biases — that implement the circuit inside a chosen slice of a pretrained transformer. Unmodified layers keep their language-modelling behaviour; the compiled block activates on structured inputs (e.g. "`123456 + 789012 =`") and routes back to natural-language generation after emitting the answer.

## How it works

1. **Circuit spec.** Define the target computation as a Boolean/arithmetic DAG — ripple-carry adder, 2-input multiplier, $n$-bit comparator.
2. **Allocation.** Pick a subset of attention heads and MLP neurons to repurpose in the fewest possible blocks; the rest of the model is untouched so language ability is preserved.
3. **Weight synthesis.** For each gate, solve a small linear system: the attention pattern reads operand tokens, the MLP implements the gate's truth table via a piecewise-linear nonlinearity, biases shift activations into the usable range of the subsequent nonlinearity.
4. **Routing.** Add detection circuitry so the compiled block fires only on the expected input pattern; off-pattern inputs pass through unchanged.
5. **No training.** Everything is closed-form — no gradient descent, no calibration data.

## Why it matters

- **Exact semantics.** Unlike learned arithmetic (which hallucinates in the long-digit regime), compiled circuits are provably correct on in-range inputs.
- **Capability patching without fine-tuning.** Install arithmetic / logic into a model without touching most of its weights or its training pipeline.
- **Inverts the interp game.** Classical mech-interp *discovers* emergent circuits; circuit compilation *installs* a chosen one, giving a ground-truth probe for the discovery tools.
- **Stacks with self-editing (He et al., 2026).** If an LLM can emit closed-form edits, the compiler becomes something the model itself could call.

## Gotchas & tricks

- **In-distribution only.** The circuit fires on exact input patterns; adversarial formatting (unexpected tokenisation of digits, mixed-language prompts) can route around it.
- **Precision ceiling.** Floating-point weight storage limits the operand width before carries corrupt. The paper reports clean 6-digit arithmetic.
- **Backbone cost.** Allocating heads and neurons to the circuit removes them from language tasks; benchmark neutrality depends on using a small slice of a sufficiently large model.
- **Composition is hard.** Multiple compiled circuits in the same model share a residual stream; isolation requires careful allocation of non-overlapping subspaces.

## Sources

- Paper: *A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights* — 2026 — primary source; 6-digit arithmetic and Boolean-logic compilation, with retained natural-language performance.
- Related: *Causal Mediation Analysis* / ROME / MEMIT — discovered-circuit editing, the inverse of this technique.

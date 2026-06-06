# Arithmetic Geometry in LLM Activations
*Depth — the residual-stream geometry that LLMs use to perform multi-operand addition, and how it explains both their errors and how to detect them.*

**TL;DR:** When an LLM performs multi-digit addition, the residual stream organizes itself along an **Iso-Raw-Sum Trajectory (IRST)**: representations are anchored by discrete *semantic-digit* positions and modulated along continuous *carry fibers*. Arithmetic errors are **geometric slippages** — internal noise pushes a continuous carry potential across a quantization threshold, flipping the predicted digit. Lightweight linear probes can read off both the truth and the hallucinated answer from the same activation vector ("probe versatility"), enabling an inference-time consistency check that detects and corrects failed additions.

**Prereqs:** [interpretability/README](README.md)
**Related:** [post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

The puzzle: LLMs that solve hard math word problems still get basic multi-digit addition wrong with curious regularity. The fragility is *too* structured to be random noise — small operands work, certain digit positions fail more than others, errors are systematic.

The Shape-of-Addition paper (2026) argues this is *geometry*. The residual stream during arithmetic exhibits a low-dimensional structure:

- **Semantic-digit anchors.** Discrete clusters in activation space, one per possible digit value at a given position.
- **Carry fibers.** A continuous 1-D direction connecting "no carry" and "carry" states for each digit position.
- **Iso-Raw-Sum Trajectory (IRST).** As inputs vary while the raw column sum stays constant, the representation moves along a specific manifold — separating the *raw-sum* axis from the *carry-resolved* axis.

## How it works

The Noisy Quantization Model:

```
raw column sum  →  point on a continuous carry potential
   + neural noise
   → snapped to nearest quantized digit by the LM head
```

When the carry potential is close to a threshold (e.g., raw sum = 9 or 10), noise pushes it across, producing a *geometric slippage* — the model outputs the wrong digit not because it didn't "know," but because the continuous latent crossed a discrete boundary.

**Probe versatility.** A single activation vector contains multiple latent signals (truth, hallucination, partial state) that *lightweight linear probes* can disentangle. Trained on labeled examples, probes recover:

- Ground-truth digit
- Hallucinated digit
- Carry-state direction

…all from the same residual-stream slice.

**Inference-time consistency check.** At each digit position, probe both the model's emitted answer and the "carry-potential" reading. Inconsistency → flag and correct (re-sample, edit, or override). The correction is cheap (one probe per layer, deterministic).

## Why it matters

- **Mechanistic explanation for a famous LLM weakness.** Arithmetic fragility is not "the model doesn't know math" — it's a *thresholding artifact* with a precise geometric description.
- **Bridges interp and capability.** Most mech-interp work explains *behavior*; this paper produces an inference-time *fix* from the explanation.
- **Probe versatility is a transferable observation.** That a single residual vector can be linearly decoded into multiple co-existing facts (truth + hallucination) suggests SAE-free shortcuts for many "is the model lying?" questions.

## Gotchas & tricks

- **Generalizes across models in spirit, not in specifics.** The fiber structure exists in multiple LLMs but probe weights don't transfer — retrain per model.
- **Works on raw arithmetic, not word problems.** When addition is embedded in a long CoT, the residual stream carries more state and the clean IRST blurs. Pre-extract the arithmetic substep.
- **Layer choice matters.** Mid-layer residuals (where the carry computation lives) probe better than late layers (where the LM head has committed).
- **Not the same as "circuits for addition."** Earlier mech-interp work (Nanda et al. on modular addition) identified Fourier-style features; IRST is a complementary, larger-scale geometric description on natural decimal addition.
- **A correction *layer*, not a *training* fix.** This is inference-time mitigation, not a recipe for training models that don't slip in the first place.

## Sources

- Paper: *Geometric Structures of Arithmetic in Large Language Models* (the Shape of Addition) — Anonymous (CN academic) authors, 2026 — [arXiv:2606.03645](https://arxiv.org/abs/2606.03645) — primary source.
- Paper: *Progress measures for grokking via mechanistic interpretability* — Nanda et al., 2023 — earlier circuit-level account of modular addition.

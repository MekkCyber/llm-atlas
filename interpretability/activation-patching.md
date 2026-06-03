# Activation Patching
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A causal-mediation technique from mechanistic interpretability. Run the model on two prompts (a "clean" run and a "corrupted" run); cache activations from the clean run; then re-run the corrupted prompt while **patching** (swapping in) the clean activations at one location — a single layer's residual stream, one attention head's output, one MLP neuron. If the patched output flips back toward the clean answer, that location was *causally responsible* for the behavior. Standard primitive behind circuit discovery, model editing, and (more recently) **unlearning audits** like UDS.

**Prereqs:** [README](README.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../safety/README.md](../safety/README.md)

---

## What it is

Correlational probes ("this layer's representations correlate with concept X") leave the question of *causation* open. Activation patching addresses this directly: it intervenes on internal activations and measures the downstream effect. The standard recipe:

1. Pick two prompts that differ in a controlled way — e.g. clean = "The Eiffel Tower is in" (predicts "Paris"); corrupted = "The Statue of Liberty is in" (predicts "New York").
2. Run the model on both, caching every activation in the clean run.
3. Re-run the corrupted prompt, but at one chosen location, replace the corrupted activation with the cached clean activation.
4. Measure how much the output logits move from the corrupted answer ("New York") toward the clean answer ("Paris").

The fraction of the gap that's recovered is the **patching effect** at that location.

---

## How it works

### Standard patching variants

- **Residual stream patching:** replace the residual stream at layer $\ell$, position $p$. Coarsest but cheapest; tells you *where* in the depth/sequence axis the relevant signal lives.
- **Attention-head patching:** replace one head's output (post-projection but pre-residual-add). Identifies which specific heads carry the signal.
- **MLP / neuron patching:** replace single neurons in an FFN's activations. Most fine-grained.
- **Path patching** (Wang et al., 2023): patch one component while *also* patching all downstream components from the corrupted run — isolates the component's *direct* causal effect from its indirect effects through later layers.

### Metrics

- **Logit difference recovery:** `(logit_correct - logit_incorrect)` measured on the patched run, normalized so 0 = corrupted baseline, 1 = clean baseline.
- **KL divergence to clean distribution:** for distributional rather than logit-difference settings.
- **Output flip rate:** for binary tasks, the fraction of patched runs where the argmax flips back to the clean answer.

### Cost

One forward pass per patching location per prompt pair. For a 32-layer model at 100 positions, exhaustive residual-stream patching is ~3200 forward passes per (clean, corrupted) pair. Scaled with **attribution patching** (a first-order Taylor approximation that needs only one forward + one backward pass) — much cheaper but slightly biased.

---

## Why it matters

- **Causal vs correlational.** Probes tell you *what's encoded*; patching tells you *what's used*. The difference matters enormously for safety claims — a probe finding "the model represents X" doesn't imply the model *acts on* X.
- **The substrate of circuit discovery.** Mechanistic-interpretability circuits (induction heads, IOI circuit, factual-recall circuits) were all identified by some form of activation patching. It's the workhorse primitive.
- **Generalizes beyond research.** UDS (Lee et al. 2026, *Measuring the Depth of LLM Unlearning via Activation Patching*) operationalizes patching as an **audit metric**: patch retain-model activations into an unlearned model and measure how much of the to-be-forgotten knowledge resurfaces. A causal score for "did the unlearning actually erase the representation, or just suppress the output?"
- **Compatible with model editing.** ROME and MEMIT identify edit locations via patching and then write back targeted weight changes.

---

## Gotchas & tricks

- **Clean/corrupted prompts must match in length and grammar.** Otherwise position alignment fails and patches inject noise rather than signal.
- **Patching effects compose unintuitively.** Patching one head can have a big effect *because of* the head two layers downstream that consumes its output. **Path patching** is the right tool when you care about a component's direct effect.
- **Attribution patching is biased but cheap.** A first-order Taylor approximation; use it for screening, then verify hits with exact patching. Don't make causal claims from attribution patching alone.
- **Distribution shift confounds.** If clean and corrupted prompts differ in more than the intended variable, the patched effect can pick up the unintended axis. Tightly controlled prompt pairs are essential.
- **Layer normalization matters.** Patching activations *before* a LayerNorm gives different results than patching after — the normalization redistributes the signal. Be explicit about the patching site.
- **Models with shared embeddings (e.g. MoE with shared experts) have routing-dependent activations.** Patching at an MoE layer requires also patching the router's decisions, otherwise the patched activation goes to the wrong experts.
- **Causal mediation analysis is the broader statistical framing** — Pearl-style do-calculus interventions. Activation patching is the LLM-shaped special case.

---

## Sources

- Paper: *Locating and Editing Factual Associations in GPT (ROME)* — Meng, Bau, Andonian, Belinkov, NeurIPS 2022 — popularized causal patching as the analysis substrate for factual-recall circuits.
- Paper: *Interpretability in the Wild: a Circuit for Indirect Object Identification* — Wang, Variengien, Conmy, Shlegeris, Steinhardt, ICLR 2023 — introduces path patching.
- Paper: *Attribution Patching: Activation Patching at Industrial Scale* — Syed, Rager, Conmy, Neel Nanda blog, 2023 — the gradient-based approximation.
- Paper: *Measuring the Depth of LLM Unlearning via Activation Patching (UDS)* — Lee, Kim, Jo, 2026 — [arXiv:2605.24614](https://arxiv.org/abs/2605.24614) — operationalizes patching as an unlearning audit metric.
- Tooling: TransformerLens (Nanda et al.) — the standard mechinterp library; ships activation-patching utilities.

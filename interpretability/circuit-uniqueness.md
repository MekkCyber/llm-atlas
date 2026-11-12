# Circuit Uniqueness Audit
*Depth — a protocol that tests whether an identified "circuit" is actually the unique causal path or just one of many.*

**TL;DR:** Mechanistic-interpretability work routinely identifies a small neuron group as "the circuit" for a behavior, then shows ablating it breaks the behavior. This paper argues *necessity* and *uniqueness* are different claims, often conflated. Exhaustively enumerating every subset of a small neuron group on Qwen3 / Llama-3.2 / Gemma-3, the authors find small *disjoint* subsets that each flip the model's answer — so the group is not the unique causal path, and some ablation signals turn out to be near-constant activations, not genuine necessity.

**Prereqs:** [README.md](./README.md).
**Related:** [README.md](./README.md).

---

## What it is

A methodology for auditing claims about causal neuron circuits. For a group of $n$ neurons implicated in some behavior (here $n = 12$), run all $2^n$ subset experiments rather than only the all-or-nothing ablation. Two canonical probes:

- **Zero-ablation** on each subset — does the behavior break?
- **Activation patching** — swap the subset's activations from a different-prompt run; does the answer change?

Compare "group-level" effects to disjoint-subset effects: if two disjoint subsets each flip the answer, the group is not uniquely causal.

## How it works

Experimental template (per behavior):

1. Pick a candidate group of neurons from any existing circuit-discovery method.
2. For each of its $2^n$ subsets:
   - Measure margin change under zero-ablation.
   - Measure margin change under activation patching from alternative prompts.
3. Classify each subset:
   - *Necessary*: ablation breaks behavior.
   - *Transferable*: patching flips answer to the alternative prompt's target.
   - *Near-constant*: activations are nearly the same across prompts — patching does nothing, but ablation still breaks because the information was the constant itself.
4. Check whether disjoint subsets both pass the same test. If yes, the circuit is non-unique.

## Why it matters

- **Separates necessity from uniqueness.** A behavior-breaking ablation is evidence of necessity for *some* path; it says nothing about whether other paths exist. The uniqueness claim is what most "we found the circuit for X" headlines actually imply.
- **Near-constant activation pitfall.** On the capital-city task, the 12 neurons' activations vary by <5% across prompts. Zero-ablation looks like a big necessity signal (−27% margin), but patching from other prompts barely changes the output — because the "information" was just a constant bias, equivalent to a learned threshold.
- **Transferability caveat.** The authors show two-hop chain patches transfer to *paraphrases* of the same question but not to other chains, suggesting the circuit is behavior-specific but not task-general.
- **Pushes field practice.** Any future "we found the X circuit" claim should carry a uniqueness audit — otherwise, "a circuit" is the right article, not "the circuit."

## Gotchas & tricks

- **Combinatorial blow-up.** Exhaustive over $2^n$ only works for small $n$ (≤ 20 or so). For bigger groups, use sampled-subset bounds; the uniqueness question is still answerable, with wider confidence intervals.
- **Near-constant detection is a one-line check.** Compute activation variance across the eval set first — if it's near-zero, the "circuit" may be a bias in disguise.
- **Patching flips to the alternative prompt's target** is the clean transferability signal. Patching flips to *nothing* (the answer collapses) is a weaker signal — it may just be out-of-distribution.
- **Scope is small MLPs.** The paper audits 12-neuron MLP groups on <2B models. The same protocol should be run on SAE features and attention heads too, but is not demonstrated here.

## Sources

- Paper: *Are Causal Neuron Circuits Unique and Transferable?* — Ömer Tekin Yavuz, 2026 — [arXiv:2611.13885](https://arxiv.org/abs/2611.13885).

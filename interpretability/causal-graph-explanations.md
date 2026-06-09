# Causal graphs of LLM inference (σ-CG + counterfactual chains)
*Depth — recover a concept-level causal graph whose nodes are concepts an LLM actually uses to make a prediction.*

**TL;DR:** Treat the LLM as the system under study, not a tool to study external systems. Discover human-interpretable concepts that discriminate classes, map each input to "LLM-perceived" concept states, then run causal discovery (σ-CG) on the resulting data — augmented with MCMC-inspired *counterfactual chains* — to recover a stable concept-level graph that predicts the LLM's behaviour.

**Prereqs:** [attention](../fundamentals/attention.md), [logit-lens](logit-lens.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

Most "LLM + causal graph" work uses LLMs as oracles to recover causal graphs of the *world*. This flips the direction: the causal graph models the **LLM's own reasoning** about a class of inputs (disease diagnosis, sentiment, judge classification). Nodes are class-discriminative, human-interpretable concepts the LLM is sensitive to; edges encode probabilistic dependencies in how the LLM combines them.

## How it works

A four-phase pipeline:

1. **Concept discovery.** From a small labelled set, extract candidate concepts that are class-discriminative *and* human-interpretable (LLM-assisted but human-auditable list).
2. **Concept-state mapping.** For each input, query the target LLM about whether each concept is present / which value it takes; produces a tabular dataset of inputs × concepts × LLM-perceived states.
3. **MCMC-inspired counterfactual augmentation.** Starting from observed inputs, walk chains of single-concept counterfactual edits ("flip exactly this concept, regenerate"). Each step requires querying the LLM to update perceived states. Expands the sparse observational data into something causal discovery can chew on.
4. **σ-CG causal discovery.** Run σ-CG (a constraint-based causal-discovery algorithm robust to small samples) on the augmented data. Output: a DAG of concepts with edge weights, where the graph predicts the LLM's class output as a function of the perceived concept states.

The graph is the artefact; it can be inspected, edited, and used to predict the LLM's behaviour on new inputs.

## Why it matters

- Concept-level explanations are what regulated domains demand. A faithful concept-DAG is auditable in a way that attention maps or token-attribution heatmaps are not.
- Black-box compatible — only requires API access to the target LLM. Complements white-box mech-interp (SAEs, probing) from the behavioural side.
- The counterfactual-chain trick is independently useful: a cheap data-augmentation primitive for any low-data causal-discovery setup.

## Gotchas & tricks

- **Concept curation is the ML work.** Garbage concepts ⇒ garbage graph. Iterating with humans on the concept set dominates wall-clock cost.
- **LLM-perceived states ≠ ground truth.** The graph reflects how the LLM *sees* concepts, which can diverge from the world. That's a feature for explanation but a bug if you read the graph as a world model.
- **σ-CG assumes faithfulness.** Spurious independences caused by the LLM's lossy concept perception can hide true edges; the chain augmentation helps but doesn't eliminate this.
- **Validation is two-pronged.** Predictive fidelity (does a tiny model that uses only the graph match the LLM on held-out data?) and structural stability across re-runs (Jaccard similarity of recovered edges).

## Sources

- Paper: *LLM Explainability with Counterfactual Chains and Causal Graphs* — Nussbaum-Hoffer, Calderon, Ein-Dor, Reichart — 2026 — [arXiv:2606.05972](https://arxiv.org/abs/2606.05972)

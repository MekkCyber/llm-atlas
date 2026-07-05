# Retrieval Heads

*Taxonomy — attention heads specialized for pulling information from earlier in the context to influence the next token.*

**TL;DR:** Long-context models don't uniformly attend to the past; a small set of "retrieval heads" carries most of the work. But *retrieval* is not one thing — copy heads move a token verbatim, induction heads pattern-match and copy, and **non-literal retrieval heads** transform semantically without copying. Different diagnostics find different subsets of these heads; picking the diagnostic that matches the failure you care about is the whole game.

**Related taxonomies:** [../architectures/_normalization.md](../architectures/_normalization.md)
**Depth files covered here:** [locos](locos.md)

---

## The problem

Long-context evaluation ("needle in a haystack," multi-hop QA, paraphrased retrieval) exposes very different behaviors. Some models fail on literal copy; others copy fine but can't paraphrase-then-answer. Attributing these failures to specific heads is the mechanistic-interp goal, but the standard diagnostic — "which head attends to the needle position?" — over-counts *copy* behavior and misses semantic transformation.

## The shared pattern

Every retrieval-head method has the same shape: pick a **task** where information from earlier in the context must reach the answer; score each head for its **contribution** to that task; verify by **ablating** the top-scored heads and measuring the drop. What differs is the scoring signal.

## Variants

| Technique | Key idea | Signal | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Attention-pattern retrieval heads | Rank heads by attention weight on the needle position | *Where* the head attends | Simple; conflates copying with retrieval | Copy-heavy tasks (verbatim recall) |
| Induction heads | Two-head circuit that pattern-matches recent tokens | Head-pair prevalence signature | Well-studied circuit; specific to induction | Explaining in-context learning |
| [LOCOS](locos.md) (logit-contribution scoring) | Score by how much each head's output projects onto the correct-token direction | *What* the head transmits | Isolates semantic transformation, more compute per head | Non-literal / paraphrased retrieval |
| Path-patching / activation-patching | Causally intervene on a head's output and measure downstream effect | Counterfactual behavior | Gold-standard causality; expensive at scale | Small models, targeted studies |

## How to choose

Reach for attention-pattern scoring first for a quick heat-map. Move to [LOCOS](locos.md) whenever the failure mode you care about is *non-literal* — paraphrase, summarize-then-answer, cross-language retrieval — because attention patterns will over-count copy heads and miss the transforming ones. Use path patching when you need causal evidence for a small circuit and can afford the compute.

## Adjacent but distinct

- **Sparse autoencoders (SAEs)** — decompose an activation into interpretable features, not into heads. Different granularity, same interp goal.
- **Probing** — trains a classifier on hidden states to detect a feature. Doesn't attribute to individual heads.
- **Logit lens** — projects intermediate residual streams through the unembedding to inspect predictions across depth. Related to LOCOS in tooling but scores layers, not heads.

## Sources

- Paper: *Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads (LOCOS)* — Gema, Alex, Minervini, 2026 — [arXiv:2607.01002](https://arxiv.org/abs/2607.01002).
- Paper: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — attention-pattern retrieval heads.
- Paper: *In-context Learning and Induction Heads* — Olsson et al., Anthropic, 2022 — the canonical induction-head circuit.

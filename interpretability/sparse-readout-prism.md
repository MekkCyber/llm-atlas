# Sparse Readout Prism (SRP)

*Depth — decomposing a lens's readout into sparse features so logit-lens scores can be explained in features, not tokens, and independently of the fitting corpus.*

**TL;DR:** A logit-lens reading is $\text{decode}(h_\ell; R)$ — a product of a hidden state and a readout $R$. Two lenses that differ only in the fitting corpus can return different tokens for the same $h_\ell$ (*corpus conditionality*). **Sparse Readout Prism (SRP)** (He et al., 2026) decomposes any readout — the unembedding matrix or a fitted lens — using only its weights (no corpus), so that every token logit / logit difference becomes an additive sum over **sparse readout features**. SRP's sparse approximation reconstructs **8.9–17.3 pp more of tested logit differences** than the strongest of six geometric baselines, and its **dominant readout features are stable across fitting corpora even when token readings aren't**, giving lens analyses a corpus-independent control.

**Prereqs:** [logit-lens](logit-lens.md)
**Related:** [../fundamentals/attention](../fundamentals/attention.md)

---

## What it is

A decomposition of a readout matrix $R \in \mathbb{R}^{|V| \times d}$ (e.g. the unembedding $W_U$, or a lens's fitted linear head) into a sparse feature basis, together with a machinery for expressing any token logit or logit difference as a sum over feature contributions. Constructed from $R$ alone — no corpus is used in the construction — so the resulting analysis unit ("readout feature") is stable across whatever corpus the lens itself was fit on.

## How it works

### Corpus conditionality — the motivating observation

Two tuned lenses fit on different corpora but sharing the same underlying model can output *different* top-1 tokens for the same $h_\ell$. That means top-token readings mix a property of the hidden state with a property of the corpus the readout was calibrated against — an analytic ambiguity.

Because SRP is constructed from readout weights only, it isolates the state's contribution from the corpus's contribution.

### Feature extraction from readout weights

Given $R$ with token rows $r_v \in \mathbb{R}^d$, SRP:

1. Discovers a **sparse dictionary** $\{f_j\}_{j=1}^{K}$ of directions in $\mathbb{R}^d$ from the geometry of $R$ (no training corpus enters).
2. Expresses each token row as $r_v \approx \sum_j c_{v,j} f_j$ with sparse coefficients $c_{v,j}$.
3. Uses the dictionary to write any logit $\ell_v = r_v \cdot h = \sum_j c_{v,j} (f_j \cdot h)$ as an additive contribution decomposition.

### Explaining a logit difference

For a token pair $(v_1, v_2)$ at hidden state $h$:

$$
\ell_{v_1} - \ell_{v_2} = \sum_j (c_{v_1, j} - c_{v_2, j}) \cdot (f_j \cdot h)
$$

Each summand is the contribution of feature $j$ to the token comparison. Sorting by magnitude gives the analyst the top few features that drive the model's preference at that layer / position.

### Two empirical validations

- **Reconstruction quality.** Replacing $R$ with its SRP-sparse approximation and re-decoding recovers **8.9–17.3 pp more of tested logit differences** than the strongest of six baselines built on geometric relations among readout rows.
- **Ablation matches attribution.** Ablating a top-contributing SRP feature shifts the observed logit difference in proportion to that feature's SRP contribution — the attribution is causal, not just descriptive.
- **Stability under corpus change.** The *dominant readout features* remain stable across the fitting corpora that made token readings differ — the corpus-conditional ambiguity lives in which tokens each feature happens to point at, not in the feature basis itself.

## Why it matters

- **Corpus-independent lens analysis.** Any lens-based claim — "layer 20 already encodes the answer", "head 6.3 writes to a rank-1 direction" — can now be re-derived through SRP as a control, decoupling the claim from the fitting corpus.
- **Feature-level unit of analysis for lens readings.** Before SRP, lens readings were per-token; per-token can conflate the readout and the state. Features let the analyst report "at layer 20 the dominant feature contribution is $f_{147}$ (which points at 'measurement units')" — a stabler and more mechanistic statement.
- **Cheap.** No corpus, no training. SRP is a linear-algebra pass over the readout weights that already sit in the model checkpoint.
- **Debugging tool for lens-based experiments.** When two labs report different lens readings, SRP tells them whether the disagreement is corpus-conditional (probably) or genuine (rarely).

## Gotchas & tricks

- **Feature naming is post-hoc.** SRP finds features geometrically; interpreting *what* $f_j$ means still requires probing (activation examples, top-token lists, ablations). SRP names the axis; naming the concept it points at is a separate step.
- **Dictionary hyperparameters matter.** Sparsity level and dictionary size $K$ shift the reconstruction accuracy / interpretability tradeoff. Report both — the "8.9–17.3 pp reconstruction" numbers are at the paper's chosen operating point.
- **Not a replacement for corpus-fit lenses.** SRP is a *control*, not a lens. A tuned lens on a domain-matched corpus still gives the sharpest token readings; SRP explains what that lens is doing feature-wise.
- **Doesn't fix hidden-state noise.** If the hidden state itself is noisy (e.g. very early layers before the residual stream has stabilized), SRP's decomposition inherits that noise.
- **Compare against direct logit attribution as a sibling.** DLA decomposes by *residual-stream writer* (which head / MLP wrote the direction); SRP decomposes by *readout feature*. Together they give a two-axis attribution.

## Sources

- Paper: *Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens* — Matteo He, William F. Shen, Xinchi Qiu, Nicholas D. Lane — 2026 — [arXiv:2609.01936](https://arxiv.org/abs/2609.01936) — University of Cambridge.
- Related: [logit-lens](logit-lens.md) — the family SRP analyzes.
- Related: *A Mathematical Framework for Transformer Circuits* — Elhage et al., 2021 — for direct logit attribution, SRP's sibling axis.

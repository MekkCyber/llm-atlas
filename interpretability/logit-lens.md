# Logit Lens

*Depth — decoding intermediate hidden states through the model's own unembedding matrix to see how a prediction develops layer by layer.*

**TL;DR:** For a transformer with residual stream $h_\ell$ at layer $\ell$ and unembedding matrix $W_U$, the **logit lens** decodes $\text{softmax}(W_U h_\ell)$ at every layer and reads off the model's "current best guess" as it flows through the stack. It is a workhorse of mechanistic interpretability — cheap, no training, gives a per-layer trajectory. Variants — **tuned lens** (fit a per-layer affine correction), **linear probe lenses**, corpus-fit lenses — all consist of *hidden state* × *readout*, and lens readings depend on both. The **Sparse Readout Prism** ([sparse-readout-prism](sparse-readout-prism.md)) formalizes this dependence as *corpus conditionality* and decomposes the readout to give a corpus-independent unit of analysis.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [sparse-readout-prism](sparse-readout-prism.md)

---

## What it is

A family of interpretability probes that reuse the model's own unembedding matrix (or a learned affine map onto it) to interpret intermediate residual-stream activations as distributions over tokens. Introduced informally by nostalgebraist (2020) for GPT-2 as a way to see "what the model thinks the next token is at layer $\ell$."

## How it works

### Vanilla logit lens

For a token position with residual-stream state $h_\ell \in \mathbb{R}^d$ at layer $\ell$:

$$
p_\ell = \text{softmax}(W_U \, h_\ell) \in \Delta^{|V|-1}
$$

Optionally with layer-norm applied first (matching the model's final-layer computation). The top-k tokens of $p_\ell$ are the "lens reading" at that layer. Comparing readings across $\ell$ shows how a prediction *develops* — early layers often output filler / most-common tokens, middle layers show ambiguous mixtures, late layers converge on the final answer.

### Tuned lens

Vanilla logit lens has a distribution mismatch: intermediate hidden states aren't distributed like the final-layer state $W_U$ was trained on. **Tuned lens** (Belrose et al., 2023) fits a per-layer affine transform $A_\ell h_\ell + b_\ell$ on a calibration corpus so the resulting distributions align with the model's final-layer statistics.

### Other readouts

- **Linear probes**: train a linear classifier on top of $h_\ell$ for a specific concept; the "readout" is the classifier weight, not $W_U$.
- **Direct logit attribution** (Elhage et al., 2022): decompose $W_U h_\ell$ into contributions from each residual-stream write (each attention head, each MLP layer).

### The lens-anatomy point

Any lens reading is $\text{decode}(h_\ell; R)$ where $R$ is the readout. Two lenses differing *only in $R$* — fit on different corpora, or with different regularization — will disagree on which token is on top for the same $h_\ell$. This *corpus conditionality* was named and quantified by SRP (2026); it explains earlier reports of "lens disagreement" that had no unified framing.

## Why it matters

- **Cheap and universal.** No training (vanilla), one linear fit (tuned), no per-behavior architecture. Applied to every open-weights LLM release.
- **The layer-by-layer trajectory is the object of study.** Many results — "induction heads emerge at layer $\ell$", "the model finalizes the answer by layer $L{-}3$" — are logit-lens results. The lens is the microscope; the trajectory is the specimen.
- **Base for many downstream methods.** Direct logit attribution, path patching, and activation steering all interpret intermediate states through some readout; the logit lens is the simplest instantiation.
- **Now decomposable.** Corpus-fit lenses have been the frontier for years; SRP argues that focus on features (not tokens) is more stable, extending the lens family with a corpus-free control ([sparse-readout-prism](sparse-readout-prism.md)).

## Gotchas & tricks

- **Vanilla lens is biased in early layers.** Intermediate states haven't been re-normalized; readings often look like unigram statistics rather than task-conditional predictions. Tuned lens fixes most of this.
- **Corpus conditionality (SRP 2026).** Two tuned lenses fit on different corpora can return different top tokens for the same $h_\ell$. When a lens result is load-bearing for a claim, replicate with at least two corpora or use SRP's readout-only decomposition.
- **Layer-norm matters.** Applying the model's final LN before $W_U$ vs. not applying it changes readings materially. Report which convention is used.
- **Post-softmax vs pre-softmax.** Logit differences (pre-softmax) are the primary analytic quantity; taking argmax after softmax discards magnitude information that direct logit attribution needs.
- **Multi-token targets.** Lens readings target the *next* token. Multi-token answers require chaining or a task-specific probe.
- **Not a causal method.** Logit-lens readings are correlational — they show *what* the model represents mid-stack, not that intermediate representations *cause* the final output. Pair with activation patching for causal claims.

## Sources

- Blog: *interpreting GPT: the logit lens* — nostalgebraist, 2020 — [LessWrong post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
- Paper: *Eliciting Latent Predictions from Transformers with the Tuned Lens* — Belrose et al., 2023 — [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- Paper: *A Mathematical Framework for Transformer Circuits* — Elhage et al., Anthropic, 2021 — introduces direct logit attribution.
- Paper: *Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens* — He et al., 2026 — [arXiv:2609.01936](https://arxiv.org/abs/2609.01936).

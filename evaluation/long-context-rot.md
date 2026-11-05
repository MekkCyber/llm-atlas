# Long-Context Rot: Length vs Competition
*Depth — an empirical decomposition of long-context degradation into a length axis and a competition axis, with attention-on-evidence as a tight predictor.*

**TL;DR:** LLMs degrade on long retrieval inputs mainly because of **competing content**, not length. Up to 32K tokens of neutral filler cause little loss; statements that **conflict** with the evidence cause large drops. Across **126,473 trials**, the attention mass remaining on the evidence tokens predicts accuracy loss far better than $\log(\text{length})$. A masking-based long-context stress test popular in prior work fails a visible-text control, undermining its length-only conclusions.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../interpretability/README.md](../interpretability/README.md) · [README.md](README.md)

---

## What it is

Benchmarks measure "long-context rot" by filling a prompt with distractor text and watching accuracy drop. Because the filler also *lengthens* the prompt, length and distractor density move together and prior work attributes the degradation to length. This paper disentangles the two and shows the degradation is almost entirely driven by *competing content*, not length.

---

## How it works

Two experimental axes crossed on the same probe task (needle-in-haystack retrieval with verifiable answers):

1. **Length-only axis.** Neutral filler (unrelated news text, random lorem-ipsum-like passages) padding the context up to 32K tokens. Length varies, competition stays near zero.
2. **Competition-only axis.** Distractor density varies while context length is held fixed; the distractors are statements that **conflict** with the evidence (same entities, different facts).

For each configuration, record accuracy and a **mechanistic probe**: the attention mass left on the evidence tokens, aggregated across heads and depth.

### Findings

- **Length-only axis**: accuracy is roughly flat across 32K tokens of neutral filler.
- **Competition-only axis**: accuracy drops sharply with distractor density. The drop is explained better by attention-on-evidence ($R^2$ reported in paper) than by $\log(\text{length})$.
- **Masking test control**: a common masking-based stress test (mask tokens and compare to "unmasked" baseline) fails the authors' visible-text control, suggesting the test confounds length with visibility.

### The probe

Let $A^{(\ell, h)}_t$ be the attention matrix at layer $\ell$, head $h$, position $t$. Define

$$
\text{AoE}(t) = \sum_{\ell, h} \sum_{s \in \text{evidence}} A^{(\ell, h)}_t[s]
$$

the total attention mass from position $t$ onto the evidence span. Regressing error on $\text{AoE}$ (at the answer-generating position) explains substantially more variance than regressing on $\log(\text{length})$.

---

## Why it matters

- **Reframes long-context evaluation.** Benchmarks that measure "context length tolerance" without controlling competition are actually measuring competition tolerance.
- **Actionable signal for probe-based evals.** Attention-on-evidence is a mechanistic predictor that can be computed on any transformer — no new training, no new labels.
- **Challenges a research axis.** "Scale context window" work is often motivated by "longer helps" assumptions; this paper argues the bottleneck lives elsewhere.
- **Releases a reproducible probe and trial-level data.** 126,473 trials with per-configuration scores let follow-ups re-slice the data.

---

## Gotchas & tricks

- **"Competing" is a strong condition.** Mild distractors (same topic, different facts) matter most; adversarially constructed conflicts matter more; random off-topic filler matters little.
- **Attention-on-evidence is head-sensitive.** Aggregating across all heads is noisy; the paper reports results primarily on the "retrieval heads" subset, which may not generalize to every model. Sweep heads before using as a predictor on a new architecture.
- **Position matters.** Attention-on-evidence at the answer-generation position is the predictor; AoE averaged across all positions is noisier and weaker.
- **Does not disprove length-related effects entirely.** At extreme lengths (64K+) raw-length effects (positional encoding extrapolation failures, KV-cache compression artifacts) can dominate. The paper's claim is about the 1K–32K regime common in practice.
- **Benchmark designers should separate axes.** Needle-in-haystack suites should include a "competing needle" ablation to isolate competition. Length-only reports are incomplete.

---

## Sources

- Paper: *Length Is Cheap, Competition Is Not: Attention Predicts Context Rot in Retrieval* — Liu (UC San Diego CSE), 2026, arXiv 2611.04169 — primary source; 126,473-trial probe and released trial-level data.

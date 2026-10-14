# Long-Tail Creativity (q@Tail)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Formalizes "LLM creativity" as task-specific performance in the *long tail* of a reference input-output distribution. Uses n-gram absence from pretraining data as the novelty proxy, ranks subsets by tail-α, and scores the model only on the tail. The `q@Tail(α)` diagnostic is cheap, correlates with LMArena creative-writing rankings, and shows that creativity benchmarks sit measurably further in the tail than standard benchmarks.

**Prereqs:** [evaluation/README](../evaluation/README.md)
**Related:** [decontamination](../data/decontamination.md)

---

## What it is

A unified, pretraining-anchored way to measure creativity. Rather than inventing new metrics per benchmark (image diversity, text novelty, judge preferences), it reframes creativity as *performance restricted to tail inputs/outputs under a reference corpus* — typically the pretraining data. One axis, cross-benchmark comparable.

## How it works

Given a benchmark and a reference corpus:

1. **Score each (input, reference-output) pair's tail position.** Use n-gram absence or rarity against the reference corpus — a pair is "deeper in the tail" when its reference output contains n-grams the corpus rarely sees.
2. **Pick a tail cutoff α.** Keep the α-fraction of the benchmark that is furthest in the tail.
3. **Evaluate normally on that subset.** Standard benchmark metric (accuracy, judge preference, etc.) restricted to the tail subset = `q@Tail(α)`.
4. **Sweep α** to see how model ranking changes as you push deeper into the tail. "Creative" models dominate at high α; "memorizing" models dominate at low α.

The paper evaluates 15 creativity benchmarks × 26 open-weight models and verifies that:

- Creativity benchmarks sit further in the tail than standard reasoning / factual benchmarks (sanity check).
- Higher tail-α subsets correlate with human LMArena creative-writing rankings.
- `q@Tail(α)` is cheap enough to compute during model selection.

## Why it matters

- **Cross-benchmark comparability.** One number per model per tail cutoff, instead of a per-benchmark metric zoo.
- **Pretraining-anchored.** The novelty proxy is grounded in corpus statistics, not in a judge model that may share biases with the system under test.
- **Diagnostic for memorization.** A model that excels at α=0 but collapses at α=0.9 is likely leveraging memorized patterns. Models that hold up deep in the tail have genuine generative range.
- **Model-selection signal.** Enables routine creativity screening during post-training, not just at launch.

Suggests substantial room to improve LLM creativity by directly targeting the tail during training — the current field averages are well below what tail-performance peaks suggest is possible.

## Gotchas & tricks

- Choice of reference corpus matters: using the test corpus itself as reference is circular; use the pretraining data (or a close proxy).
- N-gram-absence is a crude novelty signal — semantically creative outputs may be lexically ordinary. The paper acknowledges this and treats `q@Tail` as a proxy, not ground truth.
- Correlation with LMArena is a *field validity* check, not an endorsement of LMArena as the ultimate creativity oracle.
- Tail-α choice: too low loses the creativity signal; too high leaves too few examples for stable estimates.

## Sources

- Paper: *Evaluating LLM Creativity through Long-Tail Performance* — anon, 2026 — [arXiv:2610.12998](https://arxiv.org/abs/2610.12998)

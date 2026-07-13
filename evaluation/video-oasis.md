# Video-Oasis
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **diagnostic audit** of existing Video-LLM benchmarks that finds **55% of samples are solvable without visual input or temporal context** — i.e., they measure text priors and world knowledge, not video understanding. After filtering out the shortcut samples, state-of-the-art Video-LLMs perform "only marginally above random guessing" on the remaining **video-native** subset. Not a new benchmark; a **sustainable diagnostic suite** for auditing any existing one, plus a distilled video-native testbed for studying which algorithmic design choices actually contribute. Introduced by Park et al. (Sejong U. / NAVER Cloud), 2026 (arXiv 2603.29616).

**Prereqs:** *(none)*
**Related:** [../data/decontamination.md](./../data/decontamination.md) · [mmlu.md](./mmlu.md) · [ifeval.md](./ifeval.md)

---

## What it is

A tool that treats existing Video-LLM benchmarks as *data* and asks per-sample: is visual or temporal information actually required to answer this? If a text-only or single-frame-only model can answer, the sample is a **shortcut** — it tests knowledge priors or textual reasoning, not video understanding.

The audit's finding — 55% shortcut rate — retroactively invalidates a large fraction of Video-LLM leaderboard progress. The filtered **video-native** subset is Video-Oasis's second contribution: a small, hard testbed on which SOTA models sit near chance, useful for studying which Video-LLM design choices matter.

## How it works

**Audit procedure.** For each sample in a target benchmark:

1. Run a strong text-only LLM on the question, with no visual input.
2. Run a strong LLM on the question + a *single, arbitrary* frame (no temporal context).
3. If either can answer, the sample is flagged as a shortcut.

The comparison uses matched prompt templates so the audit isn't hostile to the target benchmark.

**Video-native distilled subset.** The samples that survived the audit form the video-native testbed. On this filtered subset, SOTA Video-LLMs perform only marginally above random. That gap is what algorithmic-choice ablations on top of Video-Oasis are meant to measure.

**Design-choice ablations.** The paper uses the distilled testbed to study which Video-LLM design choices (temporal encoder type, frame-sampling density, visual-token budget per frame) actually move the video-native number.

## Why it matters

- **Retrospective correction on Video-LLM progress.** A large fraction of reported gains on prior benchmarks were driven by text/knowledge priors, not visual reasoning. Video-Oasis provides the audit that lets the community subtract that out.
- **Reusable diagnostic.** Rather than proposing yet another leaderboard, Video-Oasis is a tool. Any new Video-LLM benchmark can (and should) be audited before publication.
- **Actionable design guidance.** On the filtered testbed, small design changes have big effects; on the un-filtered version, they don't — because the un-filtered version is measuring the wrong thing.
- **Analogous to text benchmarks' contamination problem.** Same shape as data-contamination audits (Frontier-Math, contamination-checkers): the problem is what the benchmark is *actually testing*, not whether the model has seen it.

## Gotchas & tricks

- **Shortcut ≠ bad sample.** A shortcut sample may still measure a valid capability (text reasoning, general knowledge). The point is it doesn't measure *video* understanding — file it under a different eval.
- **Text-only prober choice matters.** A stronger text-only prober flags more shortcuts. Report the prober alongside the shortcut rate.
- **Video-native subset is small.** After filtering, the remaining sample count is much smaller than the parent benchmark; SOTA numbers on it have higher variance. Read them as directional.
- **The 55% is aggregate across audited benchmarks.** Per-benchmark rates vary considerably; check the paper for which ones are worst.
- **Doesn't cover generative video.** The audit is for Video-LLM understanding benchmarks. Video-generation evaluation is a separate problem.

## Sources

- Paper: *Rethinking Evaluation of Video Understanding (Video-Oasis)* — Park, Lee, Lee, Kim, Wee, Shim, Choi (Sejong U. / NAVER Cloud), 2026 — [arXiv 2603.29616](https://arxiv.org/abs/2603.29616).
- Related: data contamination diagnostics — [../data/decontamination.md](./../data/decontamination.md).

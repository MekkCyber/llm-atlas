# Corpus-Grounded Process Rewards (CorVer)

*Depth — a lightweight deterministic process reward for factual QA that replaces neural verifiers with corpus co-occurrence statistics.*

**TL;DR:** Sentence-level rewards for factual QA usually require expensive NLI verifiers, LLM judges, or full knowledge-graph pipelines, all of which fail on rare entities — exactly where reward accuracy matters most. CorVer derives the signal from a static corpus instead: extract facts from each sentence with a 0.5B extractor, look up co-occurrence statistics in Wikipedia, score the sentence accordingly, then align sentence-level credit to token-level advantages. Deterministic, cheap, especially reliable for rare facts.

**Prereqs:** [reasoning/prm.md](reasoning/prm.md), [_rewards.md](_rewards.md), [rlvr.md](rlvr.md)
**Related:** [grpo.md](grpo.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

A process reward (per-sentence, then mapped to per-token) for RL post-training on knowledge-intensive QA. Extends RLVR's "rule-verifiable reward" recipe from math/code into the factual domain by treating *Wikipedia co-occurrence statistics* as the rule.

## How it works

For each sentence $s_i$ in a generated reasoning trace:

1. **Fact extraction.** A small 0.5B extractor identifies the candidate facts in $s_i$ — typically (subject, relation, object) tuples.
2. **Corpus lookup.** Look up each fact's co-occurrence statistic in the reference Wikipedia corpus — a single lookup per fact (precomputed inverted index).
3. **Sentence score.** The sentence reward is a function of how strongly the extracted facts co-occur in the reference corpus, normalized for entity frequency.
4. **Token alignment.** The sentence score is broadcast over the sentence's tokens via a simple alignment, producing per-token advantages compatible with PPO / GRPO.

No NLI model, no judge LLM, no knowledge-graph traversal. The verifier is the corpus itself.

## Why it matters

- **Cheap.** A 0.5B extractor + one lookup per sentence vs. a neural NLI verifier called per sentence per rollout. The paper reports 4.8–8.4× faster training.
- **Reliable for rare entities.** Neural verifiers fail most on rare-entity claims — those are out-of-distribution for any learned verifier. Corpus statistics fail more gracefully: rare entities have sparse but unambiguous co-occurrence patterns.
- **Extends RLVR.** Math and code have rule verifiers; until CorVer, factual QA had to fall back to expensive neural verifiers or coarse response-level rewards. This brings RLVR-style cheap deterministic rewards to factual recall.
- Across 30 (model × benchmark) cells with 3B–14B instruction-tuned models, CorVer improves the raw baseline in every cell (avg +4.1 pp on TriviaQA) and beats four neural-verifier baselines in 18/20 cells.

## Gotchas & tricks

- Corpus choice matters. Wikipedia gives broad, balanced coverage; domain-specific corpora (PubMed, arXiv) work for narrower tasks. Mixing corpora dilutes the signal.
- The extractor is the failure point. A 0.5B model can miss multi-hop facts; tune the extractor on the same domain as the RL prompts.
- Sentence-level credit assignment inherits the PRM literature's caveat: sentence boundaries are fuzzy for non-list reasoning. Works well for structured QA, less well for free-form narrative answers.
- Not a substitute for a human / judge eval — corpus co-occurrence rewards *recall* of in-corpus facts but cannot judge *novel* claims. Pair with a held-out judge for absolute accuracy reporting.

## Sources

- Paper: *Verifiable Rewards Beyond Math and Code: Lightweight Corpus-Grounded Process Supervision for Factual Question Answering* — Fan et al., UIC, 2026 — [arXiv 2605.29648](https://arxiv.org/abs/2605.29648).

# Latent-swarm evolutionary search
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM-guided evolutionary search normally evaluates a population by decoding many candidate outputs and comparing them — which gets expensive fast. Latent-swarm evolution keeps each candidate as a KV cache rather than decoded text, defines crossover as *cache merging* by a summarizer agent, and runs a novelty filter in embedding space. Text is only decoded at the final elite step. Introduced in SwarmEvolve (2611.08483).

**Prereqs:** none
**Related:** [harness-ensemble-training](harness-ensemble-training.md)

---

## What it is

A population-based LLM search where:
- Each **individual** in the population is a KV cache representing a candidate hypothesis (program, proof, scientific idea).
- **Fitness** is scored in embedding space.
- **Mutation** and **crossover** happen via LLM operators on caches, not on decoded strings.
- **Selection** uses a novelty filter to maintain population diversity.

The population can be hundreds of agents because the per-agent cost is KV-cache attention, not full text generation.

## How it works

Loop:

1. **Initialize** $N$ agents, each conditioned on a slightly different prompt / seed; capture each agent's KV cache after an initial forward pass.
2. **Fitness.** For each agent, compute a feature vector from its KV cache (readout from the final layer) and score it via a task-specific head.
3. **Novelty filter.** Rank agents by novelty in embedding space; keep a diverse elite.
4. **Crossover (cache merging).** A *summarizer agent* consumes the KV caches of two or more parents and emits a new program hypothesis — this is the only step that may decode text, and it does so once per crossover, not per fitness evaluation.
5. **Mutation.** Perturb the parent cache or re-prompt from a related seed.
6. Repeat.

Only at the end does the elite's program get fully decoded for evaluation. On 21 mathematical and scientific-computing problems, the method reports the highest average score and faster convergence than OpenEvolve, ShinkaEvolve, and CodeEvolve using Qwen3 models.

## Why it matters

- **Decoding is the bottleneck of evo-LLM search.** Removing it from the inner loop converts an $O(\text{pop} \times \text{tokens})$ cost into $O(\text{pop})$ attention passes.
- **Diversity preservation** is first-class via a novelty filter in embedding space, rather than token-overlap metrics that misfire on equivalent-but-reworded programs.
- **General primitive.** The "population of KV caches + LLM crossover operator" pattern should apply to RL population training, synthetic-data generation, and reasoning-chain search.

## Gotchas & tricks

- KV caches are **tokenizer-specific**; the whole population must share one base model.
- The summarizer agent is the only decoder in the inner loop — its budget dominates cost once the population is large, so it must stay small (few hundred output tokens).
- Novelty filtering in embedding space can collapse if the embedding model is weak; use the base model's own final-layer features rather than a generic sentence encoder.
- Fitness scoring from a cache readout is *approximate* — the final elite is still verified by decoding and running the candidate.

## Sources

- Paper: SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization — 2026 — [arXiv:2611.08483](https://arxiv.org/abs/2611.08483)

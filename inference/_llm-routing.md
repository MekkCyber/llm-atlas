# LLM routing

*Taxonomy — picking which model (or ensemble of models) answers each request, at inference time.*

**TL;DR:** Modern deployments usually host a pool of models of different cost and capability. A router decides, per prompt, which model(s) should answer. Routers range from static rules to prompt-adaptive learned policies to per-prompt ensembles. The modern default for mixed workloads is a *prompt-adaptive cost-aware single-model router* with an ensemble fallback for high-stakes tails.

**Related taxonomies:** —
**Depth files covered here:** *(none yet)*

---

## The problem

Serving one model for every query wastes compute on easy prompts and loses accuracy on hard ones. Serving the biggest model for everything is linear in traffic and untenable at scale. The routing problem: map each prompt to the cheapest model whose expected quality meets the SLA.

## The shared pattern

Every router has the same three parts:

- **Prompt features.** Cheap features of the input (embedding, length, difficulty hints).
- **Capability estimates.** Per-model expected quality, and often cost, on prompts with those features.
- **Selection rule.** Picks one model (or a subset) given a budget / quality target.

The rows below differ in how fancy each of these three parts gets.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Static rules | Hand-written heuristics ("short → small, long → large") | Zero learning, brittle under drift | Early deployments, well-understood traffic |
| Length / difficulty cascades | Small model first; escalate on low confidence | Two forward passes on hard prompts | Latency-insensitive; easy-prompt-heavy traffic |
| Learned single-model routers | Supervised or RL policy over per-model quality | Needs labeled traffic | Mixed traffic with a clear quality floor |
| Cluster-capability routers (PISA) | Represent capability per prompt cluster, pick per prompt | Needs a good cluster space | Heterogeneous model pool, OOD-robust |
| Prompt-adaptive ensemble routers (PISA-E) | Pick *membership* and *size* of an ensemble per prompt | Ensemble cost adds up fast | High-quality or high-stakes tails, Pareto focus |
| Mixture-of-experts style routers (serving-time) | Learned gating across live replicas | Complex control plane | Very large pools with specialized models |

## How to choose

- **Default** for a 2–3 model pool: a learned single-model router with a confidence-triggered escalate-to-ensemble path.
- If the pool is heterogeneous and prompts span distributions (code + chat + math + agent), move to a **cluster-capability** router (PISA-style). Capability per cluster generalizes better OOD than a flat regressor.
- Use **ensemble routing** (PISA-E) only on the slice where tail quality matters more than per-prompt cost — e.g. code review, legal, medical summarization.
- Static rules are fine while traffic is small and your feedback loop is slow.

## Adjacent but distinct

- [Sparse reading](../agents/sparse-reading.md) — prunes *context*, not models; stacks before any router.
- Speculative decoding — prunes *decoded tokens* with a draft model; orthogonal to which model is final.
- Model cascades in the training recipe (student + teacher) — not a serving-time choice.

## Sources

- Primary: *Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference* — Urso, Carlini, Dazzi, Mordacchini (2026) — [arXiv:2611.10193](https://arxiv.org/abs/2611.10193) *(PISA / PISA-E variants in the table)*

---

## Conventions

- **Filename:** `_llm-routing.md` (taxonomy, leading underscore).
- **Folder placement:** `inference/`.
- **Linking:** depth files for individual routers link back here via their `Related:` lines when they land.
- **Scope:** inference-time routing only. Training-time router choices (MoE gating losses, expert specialization) belong in `architectures/_moe.md`.

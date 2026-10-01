# LLM Routing
*Depth — assigning queries to models to minimize cost at a quality bar.*

**TL;DR:** An LLM router sits at the serving front-door and sends each query to a cheap or expensive model based on predicted quality/cost. The field has focused on serving-time efficiency; **SaveRouter** (Lai et al., Nanjing University, 2026) adds the question nobody asked: are the savings big enough to pay back the **upfront supervision cost** of training the router in the first place? Routers are economic only if they recoup the dense query-model-feedback they trained on — SaveRouter sparsifies that collection.

**Prereqs:** [README.md](./README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md), [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Modern LLM serving uses a mix of models with wildly different cost-per-token (small open models, mid-tier, frontier closed). A router predicts, per query, which model will produce an acceptable answer and routes accordingly. Historically trained with **dense supervision**: run *every* candidate model on *every* historical query to collect `(query, model, quality)` labels, then train a classifier.

The usually-ignored cost is this label-collection step. For a 5-model pool and 100K training queries, that's **500K forward passes on potentially frontier models** before the router sees its first production query.

## How it works

### Classical dense routing

1. Build a historical query set.
2. Run all candidate models on all queries.
3. Score each `(query, model)` pair with a quality metric or judge model.
4. Train a classifier or ranker.
5. At serving time, predict per-query and route.

### SaveRouter — sparse supervision

Observation: router quality **saturates well before all labels are collected.** SaveRouter treats supervision acquisition as a sparse budget-allocation problem:

- **Selective acquisition.** At each step, pick the most informative `(query, model)` pair to label next, based on current router uncertainty.
- **Capability sharing.** Share capability info across *related* queries so unlabeled queries inherit nearby signals.
- **Query-level refinement.** Keep per-query refinement for fine-grained routing — don't collapse to a bag-of-templates classifier.

### Economic framing

SaveRouter evaluates routing by jointly accounting for:

- **Supervision expenditure** (upfront labeling cost).
- **Serving-time savings** (per-query cost reduction).

The headline metric is **break-even deployment volume**: how many production queries you need to serve before the router has paid for itself.

## Why it matters

Across **4 routing benchmarks**, SaveRouter uses only **33–41%** of available training feedback at competitive-or-better routing quality, and reduces break-even deployment volume by **1.9×–9.5×** vs. the fastest conventional router. More supervision is not always economic: the minimum-serving-cost supervision level can differ from the fastest-payback level.

Significance: LLM routing has been treated as a free optimization layer. SaveRouter forces the field to account for the training it hides — and shows the hidden cost is big enough to invert conclusions.

## Gotchas & tricks

- **Capability-sharing hinges on similarity.** The sharing step uses embedding neighborhoods; if your query distribution has mode-collapse (lots of near-duplicates), sharing works. On diverse workloads, selective acquisition carries more of the load.
- **Fastest payback ≠ cheapest serving.** The paper shows these can differ. Pick based on expected deployment volume: short lifetime → fastest payback; long lifetime → minimum serving cost.
- **Judge model matters.** If quality labels come from a learned judge, the router inherits its biases. SaveRouter's economics are *relative* — any bias is shared across methods.
- **Composable with speculative decoding and model cascades.** Routing happens at request level; speculative/cascade happens inside a request. They don't conflict but the economics do interact.

## Sources

- Paper: *Routing Should Pay for Itself: Sparse Supervision for Economical LLM Routing* — Lai, Bian, Ma, Jiang, Chen, Liu, Tan, Ye — Nanjing University, 2026 — [arXiv:2609.37402](https://arxiv.org/abs/2609.37402).

# Granularity-Aware Search (GRASP)
*Depth — RL-trained agent that picks retrieval modality and evidence granularity as first-class actions.*

**TL;DR:** GRASP trains an agentic-RAG policy where the agent's action space includes **three retrieval tools at different granularities**: semantic search (broad exploration), keyword search (entity-specific pinning), and paragraph reading (local verification). A composite reward shapes the policy to skim broadly, zoom in only when needed, and stop early — yielding interpretable skimming/scanning behavior on multi-hop QA.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [search-oriented-context.md](search-oriented-context.md)

---

## What it is

Prompting-based agentic RAG treats retrieval as a black-box tool: the agent decides *whether* to call, but rarely picks *how* to search. GRASP moves the granularity decision into the action space and trains it with RL, so the policy learns *when* semantic search wins, *when* keyword search is the right pin, and *when* to zoom in with paragraph reading rather than issue another top-level query.

## How it works

### Action space

Three retrieval actions at complementary granularities:

- **Semantic search.** Vector search over the corpus; returns candidate passages ranked by similarity. Best for broad exploration when the target concept isn't known by name.
- **Keyword search.** Lexical retrieval; returns exact-match hits. Best for entity-specific evidence — names, dates, technical terms.
- **Paragraph reading.** Load a specific paragraph in full for verification. Best for local grounding after retrieval has narrowed the search.

Plus the standard "answer" and "continue searching" actions that end or extend a step.

### Reward composition

The policy is trained with a multi-term reward:

- **Answer accuracy** — task-level correctness.
- **Grounded reading** — reward for citing evidence the agent actually loaded via paragraph reading (not hallucinated).
- **Complementary search** — bonus for combining modalities (e.g., semantic → keyword → paragraph) rather than repeating the same tool.
- **Turn efficiency** — penalty for extra retrieval steps, encouraging early stopping.

Rewards sum before entering the RL update (GRPO-family). The composition is what shapes the emergent skimming/scanning behavior.

### Emergent skills

Ablations show three interpretable behaviors learned by the trained policy:

- Semantic search dominates early turns (broad exploration).
- Keyword search fires when the agent has identified a specific entity to pin.
- Paragraph reading is used for final verification before answering.

None of these are hand-coded; the reward structure induces them.

## Why it matters

- **Granularity as a first-class axis.** Most agentic RAG work adds *more* tools; GRASP shows the productive axis is *complementary* granularity control.
- **Beats prompting-based and RL-based retrieval baselines** on multi-hop QA (specific benchmarks in the paper).
- **Interpretable trained policy.** Emergent skim/scan/pin behaviors are inspectable — a rare property in RL-trained agents.
- **Composes with state externalization.** GRASP + SearchOS-style SOCM is a natural stack: granularity-aware actions on top of externalized coverage state.

## Gotchas & tricks

- **The reward shape is the whole method.** Drop the grounded-reading term and the agent learns to skip paragraph reading. Drop the turn-efficiency term and it over-searches. Balancing coefficients matters.
- **Requires a real dual-index corpus.** You need both a vector index and a lexical index available at rollout time; single-index setups can't train the granularity choice.
- **Training-time budget affects learned behavior.** If rollouts allow only 3 turns, the agent never learns paragraph reading; if allowed 20, it may over-verify. Match training budget to deployment budget.
- **Domain transfer is unstudied.** The emergent skim/scan/pin decomposition is reported on multi-hop QA; whether the same policy transfers to closed-domain research or code search is an open question.

## Sources

- Paper: *GRASP: GRanularity-Aware Search Policy for Agentic RAG* — Gandhi, Lee, Todmal, Dernoncourt, Rossi, Wang, Lan — University of Massachusetts Amherst / Adobe Research, 2026.

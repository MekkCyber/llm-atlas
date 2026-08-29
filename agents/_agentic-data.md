# Agentic Data Generation
*Taxonomy — how pipelines produce interaction data for training LLM agents.*

**TL;DR:** LLM agents learn from generated interaction data more than from labelled text, and the field's pipelines look domain-specific (browser vs code vs computer-use) but factorize the same way. Represent every agentic-data pipeline as the tuple **(E, q, τ, v)** — environment spec, task signal, interaction realization (trajectory), optional verifier — and score its output through the **ACE lens**: **A**ccuracy (grounded and internally consistent), **C**omplexity (calibrated to a declared learner's frontier), div**E**rsity (behavioral coverage, not surface variation). Most 2025–2026 pipelines can be located on this surface; the field is trending toward execution-grounded accuracy, learner-relative complexity, and behavior-space diversity.

**Related taxonomies:** [../data/_data-curation.md](../data/_data-curation.md)
**Depth files covered here:** (populated as depth pages land)

---

## The problem

Agents need trajectories: `(observation → tool_call → result → observation → …)` with a success signal at the end. Human-annotated trajectories are prohibitively expensive at scale, and pretrained LLMs don't generate them natively. Every serious agent training run generates its own — with wildly different pipelines whose reported strengths look incomparable. Without a shared factorization, "our data is better" is unfalsifiable.

## The shared pattern

Every agentic-data pipeline picks four things:

- **E — Environment specification.** The interface the agent acts through (browser, terminal, notebook, virtual OS, tool registry). Fidelity ranges from a text-only tool schema to a full VM.
- **q — Task signal.** How tasks are drawn: from a hand-curated seed set, from a task generator, from real user queries, or from bootstrapping (agent proposes tasks).
- **τ — Interaction realization.** How trajectories are produced: single expert model, self-play, teacher-student rollouts, tree search over branches, human demonstrations.
- **v — Verifier (optional).** How success is decided: rule-based check, model judge, execution result, human, or no verifier at all (rely on q's ground truth).

Pipelines differ in which of these is the *anchor* (fixed) and which are dependent variables. E-anchored pipelines start from a heavy environment; q-anchored ones start from a task distribution; τ-anchored ones start from a teacher model.

## Variants

| Approach | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Teacher-model rollouts** (τ-anchored) | Strong LLM drives the trajectory in a light environment | Ceiling capped by teacher | Any well-covered domain with a good teacher |
| **Self-play / bootstrapping** (q-anchored generator) | Agent proposes tasks it can partially solve, verifier gates them | Diversity of proposed tasks limits coverage | Domains with cheap verifiers (math, code) |
| **Execution-grounded** (v-anchored) | Every trajectory validated by real execution / unit tests | Requires an executable environment | Code, tool-use with rich runtime signals |
| **Tree-search rollouts** (τ-anchored) | Branch mid-trajectory, keep winning branches | Compute-heavy | High-value tasks where per-trajectory cost is OK |
| **Human-in-the-loop** (τ-anchored) | Humans complete/correct trajectories | Doesn't scale; annotation lag | Bootstrapping a new domain from zero |

## How to choose

- **Ask what your verifier is first.** If execution can adjudicate (code, tool calls with real APIs), anchor there — verifier-grounded pipelines dominate on accuracy in 2026 agent-data reads.
- **Match complexity to the declared learner.** ACE's C is learner-relative — data that's trivial for a 70B model is well-calibrated for a 3B. Don't reuse curriculum across scales without recalibrating difficulty.
- **Diversity is behavioral, not surface.** Two trajectories with different tool arguments but the same tool sequence add nothing. Score coverage over abstract action patterns, not string diversity.
- **Don't conflate candidate generation with selection.** Most pipelines mix them; the ACE framework asks you to separate "how did I propose it" from "how did I decide to keep it."

## Adjacent but distinct

- [../data/_data-curation.md](../data/_data-curation.md) — pretraining data curation shares the accuracy-vs-coverage tension, but agentic data adds a *trajectory-shape* dimension pretraining doesn't have.
- [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md) — the "which prompts to RL on" question is the C-axis at the batch level; ACE generalizes it to whole pipelines.

## Sources

- Paper: *What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents* — Zeng et al. (Huawei Noah's Ark / SJTU), 2026 — arXiv:2608.27260 — introduces the (E, q, τ, v) factorization and the ACE lens.

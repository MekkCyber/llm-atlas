# Experience Internalization for Self-Evolving Agents
*Depth — the design space for turning an agent's past interactions into reusable parametric capability.*

**TL;DR:** "Self-evolving" LLM agents promise to learn from their own past interactions. The hard part isn't *whether* — it's *how* to convert raw trajectories into parametric updates without destroying earlier capability. Chen et al. (2026) isolate three axes — **experience granularity**, **experience injection pattern**, and **internalization regime** — and show the choices interact strongly. Naive continual SFT, the default in the literature, is dominated by combinations across the three axes.

**Prereqs:** [agents/README.md](./README.md), [post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [post-training/_post-training.md](../post-training/_post-training.md), [post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Experience internalization is the step where an agent absorbs contextual information from past interactions into model weights so the next interaction doesn't need that context in-prompt. Three orthogonal design choices fully describe a recipe:

1. **Experience granularity** — what is the unit of internalisation? A full trajectory (prompt + tool calls + outcome), a distilled "skill" (a recurring pattern across trajectories), or a single decision point.
2. **Experience injection pattern** — how is the experience turned into training data? Dense (every trajectory becomes one or more SFT examples), periodic (batch up experiences and consolidate every N steps), or filtered (only successful / informative trajectories).
3. **Internalization regime** — what updates the model? Full fine-tune, LoRA / adapter, distillation from a frozen reference, or RL on the internalised behaviour.

## How it works

Each axis maps to concrete implementations:

| Axis | Choices | Tradeoff |
| --- | --- | --- |
| Granularity | trajectory · skill · decision | larger units retain context but bias to recent tasks; smaller units generalise but lose causal structure |
| Injection | dense · periodic · filtered | dense forgets fast; periodic loses recency; filtered needs a reliable success signal |
| Regime | full FT · LoRA · distillation · RL | full FT is high-capacity but high-forgetting; LoRA is bounded; distillation is stable; RL needs a verifier |

The paper runs the cross product on multiple agent benchmarks and reports which combinations approach the upper bound (oracle replay) and which collapse below baseline. Naive continual SFT — trajectory granularity, dense injection, full FT — is the most common in prior work and one of the *worst* combinations: it forgets earlier skills without consolidating new ones.

## Why it matters

- **A diagnostic for the "self-evolving" claim.** Most papers using that term land on one cell of this three-axis grid. The grid lets readers see which one, and the literature can finally compare apples to apples.
- **Recovers retention without sacrificing learning.** The strongest combinations roughly double effective skill retention vs. continual-SFT defaults — without architectural changes.
- **Maps cleanly onto memory systems.** External memory (RAG, vector stores) and internalisation are two sides of the same coin; the granularity axis is where they intersect.

## Gotchas & tricks

- **Skill granularity needs a distiller.** "What is the skill underlying this trajectory" is itself an LLM call; it can hallucinate, inflating internalised noise. Verify the skill on held-out trajectories before training on it.
- **Periodic injection is the safest default.** Consolidate every N interactions, replay a slice of older experiences, and use a small LR. This is essentially experience replay from classical RL.
- **LoRA is a soft cap on forgetting.** Bounded parameter capacity bounds the maximum drift from the base — useful if your base already covers the broad behaviour you don't want to lose.
- **Don't confuse with continual pre-training.** Continual pre-training updates with the *same* objective as base training; experience internalisation updates with a *behavioural* objective derived from agent rollouts. The data sources, supervision signals, and failure modes all differ.

## Sources

- Paper: *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* — Chen, Yang, Fan et al. (Renmin University / Meituan), 2026 — [arXiv:2606.04703](https://arxiv.org/abs/2606.04703).

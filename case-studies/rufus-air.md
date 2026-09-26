# Case Study: Rufus-Air

*Amazon Rufus's open post-training recipe on GLM-4.5-Air-Base (106B-A12B MoE): an 8-stage serial pipeline (SFT → Reasoning RL → Coding RL → IF RL → General Agent → Coding Agent → Search Agent → RLHF) built entirely from public data and open components, with no in-house teacher and no new human annotation.*

**Related concepts:** [rlvr](../post-training/rlvr.md) · [grpo](../post-training/grpo.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [dpo](../post-training/dpo.md) · [rl-prompt-curation](../post-training/rl-prompt-curation.md) · [reward-reliability-ordering](../post-training/reward-reliability-ordering.md) · [agentic-post-training](../agents/agentic-post-training.md) · [olmo-2](olmo-2.md) · [deepseek-r1](deepseek-r1.md)

---

## What this is

**Rufus-Air**, released September 2026 by Amazon's Rufus team. Two artifacts, one contribution:

1. **The recipe.** An 8-stage serial post-training pipeline applied to `GLM-4.5-Air-Base` (106B total, 12B activated MoE) that lifts it past the official GLM-4.5-Air post-trained release and into "competitive with similarly-sized open models" territory.
2. **The reproducibility contract.** Data, reward design, infrastructure, stage order, and stagewise results are all documented; the recipe uses only open-source components and public data, much of it used *as released*.

The interesting contribution is not the base model or any single stage — the individual RL algorithms are standard GRPO — it's the **ordering, difficulty filtering, and reward-reliability principle** that turn a stack of well-known techniques into a recipe that reproducibly wins.

Read this report if you want a template for how to sequence and configure post-training stages on an existing base model without a proprietary teacher.

---

## Architecture — inherited, not novel

The base model is `GLM-4.5-Air-Base` unchanged: 106B-total / 12B-active MoE, standard transformer stack. All architectural choices (attention variant, MoE gating, positional encoding) come from the base. Rufus-Air only post-trains.

This is deliberate: the paper's claim is that *recipe* rather than architecture is the frontier for open post-training, and demonstrating that on someone else's base weights makes the point cleanly.

---

## The 8-stage pipeline

```
GLM-4.5-Air-Base
    │
    ▼
[1. SFT]                    — diverse high-quality demos, capability floor
    │
    ▼
[2. Reasoning RL]           — RLVR on math/logic, verifiable rewards
    │
    ▼
[3. Coding RL]              — RLVR on code, unit-test rewards
    │
    ▼
[4. Instruction-Following RL] — RLVR on IF constraints (format, length, language)
    │
    ▼
[5. General Agent]          — RL on multi-turn tool use, mixed rewards
    │
    ▼
[6. Coding Agent]           — RL on code-editing agents, execution rewards
    │
    ▼
[7. Search Agent]           — RL on deep-search agents, retrieval-quality rewards
    │
    ▼
[8. RLHF]                   — preference-based polish, judge-based signals
    │
    ▼
Rufus-Air Instruct
```

Each stage's outputs feed the next as SFT seed where useful. Stages 2–7 all use group-relative policy optimization ([grpo](../post-training/grpo.md)) as the underlying algorithm — the differences are the reward function, prompt distribution, and rollout environment.

### Stage 1: SFT

Broad, curated instruction data drawn from public releases (open SFT mixes, Tülu-style demonstrations, coding datasets). No new human annotation. Establishes the format and floor capability. The paper's first finding: *diverse* high-quality SFT is worth more than *large* mediocre SFT.

### Stages 2–4: Verifiable-reward RL ([rlvr](../post-training/rlvr.md))

- **Reasoning RL** — math + logic problems, symbolic answer match.
- **Coding RL** — code problems, unit-test pass.
- **IF RL** — instruction-following constraints (format schema, length, language, keyword presence), programmatic checkers.

All three use rule-based rewards. All three use difficulty filtering ([rl-prompt-curation](../post-training/rl-prompt-curation.md)) to keep prompts in a productive learning band — too-easy prompts give zero gradient, too-hard prompts saturate rejection.

### Stages 5–7: Agentic RL

The paper's least-standard stages. Each is a full agentic rollout with tool use, environment execution, and multi-turn interactions:

- **General Agent** — mixed tool use across domains, hybrid rewards (task success + rubric).
- **Coding Agent** — code-editing loop with execution feedback; rewards on repo-level task success.
- **Search Agent** — deep-search loop over retrieval tools; rewards on final-answer correctness plus turn-level rubric.

See [agentic-post-training](../agents/agentic-post-training.md) for the shared shape of these stages.

### Stage 8: RLHF

Preference-based polish using an open reward model. Deliberately last, because judge-based rewards are the softest signal and go last per the paper's ordering rule.

---

## The ordering principle

The paper's most transferable idea: **order RL stages by how reliable their rewards are**.

- **Hard verifiable (RLVR)** — exact-match, unit tests, schema checks. Rewards can be trusted; put these stages *first* so hacking is impossible and the policy learns the desired capability cleanly.
- **Environment-grounded rewards** — code execution, tool responses. Still concrete but noisier. Middle of the pipeline.
- **Judge-based rewards** — LLM judges, preference models. Most gameable. Go *last*, after capabilities are locked in by earlier stages.

Reversing the order collapses the recipe: RLHF first over-fits the model to preference judges and destroys the verifiable capabilities RLVR would otherwise install. See [reward-reliability-ordering](../post-training/reward-reliability-ordering.md).

---

## Infrastructure as recipe

The report explicitly claims infrastructure is not "implementation detail":

- Rollout serving throughput (batching, KV cache reuse, disaggregated prefill/decode) sets the practical RL step count per day.
- Reward-verifier latency (unit-test runners, judge inference) sets rollout wall-time.
- Reference-model KV serving determines KL cost.

Documented so a reproducer can hit similar throughput; the paper frames the infra choices as part of the recipe surface, not below it.

---

## Results snapshot

| Benchmark | GLM-4.5-Air (official) | Rufus-Air |
| --- | --- | --- |
| Reasoning (aggregate) | baseline | **improved** |
| Coding (aggregate) | baseline | **improved** |
| IF (aggregate) | baseline | **improved** |
| Agent (aggregate) | baseline | **improved** |

Numbers are qualitative here — see the paper for exact per-benchmark deltas. The headline: purely-open post-training on an open base can match or exceed a same-family official release *without* a proprietary teacher or new human annotation.

---

## Key takeaways

1. **Diverse SFT sets the floor.** A wider SFT distribution is worth more than a deeper one; RL cannot rescue a narrow SFT.
2. **Difficulty-filter every RL stage.** Prompts must fall into the "some rollouts succeed, some fail" band; outside it, gradients vanish.
3. **Order RL stages by reward reliability.** Verifiable rewards first, judge-based last. Reversing collapses the pipeline. ([reward-reliability-ordering](../post-training/reward-reliability-ordering.md))
4. **Agentic stages come after single-turn capability RL.** General/Coding/Search Agent RL builds on the reasoning + coding capabilities installed in Stages 2–3.
5. **Infrastructure is a recipe surface.** Rollout throughput, verifier latency, and reference-model serving decide the practical step count; document them alongside data and hyperparameters.
6. **A proprietary teacher is not required.** Rufus-Air's ability to match or exceed the official post-training with only public data shows that the "you need a strong closed teacher" belief is not load-bearing at this scale.

---

*Pairs well with:* [olmo-2 case study](olmo-2.md) for a parallel "fully-open recipe" reference at smaller scale, and [deepseek-r1 case study](deepseek-r1.md) for a comparable multi-stage RL pipeline centered on reasoning.

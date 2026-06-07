# DataCOPE — Unsupervised Skill Discovery for Data-Analysis Agents
*Depth — discover reusable analytical skills without labels by iterating generation, verification, and distillation.*

**TL;DR:** Data-analysis agents have a supervision problem: report-style outputs are open-ended, and reasoning answers are diverse, so labelled examples are scarce. **DataCOPE** (Song et al., 2026) replaces labels with two task-specific *unsupervised verifiers* — an Adaptive Checklist Verifier for reports and an Answer Agreement Verifier for reasoning — then runs a loop that generates trajectories, verifies them, and distils the verified ones into named, reusable skills injected at inference time. No parameter updates.

**Prereqs:** [agents/README.md](./README.md), [agents/proactive-discovery.md](./proactive-discovery.md)
**Related:** [post-training/_rewards.md](../post-training/_rewards.md), [post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

DataCOPE is a *skill library* construction recipe. A skill is a piece of named, reusable procedural knowledge ("when given a CSV with date columns, parse them and group by week before plotting") that the agent retrieves and follows at inference. The novelty is *how* the library is built when no labels exist:

- **Adaptive Checklist Verifier (ACV)** — for report tasks (Deep Data Research). The verifier infers a checklist of properties a good report should have, then scores trajectories against it. The checklist adapts per task type.
- **Answer Agreement Verifier (AAV)** — for reasoning tasks (DABStep). Trajectories agree on a final answer with probability proportional to correctness; self-consistency across rollouts becomes the supervision signal.

## How it works

```
skills = []
for round in range(N):
    trajectories = agent.run(tasks, library=skills)
    verified = verifier.score(trajectories)    # ACV or AAV
    new_skills = distill(verified)             # extract reusable pattern
    skills.extend(new_skills)
```

Each round:

1. **Generate** trajectories with the current skill library injected as context.
2. **Verify** unsupervisedly. ACV builds a per-task checklist (criteria distilled from the task statement) and judges trajectory output coverage. AAV computes self-consistency on the final answer.
3. **Distil** verified trajectories into compact named skills. A skill carries trigger conditions (when to apply), a procedure (what to do), and an example.

The library grows monotonically; on the next task, retrieval picks relevant skills and injects them into the agent's prompt. The base model never changes weights.

## Why it matters

- **Skill libraries without supervision.** Most prior skill-discovery work needs gold labels or a strong learned reward model. DataCOPE shows a useful library can be bootstrapped from agreement and checklist signals alone.
- **Model-agnostic.** Because the library lives in context, switching base models doesn't require rebuilding — the same library transfers across backbones.
- **Big absolute gains.** +9.71% on Deep Data Research and +32.30% on DABStep across multiple bases. The DABStep number is large because reasoning over tabular data is one of the few agent benchmarks where small procedural tweaks compound aggressively.

## Gotchas & tricks

- **ACV's checklist is itself an LLM call.** It can miss properties or hallucinate criteria; on tasks where the implicit "good report" is hard to articulate, the verifier becomes the bottleneck.
- **AAV degenerates when the model is wrong consistently.** Self-consistency rewards agreement, not correctness; if every rollout reaches the same wrong answer (common on systematic errors), AAV reinforces the error.
- **Skills can collide.** Two skills with similar triggers fire together and produce confused trajectories. Periodic dedup / retrieval-side scoring helps.
- **Doesn't replace fine-tuning.** Inference-time skills are bounded by the base model's ability to follow the injected schema. Beyond ~50 active skills the prompt gets unwieldy; that's the practical ceiling.

## Sources

- Paper: *Unsupervised Skill Discovery for Agentic Data Analysis* — Song, Tang, Qiao, Liang, Chen, Deng (Zhejiang University / NUS), 2026 — [arXiv:2606.06416](https://arxiv.org/abs/2606.06416).

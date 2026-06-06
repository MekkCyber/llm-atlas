# Experience Internalization
*Depth — turning past agent interactions into parametric capability via repeated fine-tuning, without collapse over multiple iterations.*

**TL;DR:** "Self-evolving" LLM agents that fine-tune on traces from their own past interactions usually *collapse* after a few iterations rather than compound. The fix is structural: internalize **principles, not instances**; inject **per-step, not globally**; train **off-policy from a teacher's trajectories**, not on-policy from the student's. Together, these three choices yield stable multi-iteration improvement (Chen et al., 2026).

**Prereqs:** [post-training/fine-tuning/README](../post-training/fine-tuning/README.md), [post-training/on-policy-distillation](../post-training/on-policy-distillation.md)
**Related:** [agents/README](README.md)

---

## What it is

Pipeline:

```
agent runs in environment → trajectories → distill into weights → repeat
```

The hope: each iteration internalizes new experience and the agent gets better. The reality: most pipelines plateau and then degrade. Experience Internalization identifies *three design axes* that decide whether the loop compounds or collapses.

## How it works

**Axis 1 — Granularity: principles > instances.**

- *Instance-level*: "in trajectory T, action A was good." Brittle — overfits to surface details.
- *Principle-level*: "when state has property P, prefer action class C." Abstracted away from trajectory specifics; survives multi-iteration training.

Principles are extracted by summarizing successful trajectories into transferable strategies (an LLM-as-summarizer step, with verification).

**Axis 2 — Injection pattern: step-wise > global.**

- *Global*: prepend or system-prompt the experience once per task.
- *Step-wise*: inject the relevant experience at each intermediate decision state.

Step-wise wins, especially for long-horizon tool use, because it aligns the experience with the decision points where it's actually needed.

**Axis 3 — Internalization regime: off-policy distillation > on-policy.**

- *On-policy*: distill on the *student's* trajectories with the student's mistakes as supervision signal. Reinforces flawed states.
- *Off-policy context-distillation*: distill on a *teacher's* high-quality trajectories with the experience injected as context. Stable training signal; doesn't compound student errors.

Note the direct contrast with general on-policy distillation (e.g., OPRD) for *reasoning* — different regime, different conclusion. For experience internalization in long-horizon agents, off-policy wins.

## Why it matters

- **Most "self-improving" loops degrade silently.** This paper is the first to systematically diagnose why and give a working recipe.
- **Concrete, reusable axes.** Practitioners can audit a pipeline against the three axes and predict its failure mode before running it.
- **Applies to any RAG-then-distill agent training pipeline.** The "experience as parametric capability" framing covers many real systems: customer-service agents internalizing FAQs, coding agents internalizing project conventions, browsing agents internalizing site-specific behaviors.

## Gotchas & tricks

- **Principle extraction quality bounds the ceiling.** Bad principles → bad internalization. Use an LLM judge to verify principles before passing them into SFT.
- **Step-wise injection breaks vanilla supervised loss.** You need a trace-format that marks decision points and aligns context per step; not all SFT pipelines support this out of the box.
- **The on-policy / off-policy choice is regime-specific.** For verifiable reasoning, on-policy distillation tends to win. For long-horizon agentic experience, off-policy wins. Don't generalize one finding to the other domain without re-running the ablation.
- **Multi-iteration eval is essential.** A single-iteration improvement is misleading — the failure mode is *compounding collapse* over iterations 3–5. Measure the trajectory, not just the endpoint.

## Sources

- Paper: *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* — Chen et al., 2026 — [arXiv:2606.04703](https://arxiv.org/abs/2606.04703) — primary source.
- Code: `github.com/RUCBM/ExpInternalization`.

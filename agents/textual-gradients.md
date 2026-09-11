# Textual-Gradient Prompt Optimization

*Depth — treat natural-language feedback on an agent's failure as a "gradient" and update prompts by applying it; targeted intervention + semantic gradient clustering close known failure modes.*

**TL;DR:** Textual-gradient methods (TextGrad et al.) frame prompt optimization for LLM systems as an analogue of gradient descent: given a failure, an LLM generates natural-language "gradients" (edits to make) on the target prompts. In multi-agent systems (MAS) two stages are unreliable: **gradient extraction** picks a target prompt without checking that editing it fixes the failure, and **gradient aggregation** concatenates unrelated gradients into one blob. **AgentGrad** fixes both: *sequential intervention* pinpoints the responsible agent by mutating one output at a time and observing which mutation resolves the failure; *semantic gradient abstraction* clusters gradients by semantic similarity before applying them. SOTA on 5 MAS benchmarks, 2.5× wall-clock speedup.

**Prereqs:** [agent-harness](agent-harness.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [README](README.md)

---

## What it is

Prompt optimization treats agent prompts as the object to improve, with the model held fixed. **Textual gradients** cast that improvement as a descent-style loop where the "gradient" is a natural-language critique of what went wrong on a specific rollout, and the "update" is an LLM re-writing the target prompt to incorporate the critique.

For a single-agent system this is direct: one failure → one critique → one prompt edit. For a multi-agent system it breaks:

1. **Which agent's prompt is responsible?** Prior work picks a target prompt without verifying that editing it *would* resolve the failure.
2. **How do you combine gradients across failures?** Concatenating them mixes unrelated failure modes and produces prompts that fail to generalize.

AgentGrad addresses these two.

## How it works

**Sequential intervention** (per failure):

```
target_agent = None
for a in agents:
    corrected_output = LLM_expert.rewrite(a.output, failure_context)
    if rerun_with(a → corrected_output) resolves failure:
        target_agent = a
        supervision  = corrected_output
        break
```

The target agent is the earliest one whose corrected output flips the outcome. Its corrected output becomes agent-level supervision for extracting a fine-grained textual gradient.

**Semantic gradient abstraction** (across failures):

```
gradients = [extract_gradient(f, target_agent[f], supervision[f]) for f in failures]
clusters  = cluster_by_semantics(gradients)
generalized = [distill_shared_correction(c) for c in clusters]
apply(generalized)
```

Clustering prevents "prompt salad" — unrelated corrections stacked on the same prompt.

## Why it matters

- **Localization is causal.** Sequential intervention answers *"whose prompt do I edit?"* by a counterfactual test, not by heuristics.
- **Cluster-first, apply-once.** Semantic abstraction turns many single-instance edits into few generalizable ones — less prompt drift, better transfer.
- **Speed.** 2.5× wall-clock over the next-fastest baseline; the intervention loop pays for itself by avoiding wasted edits.

## Gotchas & tricks

- **Intervention cost.** Each failure runs the system once per candidate target agent to find the responsible one. Cap the search order or cache resolved-agent hits.
- **Clustering granularity.** Too many clusters ≈ no aggregation; too few ≈ prompt salad again. The paper picks a mid-granularity threshold empirically.
- **Not a replacement for weight updates.** For a persistent knowledge gap, fine-tuning still wins. Textual gradients edit the *frame* the model reasons within.
- **Distinct from post-training textual gradients.** Some post-training methods use similar "critique-and-rewrite" mechanics on rollouts (see [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)). Textual-gradient prompt optimization keeps the model frozen and edits *prompts*.

## Sources

- Paper: *AgentGrad: Intervention-guided Prompt Optimization for Multi-Agent Systems*, 2026 — [arXiv:2609.08572](https://arxiv.org/abs/2609.08572).
- Prior art: *TextGrad: Automatic "Differentiation" via Text* — Yuksekgonul et al., 2024 — the base textual-gradient framework.

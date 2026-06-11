# Delegation SFT

*Depth — training a main agent to delegate to subagents via SFT on harness-generated traces.*

**TL;DR:** Long-horizon agent tasks blow past the main model's context window. The fix in modern stacks is a *main-agent + subagents* pattern: the main agent decomposes the task, dispatches subtasks, and integrates summaries. The orchestrator is usually a runtime harness. SearchSwarm (2026) shows you can move the orchestrator's behaviour *into the model weights* by running the harness once to collect successful trajectories and SFT-ing on them. The resulting model delegates correctly without runtime scaffolding — SearchSwarm-30B-A3B hits 68.1 on BrowseComp and 73.3 on BrowseComp-ZH.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [harness-optimization](harness-optimization.md) · [dual-role-self-play](dual-role-self-play.md)

---

## What it is

Multi-agent / main-agent-with-subagents architectures are the standard pattern for *deep research*, code generation across many files, and any task with a context bigger than the model's window. The orchestrator decides:
- when to delegate (vs. answer directly),
- what subtask to spin out (decomposition),
- what to pass into the subagent (context bounding),
- what to extract from its return (summarization).

Most production stacks implement this with a runtime harness — an explicit orchestrator written in code that the main agent talks to. Delegation SFT moves all four decisions *into the model weights* by generating training data that captures correct delegation behaviour, then fine-tuning on it.

## How it works

1. **Build a harness that guides correct delegation.** It constrains: when the main agent is *allowed* to delegate, what the subagent's return format must look like, and how summaries flow back to the main thread. The harness exists only at data-generation time.
2. **Run the harness on many tasks, collect trajectories.** Each successful trajectory is a (state, delegate action, integrated summary) trace where the harness ensured the decisions were correct.
3. **Filter to clean, successful trajectories.** Standard rejection-sampling-style curation.
4. **SFT the main agent on the trajectories.** Treat the main agent's harness-driven outputs as ground-truth target sequences. The model learns *both* the surface behaviour (what to say next) and the latent policy (when to delegate, what to extract).
5. **Deploy without the harness.** At inference, the SFT'd model emits delegation actions natively. Subagents can be the same model called recursively or a smaller model.

## Why it matters

- **Removes runtime orchestrator brittleness.** Hand-coded orchestrators ossify; weight-baked delegation generalizes to tasks the orchestrator wasn't designed for.
- **Strong open results.** SearchSwarm-30B-A3B (MoE, 3B active) hits BrowseComp / BrowseComp-ZH SOTA in its scale class, with open weights, harness, and data.
- **Pattern transfers beyond deep research.** Any subagent-orchestration setting (code repos, document QA, math with computer algebra) can apply the same recipe.
- **Cheaper than RL.** Delegation as RL needs a reward for "did delegating help" — hard to define. Harness-traced SFT sidesteps that by letting the harness define correct behaviour, then learning it.

## Gotchas & tricks

- **The harness is your prior.** Whatever delegation pattern your harness encodes is the only one the model can learn. Bake in flexibility deliberately.
- **Subagent context constraints matter.** If subagents return raw outputs instead of summaries, the main agent's context fills up at inference even after SFT. The harness must enforce summary-only returns.
- **Watch for shortcut learning.** A model can memorize the surface form of delegation (always delegate at step 3) without learning when delegation is appropriate. Diverse harness invocations + filtering for varied trajectory shapes mitigate this.
- **Compatible with downstream RL.** Use SFT to get correct delegation behaviour, then RL with task rewards for fine-tuning success rates.
- **Don't over-summarize.** Summaries that drop critical evidence make the main agent fail downstream. Calibrate the harness's summary instruction to retain the smallest sufficient set.

## Sources

- Paper: *SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research* — Ning et al., Tsinghua / Peking U. / Ant Group / Renmin U., 2026 — [arXiv 2606.09730](https://arxiv.org/abs/2606.09730).
- Background: rejection-sampling for SFT data curation — [rejection-sampling](../post-training/rejection-sampling.md).

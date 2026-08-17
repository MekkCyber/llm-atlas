# DarwinX

*Depth — evolutionary search over agent harnesses with a preserve-and-extend contract and an archive of lineages for recombination.*

**TL;DR:** DarwinX treats agent self-improvement as **selection over a population of harnesses** — prompts, tools, skills, and control flow — with the underlying model frozen. Every proposed variant must **extend coverage without regressing** existing tasks (the "preserve-and-extend contract"), an archive keeps alternative lineages for later recombination, and mutations are proposed from three evidence streams (own failures, teacher critique, self-derived hypotheses) through one shared edit interface. Fitness is each benchmark's own verifier — no gold solutions, no hand-picked winners.

**Prereqs:** [_harness-optimization](_harness-optimization.md), [rlvr](../post-training/rlvr.md)
**Related:** [agent-skills](agent-skills.md), [procedural-memory](procedural-memory.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md)

---

## What it is

A frozen-model, verifier-driven optimizer for agent *harnesses*. Where the classic self-improvement loop is single-lineage — one harness edits itself, one round at a time — DarwinX runs a **population** of harnesses under evolutionary pressure. The unit of iteration is a full harness (prompt bundle + tool schemas + skill library + control flow); the operator that produces variants is a language-model edit conditioned on the parent harness plus evidence about its failures; the selection rule is monotonic coverage.

## How it works

The loop, per round:

1. **Sample parents** from the current population + the archive of past lineages.
2. **Propose variants.** For each parent, an editor LM proposes changes derived from: failure traces (what went wrong on which task), teacher feedback (a stronger critic model), and self-derived reflections. All three feed the same edit interface — insert/rewrite/delete over the harness DAG.
3. **Evaluate** every variant on the training-side task suite. The verifier for each benchmark is that benchmark's own scorer.
4. **Admit under the preserve-and-extend contract**: a variant is admitted only if it wins on ≥1 previously failed task *and* does not regress on any task the parent already passed. Ties on failed tasks are resolved by cheaper harness.
5. **Archive** admitted variants (and periodic non-admitted-but-interesting variants) for future recombination.
6. **Recombine** occasionally by mixing skill libraries and tool schemas across archived lineages.

Nothing in the loop uses gold trajectories or hand-picked winners — the verifier is the whole selection signal.

## Why it matters

- Frozen model = cheap iteration and easy transfer. A DarwinX-tuned Terminal-Bench harness in the paper transferred **unchanged to SWE-bench Verified**.
- Preserve-and-extend gives long-horizon compounding. Single-lineage self-editing degrades on multi-benchmark evals because each fix regresses something else; DarwinX makes those regressions inadmissible by construction.
- The paper reports ≈17-point average gains across four benchmarks in one loop: Terminal-Bench 2.1 +7.7 → **83.2%** on a matched base (verified frontier **84.7%** on stronger base), TerminalWorld held-out **68.3%**, WebArena-Infinity real-task pass@1 **43.5% → 93.0%** audit-clean.

DarwinX is strongest evidence to date that **harness-level search compounds** where single-lineage self-editing plateaus.

## Gotchas & tricks

- **Preserve-and-extend needs a real regression suite.** If your "already-passed" set is small or drifty, everything looks admissible and the contract stops binding.
- **Editor-model bias.** The editor LM can systematically miss a class of failures (e.g. concurrency bugs). Rotate editors or occasionally hand-seed variants.
- **Archive pruning matters.** Unbounded archives blow up recombination cost; the paper prunes by novelty + recent utility.
- **Verifier is now the attack surface.** Any exploitable verifier gets exploited — audit for false-positive passes (the paper distinguishes raw and *audit-clean* pass@1 for exactly this reason).
- **Not a substitute for the model.** DarwinX raises the harness above the model's floor; it doesn't move the ceiling. Weak base → weak DarwinX.

## Sources

- Paper: *DarwinX: Evolving Agent Harnesses Through Natural Selection* — Zhang, Dai, Tan, Yang, Mullur, Hoang, Hu, Zhu, Mui, Savarese, Xu, Chen (Salesforce AI Research / Agentforce), 2026, [arXiv:2608.07545](https://arxiv.org/abs/2608.07545)

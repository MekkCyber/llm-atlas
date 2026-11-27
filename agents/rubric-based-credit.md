# Rubric-Based Credit (RBC)
*Depth — dense RL rewards derived from per-prompt rubrics that score each intermediate retrieval / planning step, without requiring ground-truth answers or rendered outputs.*

**TL;DR:** For knowledge-intensive agents (search, planning, tool use), the final output reward is sparse and often unavailable at training time (e.g. a video generation agent whose videos are expensive to render). **Rubric-Based Credit (RBC)** builds a per-prompt rubric — a small checklist derived from the prompt itself — and scores each intermediate agent step against it. The rubric becomes a dense, verifiable, no-ground-truth-needed reward. Introduced in the 2026 **Loupe** video-generation agent, which couples RBC with SFT (RBC-SFT) and GRPO (RBC-GRPO) and reports **MiniMax-H3 overall score 32.80 → 70.67** on Search-VBench across three video generators.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [memory-agents](memory-agents.md) · [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md) · [README](README.md)

---

## What it is

A reward design that turns an open-ended agentic episode into a dense, verifiable signal. For each prompt:

1. **Rubric generation.** A rubric model reads the prompt and emits a checklist of concrete criteria the final behavior should satisfy — e.g. "mentions the discovery date," "cites a physicist," "uses an image from a scientific source." The rubric is prompt-specific but produced automatically, not hand-written.
2. **Step scoring.** As the agent acts (searches, retrieves, plans, generates sub-prompts), each intermediate step is scored against the rubric: how many criteria does *the information gathered so far* satisfy?
3. **Dense reward.** The per-step rubric score becomes the RL reward. Because the rubric is explicit text, a judge model can evaluate it reliably without needing the final rendered output.

## How it works

The Loupe pipeline uses RBC in two training modes:

- **RBC-SFT.** Collect agent rollouts, keep those with high final rubric scores, SFT on them. Rejection-sampling SFT with a rubric-derived filter in place of a gold-answer filter.
- **RBC-GRPO.** Standard GRPO ([../post-training/grpo.md](../post-training/grpo.md)) but with the group reward replaced by the rubric score. Because the reward is dense (multi-criterion per step), the advantage estimate is lower-variance than pure final-outcome rewards.

The rubric itself is produced by the same base model the agent runs on — a self-rubric loop — gated by the prompt's structure. The rubric model is instructed to produce only *verifiable* criteria ("mentions X," "cites Y") and not subjective ones ("is elegant").

## Why it matters

- **No ground truth required.** Classical video-generation rewards need rendered video + a learned video quality metric — both expensive and hackable. RBC replaces this with a cheap rubric check on *information gathering*, producing training signal before any video is ever rendered.
- **Dense signal from a sparse task.** Each step gets scored, not just the final one. This is the agentic analogue of what PRMs ([../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)) do for math reasoning — but with the rubric expressed in natural language instead of a learned per-step classifier.
- **Transferable beyond video.** Any knowledge-intensive agent (search-and-synthesize, deep research, plan-then-execute) has the same structure: the *process* is verifiable even when the *output* isn't. Loupe is the first worked example; the pattern is general.

## Gotchas & tricks

- **Rubrics can be hacked.** A policy that learns to satisfy "mentions X" by trivially inserting X anywhere, without context, is reward-hacking. Pair with [../safety/reward-hacking.md](../safety/reward-hacking.md)-style counterfactual audits and require rubric criteria to be *verifiable in context*, not just textual matches.
- **Rubric quality bounds everything.** If the rubric model produces vague or unverifiable criteria, the signal is noise. Loupe uses a rubric-generation prompt tuned to require concrete, evidence-grounded criteria.
- **Not a substitute for final-output rewards when you can get them.** RBC gives dense training signal and makes RL converge faster on knowledge-intensive tasks, but final-output quality is still what users care about. Compose RBC with a final-output reward where available.
- **Judge-model cost accumulates.** Scoring every step against the rubric needs a judge LM per step. For long trajectories this dominates per-rollout cost; cache aggressively and prefer small judge models.

## Sources

- Paper: *Loupe: Reinforcing Knowledge-Intensive Video Generation Agents* — An, Fang, Fang, Lyu, Li, Yu, Dai, Zeng, Xu, Zhu, Zhang (PKU / USTC / DP Technology / HUST / NTU / SJTU / Ant Group), 2026, arXiv 2611.31895.
- Related: [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md) — process rewards for reasoning; RBC is the natural-language-rubric cousin.
- Related: [../post-training/_rewards.md](../post-training/_rewards.md) — reward family taxonomy; RBC sits at the junction of rule-based and learned (judge-as-rubric-checker).

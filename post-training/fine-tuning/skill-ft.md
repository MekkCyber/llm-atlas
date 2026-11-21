# Skill-FT and AutoMaster
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-tunes on teacher trajectories whose *behaviours have been steered by explicit "skills"* — short natural-language directives attached to the teacher's prompt — instead of raw successful rollouts. **AutoMaster** is the LLM agent that designs the skill set, selects which trajectories to keep, and refines both from student eval feedback. 4B and 9B students trained this way beat students trained on the same teacher's raw successful trajectories across Terminal, Search, and SWE benchmarks. The same model can play both teacher and student roles.

**Prereqs:** [../_post-training.md](../_post-training.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [../grpo.md](../grpo.md), [../../agents/README.md](../../agents/README.md)

---

## What it is

Agent SFT with a skill-conditioned data pipeline:

- A **skill** is a short natural-language rubric (e.g. *"verify environmental assumptions before issuing a tool call"*, *"consolidate redundant file reads"*).
- **Skill-FT** is the training recipe: the teacher model rolls out *conditioned on a skill prompt*, producing cleaner trajectories; the student fine-tunes with the skill kept in the input.
- **AutoMaster** is an LLM-agent controller that proposes new skills, filters trajectories, and iterates both based on student eval scores.

## How it works

1. **Seed skill set.** A small human-written or LLM-generated set of behavioural rubrics for the target task family (Terminal, Search, SWE).
2. **Teacher rollouts under skills.** For each (task, skill), the teacher model generates a trajectory. The skill sits in the system prompt; the task is the user turn.
3. **Keep-success filtering (per skill).** Keep trajectories that both solve the task *and* audibly follow the skill (a cheap judge checks adherence).
4. **Student SFT.** Fine-tune a smaller (or the same) model on {(task + skill-in-prompt → trajectory)} pairs. Loss is standard next-token.
5. **Eval loop.** Run the student on a held-out task set. AutoMaster reads the eval traces, proposes new skills for under-covered failure modes, removes unproductive skills, and re-selects the training mix.
6. **Repeat.** A few iterations suffice; AutoMaster is itself an LLM, not a trained RL controller.

### Why keep the skill in the student's input

At inference time, the skill isn't needed — the student has internalised the behaviour. But keeping it during training *improves* learning: the skill acts as a context prefix that disambiguates which strategy the trajectory demonstrates, so the student's gradients align with that strategy instead of averaging over the teacher's whole distribution.

## Why it matters

- **Decouples "which trajectories succeed" from "which trajectories should be imitated."** A successful trajectory can still be full of wasted steps, false confidence or shallow verification; skill filtering targets *why* a trajectory is good.
- **No frontier-API dependency.** The same model is teacher and student, so small labs can run this without paying for GPT-5-class rollouts.
- **Composes with RL.** Skill-filtered SFT is a strong initialisation for subsequent [GRPO](../grpo.md) / RLVR.

## Gotchas & tricks

- **Skill redundancy.** Overlapping skills double-count trajectories; AutoMaster's dedupe step matters.
- **Reward hacking of the adherence judge.** If the adherence check is a cheap LLM-judge, the teacher can learn to produce trajectories that *claim* to follow the skill without doing so. Spot-check manually.
- **Skill transfer to new tasks.** Skills defined for Terminal don't automatically apply to SWE; AutoMaster regenerates the skill set per task family in the paper's setup.
- **Tested at 4B and 9B.** Scaling behaviour above that is open.

## Sources

- Paper: *AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher* — Shuang Sun, Fanzhe Meng, Xinyu Tang, Ruihua Song, Wayne Xin Zhao (Gaoling School of AI, Renmin University of China), 2026 — primary source for Skill-FT and the AutoMaster controller.

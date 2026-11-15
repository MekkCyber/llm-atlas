# Judge-as-Teacher for Long-Horizon Agents
*Depth — the actor/judge/teacher decoupling and its use as a dense GRPO signal.*

**TL;DR:** Decouple three roles a pretrained model can play in a long-horizon agent: **actor** (pick the next action), **judge** (score a completed trajectory), and **teacher** (produce signal to improve another policy). The paper shows that models which **fail as actors** on subgoal-ordering tasks can still reliably separate helpful from unhelpful actions when given the full trajectory — i.e., they are strong judges. Plugging those judgments into GRPO as a dense per-step signal lifts a small (1.5B) policy on ALFWorld and WebShop without needing a frontier actor.

**Prereqs:** [../post-training/grpo](../post-training/grpo.md), [../post-training/cot-reward-model](../post-training/cot-reward-model.md)
**Related:** [../post-training/_rewards](../post-training/_rewards.md)

---

## What it is

A reframing of agent post-training. Rather than ask "which large model should drive my agent?" the paper asks "which role is each model good at?" and finds the mapping is non-trivial: actor ability ≠ judge ability.

- **Actor** — must pick the right action *prospectively*, under partial information, with subgoals in correct order.
- **Judge** — reads the *completed* trajectory and labels each action as helpful or not, which is a much easier decision because it is *retrospective*.
- **Teacher** — contributes signal (judgments, demonstrations, reward shaping) to another policy's training loop.

## How it works

- **Collect rollouts** of a small student policy π (1.5B) on the target task.
- **Judge pass**: a larger frozen "judge" model (one of the six open models the paper tests) reads each completed trajectory and labels each step {+1, −1, 0}.
- **GRPO with auxiliary signal**: the student's standard GRPO objective is augmented with a dense per-step reward from the judge's labels. The group baseline continues to come from GRPO's group-of-rollouts normalization.
- **Rollouts × judgments** are the only artifacts handed over — no actor-mode calls to the big model in the training loop, which is the cost-saving.

## Why it matters

- **"Bad actor, good teacher" is cheap.** Running the big model only in judge mode (post-trajectory) is strictly cheaper than running it as the actor (step-by-step) and still improves the student.
- **New reward source for agent RL.** Adds a dense signal that is available without ground-truth rewards, which agentic benchmarks often lack per step.
- **Clarifies model selection.** For agent harnesses, the right question is "best actor" vs "best judge" — and they can be different models.

## Gotchas & tricks

- Judge reliability depends on trajectory legibility; judges struggle when the environment feedback is sparse or when the judge model is weak at reading long traces.
- The auxiliary signal is additive to the environment reward; weighting matters and the paper tunes it per task.
- The actor/judge gap is most pronounced on subgoal-ordering tasks; on single-shot tasks the two converge.

## Sources

- Paper: *Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents* — Zhu et al., Zhejiang U. / Baidu, 2026 — arXiv 2611.17167 — https://arxiv.org/abs/2611.17167
- Benchmarks referenced: ALFWorld, WebShop.

# RGPO — Rationale-Guided Policy Optimization

*Depth — scaffold reasoning rollouts with a ground-truth rationale in the prompt, then train on the student's own improved answer **without** the rationale.*

**TL;DR:** When you have ground-truth rationales (worked solutions, step-by-step answers), the usual options are to SFT on them (memorizes wording, hurts generalization) or to ignore them (wastes signal). RGPO uses them as *temporary scaffolds*: the rationale appears only in the prompt the model sees at *generation* time; the resulting improved trajectories are then trained on under the original *unguided* prompt. The scaffold does its job then disappears.

**Prereqs:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md), [long-cot-rl.md](./long-cot-rl.md)
**Related:** [../rejection-sampling.md](../rejection-sampling.md), [../rl-prompt-curation.md](../rl-prompt-curation.md), [prm.md](./prm.md)

---

## What it is

A training pipeline that decouples **where the rationale lives** from **what the model is trained on**:

1. For each prompt `q` with ground-truth rationale `R`, generate rollouts conditioned on `q + R` (the scaffolded prompt).
2. Keep only rollouts that improve over the model's unguided baseline — rejection-filter by RLVR verifier.
3. **Relabel** these improved rollouts as responses to `q` alone (strip the rationale).
4. Train with a GRPO-style policy update on `(q, kept_rollout)` pairs.

The student never sees the rationale in training-time context — only in generation-time context. The policy learns to produce the improved answers *given* the raw prompt.

## How it works

### Scaffold-generate step

Prompt format:

```
q + "\n\nHere is a worked solution to think through:\n" + R + "\n\nYour answer:\n"
```

Student samples `o ~ π_student(· | q + R)`. The rationale is prefix context for exploration; it is not a target.

### Rejection filter

Verify `o` with the RLVR verifier (math-answer match, test-pass, etc.). Keep only successful `o`.

Additionally the paper discards rollouts whose outcome under `(q + R)` doesn't beat the model's own unguided baseline on `q` — this prevents training on cases where the model would have been right anyway without the scaffold.

### Relabel + GRPO update

Build training pairs `(q, o)` — prompt without the rationale, response as sampled. Then standard GRPO on this relabeled dataset: group `G` relabeled samples per prompt, compute group-normalized advantages, PPO-clip update.

The relabel trick means the policy gradient optimizes `log π_student(o | q)`, not `log π_student(o | q + R)` — the model learns to **produce** the improved behavior from the raw prompt.

### Why this is not just SFT-on-success

Pure "SFT on the improved rollouts" loses two things RGPO keeps:

- **Group-relative baseline.** GRPO uses the group mean as the advantage baseline; some rollouts are better than others *within a successful group*, and RGPO's gradient reflects that.
- **KL anchor to a reference policy.** As in standard GRPO, the KL term to `π_ref` prevents distribution collapse.

## Why it matters

- **Scaffolded rationales are a cheap and generic exploration prior.** You get the exploration benefit of a hint without baking the hint's wording into the policy.
- **Works in both LM and VLM settings** per the paper — consistent gains over RLVR baselines across language-only and vision-language reasoning tasks.
- **The pattern — "scaffold at generation time, train on scaffold-stripped relabel" — generalizes.** It should work for tool-use traces (demonstrate the tool call sequence, train on the raw task), for in-context examples (few-shot at generation, zero-shot at training), and for any other kind of prompt-level help you'd rather the model internalize than imitate.

## Gotchas & tricks

- **Needs reliable rationales.** If `R` is wrong, the "improved" rollouts drift toward wrong-but-confident. Pair with a verifier that is strict about final answers.
- **The baseline matters.** Comparing scaffolded-rollout success to unguided-rollout success is how you detect the scaffold was actually useful; without that gate RGPO becomes "SFT on correct answers with an RL loss function."
- **Prompt-template leakage.** If the training-time prompt format differs from the generation-time format, the model may learn to associate the "worked solution: …" cue with success and underperform at inference where the cue is absent. Keep training-time formatting identical to inference.
- **Group size `G`** interacts with scaffold strength. If the scaffold is very strong, most of a group's rollouts succeed and the group-mean baseline becomes small — advantages shrink. Mix in some unguided rollouts to re-create variance.
- **For reasoning specifically**, the rationale is the model's own long CoT target distribution. Scaffolding is one tool; others include PRM-filtered sampling ([prm.md](./prm.md)) and long-CoT RL ([long-cot-rl.md](./long-cot-rl.md)).

## Sources

- Paper: *Rationale-Guided Policy Optimization: Learning to Reason with Adaptive Rationale Scaffolding* — Pham et al., NYU + Buffalo + Monash, 2026 — [arXiv:2610.07342](https://arxiv.org/abs/2610.07342).
- Related: *GRPO* — Shao et al., DeepSeek, 2024 — see [../grpo.md](../grpo.md).
- Related: *Star / Rest-EM / Expert Iteration* — the broader "generate-then-filter-then-SFT" lineage RGPO improves on.

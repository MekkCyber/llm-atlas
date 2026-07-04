# Step-Aware RL — Position-Weighted Credit Assignment for Reasoning
*Depth — a GRPO variant that penalizes tokens more heavily the earlier they appear in a failed reasoning trajectory.*

**TL;DR:** With outcome-only rewards, [GRPO](../grpo.md) broadcasts the same advantage to every token of a response. Early wrong reasoning steps then get the *same* penalty as later downstream tokens that were forced into the wrong path — the true blame gets diluted. **MRPO** (Medical Reasoning-aware Policy Optimization, but the technique is general) weights the penalty **exponentially higher on earlier tokens of failed rollouts**, leaves successful rollouts alone, and reduces early-stage reasoning failures 64.0 % → 13.0 % on medical VQA — while beating GRPO across three MLLM backbones and beating a 34B specialist with an 8B model.

**Prereqs:** [../grpo.md](../grpo.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [prm.md](prm.md), [orm.md](orm.md), [../rlvr.md](../rlvr.md)

---

## What it is

Long-CoT RL rewards the final answer only, so credit assignment is uniform across the trajectory. Empirically, wrong answers are dominated by **failure cascades**: an early misperception or miscalculation forces the rest of the chain into an unrecoverable path. All those later tokens get the same negative advantage as the true culprit, wasting gradient on tokens that were locally reasonable.

Step-aware RL fixes this by *asymmetrically* reweighting the credit on failed trajectories: earlier tokens get exponentially larger penalties, later tokens get smaller ones. Successful trajectories are left as-is — no need to change what's already working.

## How it works

Standard GRPO advantage: $A_i$ scalar per response $o_i$, broadcast to every token.

MRPO's modification: on a failed response ($A_i < 0$), scale per-token advantage by an exponentially decaying weight over token position:

$$
A_{i,t} = A_i \cdot w(t), \qquad w(t) = \exp\!\left(-\lambda \cdot t / |o_i|\right)
$$

Earlier tokens (small $t$) get $w(t) \approx 1$ — full penalty. Later tokens get $w(t) \ll 1$ — diminished penalty. $\lambda$ controls the sharpness of the schedule.

On successful trajectories ($A_i > 0$), leave $A_{i,t} = A_i$ unchanged. The asymmetry preserves the "don't punish success" invariant while sharpening blame on failure.

The rest of the GRPO objective (clipped ratio, KL to reference) is unchanged.

## Why it matters

- **Approximates PRM benefits without a PRM.** [Process reward models](prm.md) are the principled fix for uniform credit assignment, but training and querying one is expensive. Position weighting is a cheap heuristic that captures most of the gain — no PRM, no per-step verifier.
- **Cascades are the dominant failure mode.** MRPO reduces early-stage reasoning failures **64.0 % → 13.0 %** — validating both the mechanism (early errors compound) and the fix (weight them heavier).
- **Transfers across backbones.** Reported to consistently beat vanilla GRPO on three MLLM backbones, and an 8B MRPO-trained model beats a 34B specialist (HuatuoGPT-Vision) by 2.79 points on medical VQA.
- **Trivial to implement.** A one-line change to the GRPO advantage broadcast. No new networks, no new labels.

## Gotchas & tricks

- **The decay rate $\lambda$ is a real hyperparameter.** Too flat and you're back to GRPO; too sharp and only the first few tokens get any signal. Grid search on validation loss.
- **Only re-weight negatives.** Symmetric weighting on positive trajectories overweights the *opening* of a successful chain — which is often just a restatement of the problem — and underweights the crucial final steps.
- **Not a substitute for process rewards when you can afford them.** A trained PRM captures *which* step was wrong; position weighting is a monotone proxy that assumes wrongness concentrates early. On tasks with mid-chain errors, PRM will win.
- **Works with any sequential-reasoning task.** Framed as "medical" in the paper, but math, code, and multi-hop QA all exhibit the same cascade pattern.

## Sources

- Paper: *Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning* — Jung et al., 2026 — [arXiv:2606.31825](https://arxiv.org/abs/2606.31825).
- Related: *DeepSeekMath* — Shao et al., 2024 — the GRPO baseline this extends.
- Related: [prm.md](prm.md) — process reward models, the principled alternative.

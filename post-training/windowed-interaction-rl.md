# Windowed Interaction RL (WIRL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Autonomous ML-engineering agent trajectories contain many expensive experiments, each returning a sparse outcome signal many turns after the decisions that caused it. **WIRL** chops the trajectory into **execution windows** terminated by experiment results, assigns each window a *shared* outcome-based reward, and uses **asynchronous rollout collection + policy correction** so updates can land before the trajectory finishes.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [multi-session-agentic-rl.md](./multi-session-agentic-rl.md)

---

## What it is

Long-horizon agent RL faces three stacked problems:

1. **Sparse terminal reward.** The outcome is a Kaggle-style score at the end of hours of experimentation.
2. **Expensive atomic step.** Each experiment ("run this training job and read the loss") costs minutes to a GPU-hour, so you cannot afford many trajectories.
3. **Within-trajectory credit assignment.** The score at the end reflects many decisions back in the context.

WIRL answers (3) with windows and (2) by letting policy updates land asynchronously, so the second half of a trajectory benefits from improvements learned on the first half.

## How it works

### Windows terminated by experiment outcomes

A full MLE trajectory is a sequence of agent turns punctuated by *experiment calls* (train/eval runs). Group the turns between successive experiment outcomes into one **window** $W_j$. All turns in $W_j$ share a single reward:

$$
r(W_j) = f(\text{outcome}_j, \text{outcome}_{j-1})
$$

typically a bounded $\Delta$-score shaped by whether this experiment improved on the running best.

### Shared-reward GRPO inside each window

Within a window, use the standard GRPO objective but with the window-level $r(W_j)$ broadcast across its tokens. Across windows in the same trajectory, advantages are *not* summed — each window stands alone for credit assignment. (This is the WIRL/Mosaic distinction: WIRL cuts on *experiment* boundaries, Mosaic on *session* boundaries.)

### Asynchronous collection + policy correction

A traditional on-policy RL loop blocks on finishing a trajectory before updating weights. For MLE trajectories spanning hours, that is unacceptable. WIRL collects windows into a shared queue and runs updates as soon as a window completes, with an off-policy **correction term** (importance ratio) relative to the policy snapshot used when that window was generated. The correction keeps updates roughly on-policy and bounds drift.

### Hierarchical agentic SFT seed

Before WIRL, the model is fine-tuned via **hierarchical agentic SFT**: outcome-guided supervision on high-level plans *and* on low-level tool/code steps. This gives the base model the tool-use scaffolding WIRL then refines.

## Why it matters

- **Makes RL tractable at experiment granularity.** You get a credit signal every few turns instead of every few hours; the policy actually learns from each experiment outcome.
- **Lifts open MoE agents into frontier range.** On MLE-Bench Lite, WIRL takes the base model's medal rate from **37.9% → 72.7%** and matches larger closed frontier systems (Kimi-K3, GLM-5.3).
- **Doesn't require a critic.** Reward is sample-based (outcome from an actually-run experiment), so no value network at trajectory scale.

## Gotchas & tricks

- **Window boundary definition matters.** Experiment-call boundaries are natural; "every N turns" would mix outcomes across experiments and reintroduce the credit-assignment problem.
- **Reward shaping around failed experiments.** A failed training job is still an outcome and still terminates a window; the reward function needs a non-degenerate value for crash / OOM / timeout.
- **Async + correction bounds drift, doesn't eliminate it.** Importance-ratio clipping, as in PPO, applies; the paper uses a conservative clip range.
- **Scaffold vs model.** Hierarchical agentic SFT gives WIRL something to refine — training WIRL from base without the SFT seed is unlikely to converge.
- **Scope.** Reported on 35B-A3B MoE on MLE-Bench Lite / NatureBench Lite / scaffold+algorithm optimization. Generalization to non-ML coding agents is plausible but not shown.

## Sources

- Paper: *MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering* — Shi, Zhang, Wu, Zhao, Feng, Li, He, Yan, Zhang (Shanghai AI Lab + ECNU), 2026 — [arXiv:2610.22344](https://arxiv.org/abs/2610.22344).
- Code: https://github.com/InternScience/MLExpert

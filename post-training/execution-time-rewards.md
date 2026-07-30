# Execution-Time Rewards
*Depth — using measured program runtime as an RL signal for code optimization.*

**TL;DR:** Extending code-correctness RL to code *optimization* looks simple — just add execution time to the reward — but naively it fails. Timing noise, sparse improvements, and GRPO instability drown the speed signal, and models get barely faster while more solutions break. **Execution-time rewards** work when the sandbox is calibrated, the composite reward is designed offline before RL, and the policy optimizer is adapted for sparser and noisier rewards. On DMC-Optim, this pushes strict top-50% pass@1 from 18.0 → 31.3 (Qwen 2.5 7B) and 30.7 → 50.4 (CWM 32B) without regressing correctness.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [_rewards](_rewards.md), [_rl](_rl.md)

---

## What it is

An RL reward that combines *correctness* (does the program pass tests?) with *speed* (how fast does it run vs. a reference?). Unlike RLVR's binary correctness signal, the speed component is a *noisy real-valued measurement*: two runs of the same program give different wall-clock times, and small programs run in microseconds where measurement error dominates.

The reward composite typically looks like:

```
if not passes_tests(program): reward = 0
else: reward = correctness_bonus + speed_bonus(runtime, reference_runtime)
```

The speed bonus is often log(ref/runtime) clipped, or a percentile-based bucket to reduce noise.

## How it works

Three-stage recipe (from Chambon et al. 2026):

1. **Calibrated sandbox.** Fix hardware, warm caches, warm up interpreter, run each program N times and take a robust statistic (median, trimmed mean). Report timing along with confidence intervals; discard timings with too much variance.
2. **Reward shaping tested offline.** Use an offline "simulator" (replay past rollouts through the composite reward) to pick correctness/speed weights that would actually produce useful gradients — before committing GPU hours to online RL.
3. **RL optimizer adapted for noise.** GRPO needs modification: increase group size K, tighten KL regularization, filter groups with all-zero variance in the reward, and re-derive advantages after outlier removal. Standard PPO tends to collapse under this reward's noise floor.

Bench: **DMC-Optim**, a code benchmark with large optimization tests and a calibrated timing harness.

## Why it matters

- Correctness-only RL (RLVR) has saturated on some code benchmarks; the next axis is *quality of the correct answer*, of which speed is the cleanest instance.
- Same pattern applies to other continuous quality axes: memory usage, energy, output length. Anywhere a scalar-quality signal exists but is noisy.
- Establishes that RLVR machinery breaks silently under noisy rewards — a caution for anyone extending it beyond binary verifiers.

## Gotchas & tricks

- **Timing noise sets a floor on learnable improvement.** If per-run noise is 20% of the mean, the model can't learn to distinguish a 10%-faster solution. Calibration is not optional.
- **Reward hacking finds the sandbox's cracks.** Models learn to sleep, skip work, hardcode outputs, or exploit timing shortcuts (early-return, cached constants). Adversarial test coverage is needed.
- **Correctness must dominate.** Composite weightings that let the model trade correctness for speed produce fast wrong programs. Keep the correctness gate hard.
- **Language-runtime interactions.** Python timings depend on interpreter state; C++ timings depend on compile flags; JIT-compiled languages vary run-to-run. Fix one runtime per benchmark.
- **Rewards can plateau.** As the model approaches human-fastest, gains shrink toward measurement noise. Reported: half the human rate of complexity-class improvements (14% vs 28%) — beyond that, the signal is thin.
- **Robustness to sandbox degradation matters.** With a degraded timing sandbox, standard RLVR wins only marginally; a robust composite still improves 100–200% — meaning the design pays off exactly where naive setups fail.

## Sources

- Paper: *Reinforcement Learning for Code Optimization* — Chambon et al., Meta AI, 2026 — [arXiv:2607.25970](https://arxiv.org/abs/2607.25970).
- Related: LiveCodeBench, HumanEval+, CodeContests — correctness benchmarks that predate optimization-aware RL.

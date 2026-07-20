# MeanFlow (with MeanFlowNFT)
*Depth — few-step generators that predict average velocities over time intervals, and forward-process RL for them.*

**TL;DR:** MeanFlow is a class of fast few-step generators that predict **average velocity over a time interval** instead of instantaneous velocity, letting a single network call cover a whole diffusion step. **MeanFlowNFT** extends the training-free DiffusionNFT forward-process RL framework to MeanFlow by building an induced instantaneous-velocity predictor from the MeanFlow identity — reward optimization without reverse trajectories or likelihood estimation, and MeanFlow's few-step sampling is preserved.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Standard flow-matching models predict an *instantaneous* velocity $v(x_t, t)$ that must be integrated across many small steps. **MeanFlow** predicts the average velocity $\bar{v}(x_t, t \to t')$ over a finite interval, so one network call advances the state by a large step — fast few-step sampling.

**MeanFlowNFT** answers a follow-up question: how do you fine-tune a MeanFlow generator with a task-specific reward *without* running expensive reverse trajectories or estimating likelihoods? The answer is to reuse the forward-process RL framework (DiffusionNFT) via a mathematical bridge — the MeanFlow identity — between instantaneous and average velocities.

## How it works

### MeanFlow sampling

Given a learned $\bar{v}_\theta(x_t, t, t')$, generate a sample by stepping from noise $x_1 \sim \mathcal{N}(0, I)$ to data $x_0$ in $K$ steps, each of the form

$$x_{t'} = x_t + (t' - t) \cdot \bar{v}_\theta(x_t, t, t')$$

With $K$ small (say 4), sampling is orders of magnitude faster than standard $\sim 50$-step flow matching.

### The MeanFlow identity

The bridge from average to instantaneous velocity used at training time:

$$\bar{v}(x_t, t, t') = \frac{1}{t' - t} \int_t^{t'} v(x_\tau, \tau)\, d\tau$$

This is the definition, but the paper leverages a differential identity relating $\bar{v}$ and $v$ that lets you *induce* an instantaneous predictor $v_\theta$ from a MeanFlow model $\bar{v}_\theta$ without training a separate network.

### MeanFlowNFT — forward-process RL

DiffusionNFT does RL on flow models by optimizing the instantaneous velocity predictor against a reward, using forward-process rollouts (no reverse trajectories, no likelihoods). Its objective assumes an instantaneous-velocity model.

MeanFlowNFT:

1. Construct the induced $v_\theta$ from $\bar{v}_\theta$ via the MeanFlow identity.
2. Apply the DiffusionNFT objective to $v_\theta$.
3. Backprop the reward-optimizing loss through the identity into $\bar{v}_\theta$.
4. Sample with the original $\bar{v}_\theta$ for few-step generation — no change to inference.

The paper proves MeanFlowNFT inherits DiffusionNFT's **strict policy-improvement guarantee**.

## Why it matters

- **RL on the fastest samplers.** Reward-driven fine-tuning of few-step generators has been awkward — likelihood-based objectives don't apply cleanly. MeanFlowNFT closes that gap without giving up MeanFlow's speed.
- **No reverse trajectories, no likelihoods.** The DiffusionNFT + MeanFlow identity combination avoids the two most expensive operations in RL-for-diffusion.
- **Composes with the broader diffusion-RL landscape.** DDPO, DPOK, DiffusionNFT and now MeanFlowNFT are variants of a shared problem — align a stochastic generator to a scalar reward — and pick different tradeoffs.

## Gotchas & tricks

- **The induced instantaneous predictor is not exact everywhere.** The MeanFlow identity is a differential relation; at the interval boundaries the induced $v_\theta$ is well-behaved but numerical care is required.
- **Reward shape matters as much as in LLM RL.** Sparse, per-image rewards produce noisy gradients; consider dense per-timestep shaping when available.
- **Preserved sampling is the whole point.** If a "MeanFlow-RL" variant changes the sampling procedure, you've lost the speed win — MeanFlowNFT is explicit about keeping the sampler untouched.
- **Not all MeanFlow variants are covered.** Piecewise MeanFlow or curriculum-trained variants may need tweaks to the induced predictor.

## Sources

- Paper: *MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators* — Huang, Zhou, Zhang, Bo, Pang — Tencent Hunyuan / HKUST, 2026.
- Paper: *DiffusionNFT* — the forward-process RL baseline MeanFlowNFT extends.
- Paper: original MeanFlow — few-step average-velocity generator.

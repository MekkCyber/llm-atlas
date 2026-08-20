# Evolution Strategies for LLM Fine-Tuning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replace backprop-based RL post-training with **Evolution Strategies (ES)**: at each step, sample small parameter perturbations, roll out each perturbed model, and update the base parameters by a reward-weighted mean of the perturbations. Because ES uses only *forward* passes, it needs inference-level GPU memory — enabling full-parameter fine-tuning of large LLMs on modest hardware. Competitive with agentic RL on long-horizon tasks.

**Prereqs:** [_rl.md](_rl.md), [ppo.md](ppo.md), [grpo.md](grpo.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md)

---

## What it is

Evolution Strategies (ES) are a family of black-box, gradient-free optimisation methods. For LLM post-training the update rule is:

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} R(\tau_i) \cdot \frac{\epsilon_i}{\sigma}
$$

where $\theta$ are the model parameters, $\epsilon_i \sim \mathcal{N}(0, \sigma^2 I)$ are small Gaussian perturbations, and $R(\tau_i)$ is the return of the trajectory produced by the perturbed model $\theta + \epsilon_i$.

## How it works

### The training loop

1. Sample $N$ perturbations $\epsilon_i$ (antithetic pairs $\pm \epsilon_i$ are common).
2. For each $i$, deploy the perturbed model $\theta + \epsilon_i$ and let it complete a full agent trajectory in the environment (WebArena, coding harness, browser).
3. Score each trajectory with the same reward function you would give an RL trainer.
4. Update $\theta$ by a reward-weighted mean of the perturbations.

Because the update needs only rollouts (forward passes), the trainer's memory footprint is just *N × (inference memory)* — no activations, no optimizer state, no gradient tensors.

### Agentic ESOpt's twists (Zheng et al., 2026)

- **Online reward-weighted update.** Instead of a batch-mean like OpenAI-ES, update after each perturbation lands, keeping the learning signal fresh across a distributional shift in the agent's own behaviour.
- **Cosine-decayed perturbation scale.** $\sigma$ starts high (exploration) and decays on a cosine schedule to a small final value (exploitation). Mirrors RL's early-exploration / late-exploitation curve without needing entropy bonuses.
- **Full-parameter, no adapters.** Because memory is inference-scale, there's no reason to LoRA — the entire model is optimised.

### Trajectory-level credit assignment

RL back-propagates a per-token loss through possibly thousands of tokens of an agent trajectory. ES assigns credit *at the trajectory level*: one scalar return per rollout, weighted onto one perturbation vector. This removes the credit-assignment burden that scales badly with horizon length — the fundamental reason ES becomes competitive as trajectories get longer.

## Why it matters

- **Memory-bound labs unlock full-parameter fine-tuning.** Backprop-based RL on a 27B model needs multi-node ZeRO/FSDP; ES needs only $N$ inference workers. This changes who can do frontier-scale post-training.
- **Composes naturally with prompt-space evolution.** Because ES lives in a black-box interface, it stacks cleanly with skill-library evolution, DSPy, and TextGrad — one optimiser, one signal, two search spaces.
- **Horizon-agnostic.** ES's credit-assignment cost is independent of horizon length. As agent trajectories grow (500 turns of a coding agent, days of a browser agent), the RL-vs-ES trade-off tips in ES's favour.

## Main results (Zheng et al., 2026)

- **WebArena-Lite, full-parameter Qwen-3.5-27B:** ES beats no-skill baseline by **+6.69** points.
- **Test-time heuristic design (online prompt + parameter co-evolution):** wins in **28 of 36** settings vs matched agentic RL baselines.
- Advantage over agentic RL grows monotonically with horizon length in reported ablations.

## Gotchas & tricks

- **Antithetic sampling.** Always sample $+\epsilon$ and $-\epsilon$ pairs; halves the variance of the gradient estimator for free.
- **Perturbation scale matters more than learning rate.** Start $\sigma$ at ~0.02× the parameter magnitude; cosine-decay to ~0.005×.
- **Reward normalisation.** ES is scale-sensitive — normalise returns per batch (rank-shape or z-score) before applying weights.
- **Parallelism dictates population size.** $N$ needs to be at least as large as the number of inference workers available; too small and gradient variance dominates the signal.
- **Not a replacement everywhere.** For short-horizon, high-signal tasks (single-turn preference optimisation, DPO-style comparison), PPO/DPO still win — ES's advantage is *long-horizon* + *low-memory*.
- **Weight-noise vs prompt-noise.** ES perturbs weights; TextGrad perturbs prompts. Both are black-box; run them jointly for compounding gains.

## Sources

- Paper: *Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements* — Zheng, Chen, Ba, Wang, Teh, Lee — NUS / Oxford / SUSTech, 2026 — https://arxiv.org/abs/2608.17310
- Paper: *Evolution Strategies as a Scalable Alternative to Reinforcement Learning* — Salimans, Ho, Chen, Sidor, Sutskever — OpenAI, 2017 — foundational NES formulation.

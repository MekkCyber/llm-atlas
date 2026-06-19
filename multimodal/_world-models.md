# World Models

*Taxonomy — models that learn an internal simulator of the world's dynamics, ranging from passive video generators to operational state-tracking systems for embodied agents.*

**TL;DR:** A world model takes (history of observations, optionally actions) and predicts future observations. The space splits cleanly into two camps: **passive video generators** that produce pretty rollouts without state guarantees, and **operational world models** that maintain persistent internal state, can be queried about hypothetical futures, and run inside embodied control loops. The 2026 Kairos paper is the strongest current statement of the operational camp.

**Related taxonomies:** —
**Depth files covered here:** *(none yet — Kairos itself is the case study; depth files will be linked as the operational-world-model literature grows.)*

---

## The problem

Embodied agents need to predict what happens next: where will this object end up if I push it, what does the room look like behind me, what's the result of running this code. Doing this *as a planning capability* requires an internal simulator the agent can roll out faster than real time, with persistent state and bounded error over long horizons. Doing this *as a video synthesis problem* requires a generator that produces visually plausible futures.

These two goals pull in different directions. A video generator that maximizes per-frame realism can be totally inconsistent across frames (objects appear, disappear, change identity). An operational world model with rock-solid state can produce visually unimpressive rollouts. The taxonomy below organizes the design space by which goal each approach prioritizes.

## The shared pattern

All world models compute, at each step:

$$\hat{o}_{t+1} = f(o_{\leq t}, a_{\leq t}; s_t)$$

where $s_t$ is some internal state (explicit or implicit) and $f$ is the model. Differences come from (a) what $s_t$ is and how it's maintained, (b) whether $a$ is included and how, (c) what objective is used to train $f$.

## Variants

| Approach | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Diffusion video generators | Sample-by-sample denoising of frame windows | Per-frame realism; no state guarantees | Visual demos, content generation |
| Autoregressive video LMs | Token-by-token frame prediction with transformer | Composable, scales well | Mid-horizon prediction, generation |
| Latent action models (Genie-style) | Learn discrete latent actions from video, generate conditional on them | Action-controllable from passive video | Game-like environments |
| Operational world models — Kairos | Hybrid temporal attention with provable error bound + cross-embodiment curriculum | Long-horizon state, deployment-ready | Embodied agent simulators |
| Active perception (omni-modal agents) | Model decides what to observe rather than predicting full frames | Sub-linear compute in stream length | Long video, omni-modal understanding |
| JEPA-style predictive embeddings | Predict in embedding space, not pixel | Skips pixel realism; planning-friendly | Self-supervised representation learning |

## How to choose

For embodied control and planning, the modern default is the **operational** camp — pick something with persistent state and a long-horizon error bound. Kairos is the current reference point. For pure video synthesis (entertainment, ads), pick a **diffusion video generator** — DreamMachine-class, MovieGen-class. For learning representations rather than running rollouts, **JEPA-style** predictive embeddings are still the right shape.

Active perception is orthogonal: any of the above can be coupled with an active-perception controller that chooses what to attend to, and for long streams this is usually the difference between feasible and infeasible at inference.

## Adjacent but distinct

- **Video diffusion models** for content generation — same architecture lineage as diffusion video generators here but not used in closed-loop control. Belongs in a future `_video-diffusion.md`.
- **Physics simulators** (MuJoCo, Isaac) — deterministic engines with hand-coded dynamics; same role as world models but different construction. Often used to *train* world models.
- **POMDPs in RL** — the planning-side abstraction a world model implements. The model is the world; the POMDP solver is the planner that uses it.

## Sources

- Paper: *Kairos: A Native World Model Stack for Physical AI* — Kairos Team, 2026 — operational world model + provable error bound.
- Paper: *Dreamer / DreamerV3* — Hafner et al., 2020–2023 — early operational world models in latent space.
- Paper: *Genie 2* — Bruce et al., 2024 — latent-action world model from internet video.
- Paper: *I-JEPA / V-JEPA* — LeCun et al. — predictive-embedding world models for representation learning.

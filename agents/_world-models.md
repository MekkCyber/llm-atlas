# World models

*Taxonomy — generative models that predict future observations (and sometimes actions) conditioned on state, action, or intent.*

**TL;DR:** A world model learns dynamics: given the current observation and (usually) an action, predict what comes next. Modern LLM-era world models split into three shapes — pure **video generators** that produce plausible futures, **agent policies** that map observations to actions, and **World-Action Models (WAMs)** that unify both. The choice determines what you can *do* with the model: simulate, act, or both.

**Related taxonomies:** [../multimodal/README.md](../multimodal/README.md)
**Depth files covered here:** in-context WAM (Zero-WAM, see [../multimodal/vla.md](../multimodal/vla.md))

---

## The problem

Sequential decision-making needs *some* way to anticipate consequences. Model-free RL learns the map from state to action directly, wastes samples, and can't plan; classical model-based RL learns dynamics but has struggled to scale beyond low-dimensional state. Modern generative modeling gives us pixel-space world models — but with pixel realism comes the question: is the model there to *simulate*, to *act*, or both?

## The shared pattern

Every variant answers three questions:

- **What does it predict?** Next observation only, next action only, or both jointly.
- **What does it condition on?** Past observations, past actions, task text, or an in-context demo trajectory.
- **How is it verified?** Aesthetic reward, engine checks, task success, or distributional (see [PAWBench](../evaluation/pawbench.md)).

They all share an autoregressive-over-time backbone (transformer / DiT) whose conditioning mix and output head decide the class.

## Variants

| Class | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Video world model | Predict pixel futures from state (+ optional action). | No policy; can't act on its own. | Simulation, data augmentation, offline RL rollouts. |
| Interactive world model | Predict video futures conditioned on user-supplied actions. | Requires action-labeled training data. | Games, sim-to-real, imagination-augmented planning. |
| Pure agent | Observation → action, no future prediction. | No planning signal beyond reward. | Latency-critical control, teleop distillation. |
| WAM (world-action model) | Joint next-frame + next-action head. | Larger, harder to train, novel failure modes (see below). | Closed-loop gameplay, GUI agents, embodied VLAs. |
| In-context WAM | WAM conditioned on a demo trajectory. | Data-hungry pretraining. | Open-ended task generalization, zero-shot policies. |

Failure modes worth naming: **probabilistic misalignment** (single-shot plausibility ≠ distributional fidelity — see [PAWBench](../evaluation/pawbench.md)) and **Low-Frequency Action Source Imprinting** (GameWAM's diagnosis of WAMs over-copying frequent-action prototypes).

## How to choose

- **Just need synthetic rollouts / to fine-tune a policy offline?** Video world model — cheapest, most training data available.
- **Building a game agent or GUI controller?** WAM — the joint head lets the model plan in observation-space rather than delegate.
- **Deploying to many new tasks?** In-context WAM — pay pretraining once, get task swap for free.
- **Evaluating any of the above?** Move beyond FVD / CLIP-score; test distributional alignment per [PAWBench](../evaluation/pawbench.md).

## Adjacent but distinct

- **VLMs / VLAs** — [vla.md](../multimodal/vla.md) covers vision-language-action models. VLAs are the pure-agent class in this taxonomy.
- **Diffusion / flow-matching for images** — see [flow-matching.md](../multimodal/flow-matching.md). Backbone shared, but static image generation isn't a world model.
- **Model-based RL classics** (Dreamer, MuZero) — the ancestral line; different action/observation granularity.

## Sources

- *Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models* — Zhou et al., 2026 — [arXiv:2608.25518](https://arxiv.org/abs/2608.25518)
- *GameWAM: A World Action Model for Video Games* — Guo et al., 2026 — [arXiv:2608.26200](https://arxiv.org/abs/2608.26200)
- *Zero-WAM: In-Context World-Action Modeling from Human Videos* — Zhou et al., 2026 — [arXiv:2608.26103](https://arxiv.org/abs/2608.26103)
- *PAWBench: How Far Are We from Probabilistically Aligned World Modeling?* — Pu et al., 2026 — [arXiv:2608.27345](https://arxiv.org/abs/2608.27345)

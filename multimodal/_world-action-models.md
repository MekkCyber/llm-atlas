# World Action Models (WAMs)

*Taxonomy — predictive-action models that condition an action on a forecast of the future.*

**TL;DR:** WAMs are a family of *embodied predictive-action* models. They forecast a future (rendered video, latent state, or no future at all — just an action-conditioned plan) and use that forecast to derive an action. The design space breaks into four axes — **predictive substrate**, **backbone**, **action coupling**, and **deployment regime** — and the field is converging on methods that *generate less of the future* while preserving what control needs.

**Related taxonomies:** none yet
**Depth files covered here:** *(none yet — see survey as the seed)*

---

## The problem

Embodied agents need to know how their actions will play out. The naive answer — generate a full video of the future and pick the action that leads to the best one — is correct but absurdly expensive and slow. The interesting design choices are about how much of the future to actually generate, in what representation, and how to couple the generative substrate to the action policy.

## The shared pattern

Every WAM has:

- A **predictive substrate** — what is forecast and in what space.
- A **backbone** — usually a video-generation model, a VLM, or a hybrid.
- An **action coupling** — how the forecast is used to produce the action (attention head, decoder, classifier).
- A **deployment regime** — closed-loop control vs. open-loop planning, latency budget.

Variants trade representational richness against compute, memory, latency, and action-label cost.

## Variants

| Family | Predictive substrate | Backbone | Tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Rendered-future WAMs | Pixel-space video of the next step | Repurposed video-gen model | High compute, high data; richest signal | When pixels are the natural state (manipulation, driving) |
| Latent-future WAMs | Latent state forecast in a learned embedding | VAE / latent video model | Cheaper, but representation choices are sensitive | When latency matters more than pixel fidelity |
| Video-gen-free WAMs | No explicit future; action-conditioned chain-of-thought | LM / VLM | Lowest compute, weakest grounding | When the action vocabulary is symbolic / discrete |
| VLA hybrids | Implicit future via co-trained policy head | VLM | Avoids generation entirely; relies on backbone priors | When labeled action data is abundant |

## How to choose

- **Pixel-grounded control with rich dynamics** → rendered-future WAM if compute permits; latent-future WAM otherwise.
- **Latency-bound closed-loop** → video-gen-free WAM or VLA hybrid; the future generation cost is a non-starter.
- **Generalisation across embodiments / tasks** → VLA hybrid with light future grounding tends to transfer best; rendered futures over-specialise.
- The survey's headline pattern: *generate the smallest future representation that still carries what control needs.* If your evaluation metric isn't pixel fidelity, you probably shouldn't be generating pixels.

## Adjacent but distinct

- **General world models** (Sora-style video generation, no action conditioning) — generate the future but aren't coupled to an action policy.
- **Vision-Language-Action (VLA) policies** — action-conditioned but typically *don't generate a future*; the policy is end-to-end.
- **Action-grounded video world models** — video generators conditioned on actions but with no policy head; the action is an input, not an output.

These distinctions are the survey's main pedagogical move.

## Sources

- Survey: *World Action Models: A Survey* — Zhang, Liao, Li et al., National University of Singapore, 2026 — [arXiv:2606.20781](https://arxiv.org/abs/2606.20781).

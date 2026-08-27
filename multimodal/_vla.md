# Vision-Language-Action Models (VLAs)

*Taxonomy — foundation models that unify visual perception, language grounding, and continuous action generation.*

**TL;DR:** VLAs extend the VLM stack to output *actions* — motor commands, navigation trajectories, manipulation primitives — rather than just text. Early VLAs bolted a small action head onto a frozen VLM; the modern default is to train action generation jointly with vision-language understanding (and often world-model prediction), so the shared representation is action-informed from pretraining. GigaBrain-0.7 (2026) is the largest published single-system VLA under this joint-training paradigm.

**Related taxonomies:** —
**Depth files covered here:** [three-system-vla](three-system-vla.md)

---

## The problem

A VLM can look at a scene and describe it. A robot policy can execute an action given a state. Naïvely composing them — VLM as perception, policy as controller — treats vision and control as separate concerns and prevents either from adapting to the other. But a single system that must both understand a scene *and* act in it has representational needs no VLM was trained to satisfy: what matters visually is what matters *for the action*, and vice versa.

Every VLA design chooses (a) how much of the VLM stack to reuse, (b) how action generation is decoded, (c) whether world-model prediction is trained as an auxiliary objective, and (d) whether training is single-stage or cascaded.

## The shared pattern

All VLAs have three functional blocks:

1. **Vision-language understanding** — grounded scene tokens from images + language.
2. **Action generation** — a decoder head producing continuous motor commands, discrete action primitives, or trajectory tokens.
3. **(Optional) World prediction** — forecast future scenes from current scene + action; used as auxiliary loss during training or as a planning module at inference.

The design decisions are (a) *how much these blocks share* (backbone weights? just representation?) and (b) *how they are trained* (cascaded stages vs. joint).

## Variants

| Technique | Approach | Backbone sharing | Training | When it wins |
| --- | --- | --- | --- | --- |
| RT-2 (no depth file yet) | Fine-tune a big VLM (PaLI-X or PaLM-E) with action tokens as a new vocabulary | Full VLM | End-to-end fine-tune on robot data | Small robot data budget; leverages web-scale VLM priors |
| OpenVLA (no depth file yet) | Open-source RT-2-style with Llama-2 + SigLIP; discrete action tokens | Full VLM | Fine-tune on Open X-Embodiment | Reproducibility; community baseline |
| π₀ / π₀.₅ (no depth file yet) | VLM backbone + flow-matching action decoder | Full VLM | Two-stage: pretrain VLM, train action decoder | Continuous high-frequency control; smooth trajectories |
| [three-system-vla](three-system-vla.md) | Understanding + Prediction + Action as three subsystems trained jointly | Shared representation, distinct heads | **One-stage** joint alignment | Multi-embodiment generalization at scale (GigaBrain-0.7) |
| Behavior Transformer (no depth file yet) | Multimodal action decoder over discretized action space | No VLM; pure action model | Behavior cloning | Multi-modal action distributions, no language conditioning needed |

## How to choose

- **Small robot data, no multimodal complications** → OpenVLA or RT-2 style. Cheap, well-understood, community baselines exist.
- **Continuous high-frequency control** → π₀-style with flow-matching decoder.
- **Multi-embodiment generalization at scale** → three-system-vla (GigaBrain-0.7). One-stage alignment couples understanding to action correctness across embodiments.
- **Language-free control** → skip VLA entirely; use a behavior transformer or diffusion policy.

The choices compose: three-system's one-stage alignment can be paired with a flow-matching action decoder if the embodiment demands smooth continuous control.

## Adjacent but distinct

- **VLMs** — perception + language, no action head. VLAs extend them.
- **Behavior cloning / diffusion policies** — action-only models trained on demonstrations. No language grounding; not a VLA.
- **World models** — trained to predict future scenes given actions, but don't themselves produce actions. VLAs may incorporate world-model auxiliary training but produce actions.
- **Embodied LLMs** (e.g. PaLM-E, SayCan) — LLMs that emit *text-level* action instructions consumed by a downstream low-level controller. Not end-to-end VLAs; the LLM never produces motor commands directly.

## Sources

- Paper: *GigaBrain-0.7* — GigaBrain Team, 2026 — [arXiv:2608.15875](https://arxiv.org/abs/2608.15875). Three-system, one-stage alignment.
- Paper: *RT-2* — Google, 2023 — VLM-with-action-tokens paradigm.
- Paper: *OpenVLA* — Kim et al., 2024 — open-source RT-2-style.
- Paper: *π₀ / π₀.₅* — Physical Intelligence, 2024–2025 — flow-matching action decoder.

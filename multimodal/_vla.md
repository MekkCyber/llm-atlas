# Vision-Language-Action Models (VLAs)

*Taxonomy — architectures that map camera + instruction to robot actions.*

**TL;DR:** VLAs bolt an action decoder onto a vision-language backbone so a single model can drive manipulation, navigation, or full-body motion from pixels + natural-language instructions. The dominant recipe is *LLM-centric* (V→L→A: project pixels into an LLM's residual stream, then decode actions from LLM hidden states), which pays LLM-inference tax on every control step. A newer *compact-fusion* line (V+L→A) argues the LLM was optional as a fusion module and gets 30–100× cheaper inference at comparable success.

**Related taxonomies:** *(none yet)* — this bootstraps `multimodal/`.
**Depth files covered here:** *(none yet — depth files will land as the graph grows)*

---

## The problem

Two failures the old computer-vision-plus-policy stack had:
1. **Instruction generalization.** Hand-coded policies can't parse "put the blue mug on the left burner." Language conditioning has to sit inside the perception loop, not on top.
2. **Cross-task transfer.** Task-specific policies don't share representations, so every new task is a fresh dataset. A shared VL encoder is the natural home for cross-task priors.

The constraint every VLA fights: **real-time control**. Manipulation runs at 10–50 Hz; navigation faster. Any policy that runs at 1 Hz is unusable.

## The shared pattern

All VLAs have three parts:

- **Vision encoder** — image(s) → visual tokens or features.
- **Language input** — instruction tokens (sometimes robot-identity or embodiment tags).
- **Action decoder** — emits a continuous action chunk (a short horizon of joint angles / end-effector deltas / discrete skills).

They differ in *how vision and language interact* before the action decoder runs.

## Variants

| Variant | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| LLM-centric V→L→A | Project vision into an LLM's context; decode actions from LLM hidden states (Qwen-VLA-style, OpenVLA) | Rich cross-task priors from a large LM; pays full LLM inference per control step | Cross-task / cross-embodiment generalization, tolerated latency |
| Compact-fusion V+L→A | Independent V and L encoders; lightweight bidirectional fusion; small action decoder — no LLM in the hot loop (TurboVLA) | Small, fast; loses whatever emergent reasoning the LLM contributed | Consumer / edge deployment, high control frequencies |
| Action-chunk DiT decoder | Diffusion-Transformer decoder emits a chunk of continuous actions (Qwen-VLA-style) | Smooth trajectories, strong action distribution modeling; more decoder compute | Manipulation with real motor dynamics |
| Skill-vocabulary decoder | Discrete atomic-skill output translated by an external controller (HumanCLAW's evaluation setup uses this pattern) | Decouples decision from execution; skill set caps expressiveness | Evaluation of decision-making; embodied agents with strong low-level controllers |

## How to choose

**Default for research prototypes:** LLM-centric with a DiT action decoder — the recipe with the strongest transfer numbers today. **Default for deployment:** compact-fusion when the control frequency budget makes LLM inference infeasible; skill-vocabulary when a robust low-level controller already exists (mobile manipulation, humanoid full-body). The compact-fusion camp is new (TurboVLA, 2026) — treat its cross-task-transfer numbers with more caution than LLM-centric baselines until independent replications land.

## Adjacent but distinct

- **VLMs** (vision-language models) — no action head; describe / answer / ground rather than control.
- **Classical visuomotor policies** — end-to-end pixels-to-actions with no language conditioning.
- **World models for control** — learn dynamics, plan through them. VLAs are model-free.

## Sources

- Paper: *TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM* — Xie et al., HUST / Huawei, 2026 — the compact-fusion V+L→A design and its efficiency argument. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Paper: *HumanCLAW: Can Vision-Language Models Act Through a Body?* — Gu et al., Meta / NTU / UW, 2026 — the skill-vocabulary evaluation pattern; decouples decision from execution.
- Paper: *Qwen-VLA: Unifying Vision-Language-Action Modeling* — Qwen team, 2026 — canonical LLM-centric + DiT-action-decoder VLA.

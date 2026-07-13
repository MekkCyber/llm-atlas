# Flash-BoN
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Inference-time scaling for **diffusion models**, done under a wall-clock budget instead of a step budget. Flash-BoN generates many **cheap draft candidates** by combining (a) timestep truncation, (b) layer skipping, and (c) activation proxies as pseudo-rewards, then runs **multi-stage verification** to pick the best draft and refines it at full quality. Under real wall-clock constraints this beats guided-search methods that invest compute per intermediate denoising step. Gains scale with model size (**+8% AUC** on the biggest model tested) and it accelerates diffusion RL post-training convergence by ~**10×**. Introduced by Shirkavand et al. (UMD / Hugging Face), 2026 (arXiv 2607.04461).

**Prereqs:** [../post-training/rejection-sampling.md](./../post-training/rejection-sampling.md)
**Related:** [../post-training/reasoning/mcts.md](./../post-training/reasoning/mcts.md) · [../post-training/reasoning/long-cot-rl.md](./../post-training/reasoning/long-cot-rl.md)

---

## What it is

A recipe for spending extra compute at inference to improve a diffusion model's output. The dominant prior line is *guided search* — run something MCTS-flavored where each denoising step is scored and low-scoring paths are pruned. Flash-BoN's premise: when compute is measured in wall-clock rather than "steps," those intermediate scores cost more than they save, and old-fashioned Best-of-N with **cheap** samples wins.

## How it works

Three moves compose into one recipe.

**Cheap draft generation.** Produce many candidate outputs at a small fraction of the full sampling cost via any combination of:

- **Timestep truncation** — stop denoising well before the full schedule ends. Drafts are lower-fidelity but capture the coarse structure.
- **Layer skipping** — skip a subset of DiT layers on each draft step.
- **Activation proxies** — use intermediate activations as a pseudo-reward that predicts final quality without running the verifier model.

Any single trick isn't new; combining all three is what makes drafts *actually* cheap.

**Multi-stage verification.** Rather than one expensive scoring pass over all drafts, verify in stages:

1. First stage — cheap proxy (activation-based) drops the obviously bad drafts.
2. Second stage — a stronger verifier scores the survivors.
3. Third stage — the top candidate is refined at full quality (continue denoising from where the draft stopped, at full precision, full layer count).

**Composes with existing tricks.** Flash-BoN sits on top of reflection-based prompt optimization and other orthogonal inference-time scalers — the paper reports additive gains when combined.

## Why it matters

- **Reframes "inference-time scaling for diffusion."** The prior narrative — mirroring o1/R1 for LLMs — was that per-step verification would win. Flash-BoN's wall-clock measurement flips the ranking: cheap drafts + good picker beat expensive intermediate scoring once the clock is what matters.
- **Larger models benefit more.** +8% AUC on the biggest scale tested. This is the right direction — the technique doesn't stall as models grow.
- **~10× faster RL post-training.** Diffusion RL loops that use Flash-BoN in the rollout phase converge roughly an order of magnitude faster in wall-clock terms. That's a real deployment gain for anyone training image/video models with RL from RM feedback.
- **Deployment-ready.** No new architecture, no retraining. Sits in front of an existing diffusion pipeline.

## Gotchas & tricks

- **Draft aggressiveness has a floor.** Truncate too aggressively and drafts lose the structure the verifier needs to score them. Layer skipping past a threshold destroys generation quality. The recipe's cheap-drafts window is real but bounded.
- **Activation proxies are noisy.** They correlate with quality but not perfectly. The first-stage cull should keep enough headroom that a genuinely-good-but-proxy-flagged draft can survive to the stronger verifier.
- **Wall-clock, not step count, is what wins.** If someone benchmarks Flash-BoN with a fixed step budget instead of a fixed wall-clock budget, the story looks different — guided search catches up. Make sure the compute axis is clocks.
- **Verifier choice matters.** Flash-BoN's picker is a proxy for downstream reward; if the verifier is misaligned with what you actually care about, you'll efficiently produce many drafts of the wrong thing.

## Sources

- Paper: *Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models* — Shirkavand, Paul, Wen, Huang, Chen, Goldstein, Somepalli (UMD / Hugging Face), 2026 — [arXiv 2607.04461](https://arxiv.org/abs/2607.04461).
- Related: *Rejection sampling for LLMs* — [../post-training/rejection-sampling.md](./../post-training/rejection-sampling.md).

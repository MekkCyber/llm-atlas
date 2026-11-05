# Case Study: Axion 1.0

*An open "agent foundation model" trained with a deliberately minimal three-stage recipe — self-contained reasoning SFT → tool-dependent reasoning SFT → GRPO RL — with no mid-training, no DPO, and no on-policy distillation. The paper ships every artifact needed to reproduce it: data construction code, filtering, verification harnesses, prompts, and training code. 2026.*

**Related concepts:** [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [_rl](../post-training/_rl.md) · [adversarial-problem-synthesis](../post-training/adversarial-problem-synthesis.md) · [verifiable-tool-tasks](../agents/verifiable-tool-tasks.md) · [rollout-routing-replay](../systems/rollout-routing-replay.md) · [deepseek-r1 case study](deepseek-r1.md)

---

## What this is

**Axion 1.0**, released November 2026 as a fully open agent foundation model family. Two main artifacts plus two smaller SFT-only models:

- **Axion 1.0-35B** — 35B total / 3B active MoE, initialized from `Qwen3.6-35B-A3B`.
- **Axion 1.0-122B** — 122B total / 10B active MoE, initialized from `Qwen3.5-122B-A10B`.
- **Axion 1.0-4B** and **Axion 1.0-9B** — SFT-only variants (two stages, no RL).

The paper's position is that **three stages is enough** when the first two are *data-centric* rather than *loss-centric*: Stage 1 synthesizes reasoning problems adversarially matched to the base model's own ability frontier, Stage 2 constructs tool-use tasks whose correctness is checkable by deterministic computation, and Stage 3 is a standard GRPO RL phase. The paper explicitly excludes mid-training, DPO, and on-policy distillation to make a clean baseline argument.

Full release: model weights, training code (Megatron-FSDP + ms-swift for SFT, verl for RL), data construction pipelines, verification harnesses, prompts, and the exact hyperparameters below.

---

## Architecture at a glance

- Mixture-of-experts backbones inherited from Qwen3.5 / Qwen3.6, including shared-expert design and aux-loss-free-style routing (not re-derived — see [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)).
- Uses fully open Qwen-family bases as a starting point; the Axion recipe contributes the training pipeline, not a new base model.
- 35B-A3B: 35B total, 3B active.
- 122B-A10B: 122B total, 10B active.
- Context window **128K** in SFT; RL uses **12,000-token prompt cap** and **49,152-token response cap** per session.
- Training precision: BF16 throughout.

No architectural innovation; the paper's novelty is in the training recipe and data pipelines.

---

## Training recipe

### Stage 1 — Self-contained reasoning SFT

**Goal:** develop strong reasoning without tools.

**Data:** synthesized via **model-aware adversarial problem synthesis** — generate challenging problems from the base model's *own* successful solution strategies and adversarially strengthen them. Difficulty is pinned to the frontier of what the model can almost do. See [adversarial-problem-synthesis](../post-training/adversarial-problem-synthesis.md).

**Scale:** 657M tokens processed on 35B in 1h 2min; 841M tokens on 122B.

**Hyperparameters (shared across SFT stages):**

- Learning rate: $3 \times 10^{-6}$, constant, no warmup.
- Optimizer: Adam with $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 0.1.
- Gradient clipping: 1.0.
- Epochs: 3 per stage.
- Global batch size: 8.
- Precision: BF16.
- Framework: Megatron-FSDP + ms-swift.
- Full-parameter SFT with sample packing; loss supervised only on assistant tokens.

### Stage 2 — Tool-dependent reasoning SFT

**Goal:** extend reasoning to tool-mediated tasks.

**Data:** **deterministic-computation-grounded verifiable tool tasks** — tasks whose ground-truth answers can be computed by executable code against real data sources (databases, APIs, calculators). No learned critic, no human verification. See [verifiable-tool-tasks](../agents/verifiable-tool-tasks.md).

**Scale:** 1h 46min on 35B; 3h 58min on 122B.

Same hyperparameters as Stage 1.

### Stage 3 — RL (GRPO)

**Goal:** strengthen long-horizon reasoning and tool-environment interaction.

**Algorithm:** [GRPO](../post-training/grpo.md) with PPO-style clipped updates, implemented in **verl**.

**Hyperparameters:**

- Learning rate: $2 \times 10^{-6}$, constant, 5 warmup steps.
- Rollouts per question: 8.
- Rollout temperature: 1.0.
- Clip range: 0.2 / 0.28 (lower/upper).
- KL coefficient: 0.001.
- Entropy coefficient: 0.
- Batch: 24 prompts with non-degenerate rewards per update.
- Per-session response cap: 49,152 tokens. Per-session prompt cap: 12,000 tokens.

**Three RL-infrastructure innovations** that make the recipe work at MoE scale over long tool-using trajectories:

- **Asynchronous training with staleness bound.** Rollout engines generate continuously; the trainer updates once 24 prompts with non-degenerate rewards accumulate. Weights sync every two optimizer steps. Trajectories more than **11 parameter versions old are discarded**.
- **Session-level training with shared reward (TITO).** Multi-turn RL rollouts often hit **context condensation** (truncation + re-tokenization at a boundary). Standard re-tokenization breaks rollout↔training equivalence because BPE is boundary-sensitive. Axion's **Token-In-Token-Out** mechanism exchanges token IDs directly across the agent loop (generation, tool output, condensation) and tokenizes only at session start. Each session inherits the rollout's reward and group-normalized advantage; loss is normalized per session. The paper credits QUEST for the session-level framing.
- **Rollout Routing Replay (R3).** vLLM and Megatron can route the same token to different MoE experts under identical weights, breaking rollout↔trainer consistency. R3 records expert selections during rollout and replays them in the trainer's forward pass. Stored as `uint8` tensors (~20 MB per sample). See [rollout-routing-replay](../systems/rollout-routing-replay.md).

---

## Key results

**Table 1 — primary benchmarks (%):**

| Model | HLE | BMB | SEAL-0 | HiPhO | FS Olympiad | FS Research |
|---|---|---|---|---|---|---|
| Axion 1.0-35B | 52.2 | 35.3 | 62.2 | 45.7 | 86.0 | 56.7 |
| Axion 1.0-122B | 59.0 | 41.7 | 58.6 | 46.2 | 87.0 | 66.7 |

HLE scores are under restricted website access; without restriction, HLE = 59.1 (35B) and 64.8 (122B).

**Table 2 — HLE-130 across stages (%):**

| Model | Base | After Stage 1 | After Stage 2 | After Stage 3 |
|---|---|---|---|---|
| 35B | 47.7 | 59.2 | 66.2 | 70.0 |
| 122B | 40.8 | 68.5 | 68.5 | 74.6 |

Each stage adds measurable lift. On the 122B, Stage 2 adds no HLE-130 gain over Stage 1 — tool-use SFT trades a bit of raw reasoning for tool-interaction capability that RL then compounds.

**Table 3 — small SFT-only variants on HLE-130:** 4B scores **53.1**; 9B scores **63.1**.

---

## What Axion is not doing

Explicitly excluded from the recipe:

- **No mid-training phase** — the base model is used as-shipped, no separate mid-training between pretraining and SFT.
- **No DPO** — no preference optimization pass. The paper argues that model-aware adversarial data construction in Stage 1 provides implicit preference signal (hard-correct vs easy-correct).
- **No on-policy distillation** — no self-distillation or teacher-student step. RL does the "sharpen current policy" work directly.
- **No learned reward model** — all rewards are rule-based or executable-verifier-based, following the DeepSeek-R1 / RLVR lineage.

The paper's thesis is that each excluded component can be dropped when the **data construction** is tight enough.

---

## Related in the graph

- [adversarial-problem-synthesis](../post-training/adversarial-problem-synthesis.md) — Stage 1 data trick.
- [verifiable-tool-tasks](../agents/verifiable-tool-tasks.md) — Stage 2 data trick.
- [rollout-routing-replay](../systems/rollout-routing-replay.md) — MoE+RL consistency fix.
- [deepseek-r1 case study](deepseek-r1.md) — contrasting 4-stage recipe that Axion's three stages are explicitly simplifying against.
- [grpo](../post-training/grpo.md) — Axion's Stage 3 algorithm.
- [rlvr](../post-training/rlvr.md) — Axion inherits the verifier-first reward philosophy.
- [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) — inherited from the Qwen3.5/3.6 MoE backbone.

---

## Sources

- Paper: *Axion 1.0: Open Agent Foundation Models, Simplified* — 2026, arXiv 2611.03233.
- Release: full training code, data pipelines, and verification harnesses alongside the paper.
- Framework: verl (RL); Megatron-FSDP + ms-swift (SFT).

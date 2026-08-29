# Harness-Aware Training (HAT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In production, an agent's **harness** — skill IDs, hook code, tool schemas, prompt structure — is updated much more often than model weights. A compact model trained on a fixed harness overfits to those surface strings and breaks when the harness is edited. **HAT** trains harness-invariant tool use by applying **Harness-State Augmentation (HSA)** — task-preserving perturbations to every harness element — across three stages: HSA-SFT (learn tool use from strong-model trajectories under diverse harnesses), General On-Policy Distillation (restore lost generalization), HSA-RL (robustify with GRPO-style RL in perturbed environments). Deployed in Taobao Live's digital avatar service.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [README.md](README.md)

---

## What it is

Fixed-harness SFT teaches a compact model *"call skill_a34b when user asks about price."* When the harness developer renames `skill_a34b` to `skill_price_query`, or adds a new argument to the tool schema, the compact model breaks. HAT reframes this brittleness as an out-of-distribution problem where the OOD dimension is **the harness itself**, and fixes it with data augmentation across that dimension during training.

## How it works

**Harness-State Augmentation (HSA)** — the augmentation surface:

- **Skill identifiers** — rename `skill_a34b` ↔ `skill_price_query`, reshuffle skill index.
- **Skill content** — paraphrase skill descriptions.
- **Tool schemas** — permute argument order, rename argument keys, add unused optional arguments.
- **Prompt structure** — reorder system-prompt sections, swap in equivalent phrasings.
- **Hook functions** — vary the hook code that wraps skills (task-equivalent implementations).

All perturbations are **task-preserving**: the correct output is unchanged.

**Three-stage training:**

1. **HSA-SFT.** Strong-model trajectories are generated across many perturbed harnesses; the compact student SFTs on all of them. It learns *what to do* rather than *what tokens to emit*.
2. **General On-Policy Distillation.** SFT overspecializes; a second stage distills a general LLM's behavior on non-agentic prompts to restore IFEval-style capability that fixed-harness SFT usually costs.
3. **HSA-RL.** GRPO with a verifiable reward on task completion, sampled across perturbed harnesses. Makes the model *robust*, not just *aware*.

## Why it matters

- **Solves a real deployment problem.** Every deployed agent stack updates its harness weekly; the paper's Fixed-Harness SFT baseline drops IFEval by 7.7 pts from base and Harness-Variant QA by 19 pts from HAT.
- **Generalizes beyond avatars.** The HSA recipe is domain-agnostic — anywhere you compile a compact model against a specific tool set, the same augmentation surface applies.
- **Compact models beat frontier on their own harness.** HAT-trained model at 94.8 on Live-Stream QA vs the strongest general LLM at 93.0, at 3.4 s P50 latency on one H20.

## Gotchas & tricks

- **Middle stage is non-optional.** Skipping General On-Policy Distillation loses 7+ IFEval points — the SFT overspecializes even under HSA.
- **Task-preserving perturbations are the ML judgment.** Get one wrong (a "paraphrase" that changes semantics) and you're training the model to give the wrong answer.
- **RL reward has to be verifiable.** For Taobao Live it's product-QA correctness + response-format checks. Domains without a cheap verifier will need to synthesize one.
- **Latency budget is a hidden constraint.** The 3.4 s P50 target influenced the choice of compact base; a bigger model would remove the deployment need.
- **HSA-RL environment must contain the deployed harness distribution.** If production later ships a harness element the training augmentations never covered, HAT provides no guarantee.

## Sources

- Paper: *Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report* — Sun et al. (Taobao / Alibaba), 2026 — arXiv:2608.15763.

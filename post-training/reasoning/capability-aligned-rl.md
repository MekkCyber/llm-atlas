# Capability-Aligned RL (CaRL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward-shaping + hindsight-augmentation recipe that trains long-reasoning LLMs to **refuse** rather than fabricate on questions past their capability. Fixes the "specious reasoning" failure mode where models produce plausible-looking-but-wrong derivations for tasks they can't actually solve, while preserving performance on tasks they can.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [rlvr](../rlvr.md), [_rewards](../_rewards.md)
**Related:** [length-penalty](length-penalty.md), [long2short](long2short.md), [../../safety/refusal-suppression](../../safety/refusal-suppression.md)

---

## What it is

Long-reasoning LLMs display **universal capability overreach**: given a beyond-capability question, they burn thousands of tokens producing plausible-looking derivations that are subtly wrong. The dominant failure mode is **specious reasoning** — outputs that *look* valid on a spot check but contain small errors that escalate with task difficulty.

CaRL (Capability-aligned Reinforcement Learning) is the training-time counter: teach the model to *abort* when the task is out of reach, not just when the answer is obviously unknown. Two components:

1. **Reward shaping** that pays out for refusal on beyond-capability tasks.
2. **Hindsight refusal augmentation** that converts recorded failure trajectories into refusal supervision.

## How it works

**Reward shaping.** In addition to the correctness signal from a verifier (RLVR-style), add:

- **+r_refuse** for a refusal on tasks the model empirically cannot solve at high pass@k.
- **−r_futile** for confident wrong answers on beyond-capability tasks (dominant failure mode being suppressed).
- Correctness reward on within-capability tasks stays intact.

The reward composition explicitly disincentivizes plausible-sounding wrong derivations relative to a clean "I can't solve this".

**Hindsight refusal augmentation.** Recorded failure trajectories are relabeled: the outputs are truncated and their targets swapped to a refusal template. This turns each failure into a refusal-training example, so the SFT/RL mix contains many refusal examples grounded in tasks the model *actually* failed — no manual labeling.

The training loop stays a standard RLVR loop with the shaped reward and augmented data; no new algorithm, just new signal.

## Why it matters

- **Substantial reduction in futile reasoning** across task difficulties, while preserving in-capability performance — i.e. capability-aligned behavior without utility loss.
- Directly diagnoses and measures a real, universal failure of long-reasoning LLMs. Adds "capability overreach" and "specious reasoning" as first-class terms.
- **Bridges reasoning and safety.** Wrong-but-confident derivations are worse than "I don't know" in most deployments; teaching graceful abort is both a cost lever and a trust lever.
- Complementary to length penalties: length penalties limit *amount* of reasoning, CaRL controls *whether to reason at all* given capability.

## Gotchas & tricks

- **Refusal calibration is the whole game.** Reward too high and the model refuses tractable tasks; too low and futile reasoning persists. The paper tunes this against pass@k on a held-out set.
- **Hindsight augmentation depends on trajectory logging.** If you don't retain rollout traces, you can't build the refusal-augmentation set.
- **Interacts with instruction tuning.** A model heavily trained to always attempt an answer will resist refusal signal — introduce CaRL after SFT that includes some refusal patterns.
- **Doesn't distinguish "unknown fact" from "unsolvable derivation".** Both trigger refusal, which is fine for compute savings but limits the interpretability of a refusal.
- **Verifier quality matters.** The pass@k threshold used to label "beyond capability" is a verifier decision; a noisy verifier will train the wrong refusal boundary.

## Sources

- Paper: *Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning* — Xinyan Guan, Jiali Zeng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Fandong Meng (ISCAS / Tencent), 2026 — [arXiv:2607.29211](https://arxiv.org/abs/2607.29211).

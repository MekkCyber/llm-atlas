# Revision–Verification Augmented Training

*Depth — train the underlying revision and verification skills explicitly by decoupling them from successful recovery trajectories.*

**TL;DR:** Test-time scaling via sequential revision has become the dominant deployment paradigm for reasoning LLMs (verifier-driven loops, self-correction, multi-pass solves). But standard post-training optimizes **single-shot** objectives — a mismatch with how the model is used at inference. Multi-turn RL helps but optimizes over flat trajectories and wastes the rich signal in intermediate steps. REVES converts intermediate steps in **successful recovery trajectories** into decoupled **revision prompts** and **verification prompts**, then alternates online data augmentation with policy optimization to train both skills explicitly.

**Prereqs:** [grpo](../grpo.md), [long-cot-rl](long-cot-rl.md)
**Related:** [cot-reward-model](../cot-reward-model.md), [prm](prm.md), [orm](orm.md)

---

## What it is

A two-stage iterative training framework specifically targeting the inference loop most modern reasoning systems use: try → verify → revise → repeat. Decouples that loop into the two atomic skills (revision and verification) and trains each as its own task.

## How it works

**Stage A: trajectory mining.** Run the current policy on a prompt set; identify trajectories that contained at least one wrong-then-corrected step (a *recovery*). These are the high-signal training data.

**Stage B: decoupled prompt construction.** For each recovery trajectory, extract two derived prompts:

- **Verification prompt:** *"Given problem $P$ and candidate answer $a_{\text{wrong}}$, identify whether $a_{\text{wrong}}$ is correct and why / why not."* Target: the model's own actual error-identification reasoning at the moment of recovery.
- **Revision prompt:** *"Given problem $P$, candidate answer $a_{\text{wrong}}$, and error analysis $e$, produce a corrected answer."* Target: the actual correction step in the recovery.

This turns one trajectory into two specialized training examples.

**Stage C: policy optimization.** Standard GRPO / RLVR over the augmented prompt set, with the rewards specialized to each prompt type (correctness of the answer for revision, agreement with ground-truth labels for verification).

**Stage D: iterate.** Train one pass, regenerate trajectories with the updated policy, re-mine recovery traces, repeat. Online augmentation: the data evolves with the policy, so the recovery patterns being trained on stay relevant.

## Why it matters

- **Single-shot objective ≠ inference behavior.** Models are deployed in revision loops; training them as if inference is single-pass leaves performance on the table.
- **Multi-turn RL is wasteful by comparison.** Flat-trajectory optimization treats the intermediate wrong step as a low-reward token; REVES treats it as a *gold-standard verification training example*.
- **Decoupling lets the policy specialize.** Verification and revision are different skills (error-spotting vs error-fixing). Training them as separate tasks lets the policy allocate capacity to each.
- **Composable with verifier-driven inference.** A REVES-trained policy is exactly the policy you want inside a self-correction loop — both heads (verifier, reviser) are explicitly trained.

## Gotchas & tricks

- **You need enough recovery trajectories.** If the policy never recovers from mistakes, there's nothing to mine. Cold-start with multi-turn RL until recoveries are common enough to mine.
- **Verifier–reviser balance.** Over-training the verifier and the policy becomes paralyzed (refuses to commit); over-training the reviser and it becomes overconfident. Iterate with both prompt types in each batch.
- **Recovery quality matters more than quantity.** A few clean recovery traces outperform many noisy ones. Filter on whether the eventual answer matched ground truth.
- **Compatible with PRM-style supervision.** The verification prompt is essentially a binary PRM signal at the step level; if you have a separately-trained PRM, you can use it to weight the verification reward.

## Sources

- Paper: *REVES: REvision and VErification–Augmented Training for Test-Time Scaling* — Liu, Zhou, Zhao, Sharaf, Lin, Biswas, Ghavamzadeh, Wang, Hong, Northwestern / Amazon AGI / Qualcomm AI / Univ. of Minnesota, 2026 — [arXiv:2606.18910](https://arxiv.org/abs/2606.18910).

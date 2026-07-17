# On-Policy Distillation (OPD)
*Depth — distill knowledge from a teacher into a student using rollouts sampled from the **student's own** current policy, with dense token-level supervision.*

**TL;DR:** Classical knowledge distillation (KD) samples training sequences from a teacher (or a fixed dataset) and matches the student's per-token distribution to the teacher's. **On-Policy Distillation (OPD)** instead samples training sequences from the *student's current policy*, then queries the frozen teacher for its per-token distribution over the *student's* trajectories and matches to that. This gives token-level supervision on the exact states the student actually visits — including its own mistakes — which is what SFT / SeqKD / offline KD systematically undershoot. OPD has become the default recipe for recovering generation quality after aggressive compression (pruning, quantization) and for distilling a large RL-trained teacher into a smaller deploy-ready student.

**Prereqs:** [rejection-sampling](rejection-sampling.md)
**Related:** [short-opd](short-opd.md), [_post-training](_post-training.md)

---

## What it is

**On-Policy Distillation** trains a student model to match a teacher's token-level distribution on trajectories that the *student* generates.

Contrast with the alternatives:

| Method | Where do training sequences come from? | Supervision | Student sees its own mistakes? |
|---|---|---|---|
| SFT | Fixed dataset (often human) | Hard labels | No |
| SeqKD (offline KD) | Teacher rollouts | Teacher sequences | No |
| Offline KD | Fixed dataset | Teacher's per-token distribution on that dataset | No |
| **OPD** | **Student's own rollouts** | **Teacher's per-token distribution on those rollouts** | **Yes** |

The teacher is frozen; the student rolls out; the teacher is invoked as a "scorer" over the student's tokens to produce the training signal.

## How it works

For each training step:

1. **Sample a prompt** from the training distribution.
2. **Roll out the student** to produce a sequence `y = (y_1, ..., y_T)`.
3. **Query the teacher** for its per-token distribution `p_T(y_t | y_{<t}, prompt)` at every position along the student's trajectory.
4. **Compute the KL** (or forward cross-entropy) between the student's own next-token distribution `p_S` and the teacher's `p_T` at each position: `L = Σ_t KL(p_T || p_S)` or `Σ_t -log p_S(y_t) · p_T(y_t)` etc.
5. **Backprop** through the student.

The key property: gradients flow at every position of a sequence the student actually generated. Contrast SFT/SeqKD, which give the student trajectories from *another* distribution, or offline KD, which uses fixed data. OPD keeps the training states aligned with deployment states.

Because rollouts are on-policy, OPD has the same "distribution drift over training" character as RL — states visited early differ from states visited late. That's the point: the student is always being taught on the states it currently visits.

## Why it matters

- **Fixes silent generation collapse after compression.** After structured pruning or aggressive quantization, greedy generation often produces degenerate outputs even though pass@k under sampling stays useful — the useful tokens are still there but demoted. OPD, with dense token-level teacher supervision on the compressed model's own rollouts, recovers greedy quality where SFT and SeqKD stall (ShortOPD, 2026).
- **Standard endgame for train-large-distill-small pipelines.** Recipes like OvisOCR2 train a large RL-refined teacher (e.g., 4B) then distill it back to a deployable student (0.8B) via OPD — better than offline KD because the student learns to handle its own generation quirks.
- **Complements RL cleanly.** OPD's on-policy structure mirrors PPO/GRPO's, so it composes with RL training loops. In practice teams alternate or interleave OPD and RL steps.
- **Directly targets deployment behavior.** SFT can look great in evals and fall over in generation; OPD trains what actually gets used.

## Gotchas & tricks

- **Teacher call cost dominates.** Every training token requires a teacher forward pass. Batch aggressively, cache prefixes, use tensor-parallel teacher hosting; consider hosting the teacher on a separate inference cluster.
- **Long rollouts waste compute on degenerate tails.** In the recovery regime, students often produce repetitive suffixes. Fixed-horizon OPD spends most of the budget there. **[ShortOPD](short-opd.md)** targets exactly this: detect teacher-confirmed repetitive suffixes and shorten each rollout to its effective prefix, then grow the horizon adaptively.
- **KL direction matters.** Forward KL (`KL(p_T || p_S)`) is mode-covering; reverse KL is mode-seeking. Forward is the default for OPD (matches teacher).
- **Temperature during rollouts.** Sampling temperature during rollouts controls the state coverage. Too low and OPD sees a narrow slice of the distribution; too high and it wastes compute on unrealistic states.
- **Teacher and student vocabularies must match.** OPD supervision is per-token; different tokenizers require re-tokenization tricks or teacher retraining.
- **Watch for teacher-leakage into student.** OPD nudges the student toward the teacher's distribution — including the teacher's failure modes. Combine with outcome-verified filtering when the teacher is not trusted at all positions.

## Sources

- Paper: *ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation* — Yuan, Lin, Lu, Han, Sun, Li, Xu, Li, Zhao, 2026 — [arXiv 2607.13124](https://arxiv.org/abs/2607.13124). ByteDance + CAS. Introduces the short-to-long OPD schedule and validates OPD as the primary recovery lever after structured pruning.
- Paper: *OvisOCR2 Technical Report* — Lu et al., 2026 — [arXiv 2607.13639](https://arxiv.org/abs/2607.13639). Uses OPD to distill a 4B RL-trained teacher into a 0.8B deployable student.
- Related: *Sequence-Level Knowledge Distillation* — Kim & Rush, 2016 — arXiv 1606.07947. The classical SeqKD baseline OPD is contrasted against.

# Difficulty-Aware Length Control
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward-shaping mechanism for reasoning RL that gives each training instance its **own** target reasoning length, computed offline from reference statistics (accuracy and token usage per problem). Easy problems get low target length (direct answer, "NoThink"); hard problems get long-CoT budget. Combined with a verifier reward and batch-standardized advantages, this yields critic-free RL that trades off accuracy and tokens per-instance. Introduced as IDAC (Instance-level Difficulty-Aware Control) inside the When2Think framework.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [length-penalty.md](./length-penalty.md)
**Related:** [long2short.md](./long2short.md) · [grpo.md](../grpo.md) · [rlvr.md](../rlvr.md)

---

## What it is

Reasoning models trained with a uniform length penalty end up **overthinking easy problems and underthinking hard ones** — the penalty applies the same pressure on every instance regardless of difficulty. Existing fixes either use a hard router (which fails on borderline problems) or a global length target (which trades accuracy for tokens uniformly).

Difficulty-aware length control replaces the uniform target with **per-instance targets** derived from reference statistics. Each problem has a measured expected token usage and accuracy under a baseline model; the reward shaping uses those numbers to set the length target locally.

## How it works

**Offline step.** Run a reference model on the training set. For each problem $q_i$, record baseline accuracy $\alpha_i$ and mean token usage $\ell_i$.

**Reward.** For a rollout $o$ on problem $q_i$ with verifier reward $r_i \in \{0, 1\}$ and length $|o|$, the shaped reward penalizes lengths that overshoot the per-instance target proportional to how easy the problem is:

- Easy $q_i$ (high $\alpha_i$, low $\ell_i$): target is short. Penalty on extra tokens is large; NoThink direct answers are encouraged.
- Hard $q_i$ (low $\alpha_i$, high $\ell_i$): target is long. Penalty on extra tokens is small; extended reasoning is preserved.

**Advantage.** Standardize across the batch (mean 0, std 1) — this gives the GRPO-style critic-free advantage without a learned reward model or online reference queries.

**Training.** Standard GRPO/RLVR loop with the shaped reward and batch-standardized advantages.

## Why it matters

- **Kills the efficiency tax.** Uniform length penalties reduce compute on easy instances at the cost of accuracy on hard ones. Per-instance targets don't.
- **No routing model.** Hybrid "System 1 vs System 2" routing usually needs a difficulty classifier. IDAC folds the same decision into the RL reward, so the *model itself* learns when to skip the CoT.
- **Offline reference statistics scale.** The reference pass runs once per dataset, is cheap, and doesn't need to be re-done as the model trains. Contrast with online reference-model queries per step.

## Gotchas & tricks

- **Reference model choice matters.** If $\alpha_i$ and $\ell_i$ come from a model much weaker or stronger than the student, the targets skew. Best practice: reference from a nearby checkpoint of the same family.
- **The mechanism is *not* a router.** It shapes rewards; the model still decides to think or not on each generation. Failure mode is a model that ignores the reward shaping when the KL to the reference is strong — tune the KL coefficient down as targets bite.
- **Batch composition affects standardization.** If a batch is mostly easy problems, hard-problem advantages are diluted. Balance difficulty within batches when possible.
- **Verifier quality is a hard floor.** With a leaky verifier the model can reward-hack toward "think a lot on any problem the verifier rejects." Verifier hardening is a prerequisite for length control.

## Sources

- Paper: *When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models* — Shim, Kim, Kim, Bak, SKKU + Microsoft, 2026 — arXiv:2609.19671. AIME24: +10.0 pp Pass@3 with -27.9% tokens vs. base; AIME25: 40.0% Pass@3.

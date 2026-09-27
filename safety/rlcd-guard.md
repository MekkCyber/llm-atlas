# RLCD Guard — RL for Calibrated Decisions
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small guard/classifier model trained by **Reinforcement Learning for Calibrated Decisions (RLCD)** to answer *many* typed yes/no questions about a single input in one forward pass, with **calibrated probabilities**. Replaces the "one generative-judge call per criterion" pattern for alignment-failure detection at a fraction of the cost, while beating fixed-label classifiers like Llama Guard.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [README.md](README.md), [../post-training/dpo.md](../post-training/dpo.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Alignment-failure detection has two dominant designs:

- **Generative judges** — an LLM emits a yes/no (or a rationale) per criterion, taking a full decoding pass per question. Accurate but expensive at eval scale.
- **Fixed-label classifiers** (Llama Guard) — a small model reads token probabilities and emits *one label* per call from a pre-set taxonomy.

RLCD Guard (nicknamed **Jev**) sits between them: a single model that, given an input and a schema of typed questions, outputs a calibrated probability for **each** question in one call. It reads like a multi-head classifier but is trained end-to-end with RL against a **calibration loss** rather than a preference-pair loss.

## How it works

Training pipeline:

1. **Question schema.** A structured list of typed questions (deception, refusal-avoidance, tool-misuse, self-exfiltration, dual-use content, …), each with a canonical name and a yes/no ground-truth resolution.
2. **Data.** Inputs (prompts + candidate model responses) with human- or verified-LLM-labeled yes/no per question.
3. **Model.** A small (7B-class) LLM with a question-conditioned output head that emits a probability $p_q(x)$ for each question $q$ against input $x$.
4. **RL objective.** The reward is a **calibration score** — typically negative Brier loss or a proper scoring rule — computed against the ground-truth labels:
   $$r(x) = -\sum_q (p_q(x) - y_q)^2$$
   Policy is updated with GRPO-style group-relative advantages: sample $G$ probability vectors per input, score each with the Brier reward, subtract the group mean, apply PPO clip with a KL penalty to a reference model.

The RL framing matters: a simple cross-entropy classifier would optimise likelihood but not calibration; RLCD's reward is the calibration score directly.

## Why it matters

- **Cost.** One forward pass answers dozens of criteria; asymptotically classifier-cost per criterion, not decoded-token cost.
- **Calibrated probabilities.** Downstream systems (threshold-then-block, threshold-then-escalate) actually need probabilities, not labels. RLCD's optimiser aligns with that need.
- **Beats fixed-label classifiers.** Llama Guard emits one label per call; Jev answers many. On safety-evaluation suites, that translates into strictly better AUC and better throughput.
- **Scales to new question types cheaply.** Adding a new criterion is a schema entry plus labels; no new head, no re-architecting.

## Gotchas & tricks

- **Labels must be actually right.** Calibration training amplifies label noise: if 5% of labels are wrong, the model learns a 5%-wrong prior. Use a verified-LLM cross-check on the training labels.
- **Question schema is a taxonomy.** Two questions that mean the same thing under different phrasing corrupt calibration; deduplicate.
- **Reference model choice matters.** KL to a base LLM keeps the guard's language sensible; without it, RL collapses toward degenerate short-circuits.
- **Not a jailbreak detector by itself** — RLCD scores whatever schema you feed it, so it can flag alignment failures given labeled data, but attack novelty still requires red-teaming.
- **Auditable outputs.** Because each answer is a per-question probability, the guard's decisions are trivially audited — a strong point for safety review.

## Sources

- Paper: *Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures* — Guo, Liu, Deng, et al., UNSW / NTU / Griffith, 2026, [arXiv 2609.29429](https://arxiv.org/abs/2609.29429) — introduces RLCD and the multi-question guard Jev.
- Related: Meta's *Llama Guard* — Inan et al., 2023 — the fixed-label classifier baseline.

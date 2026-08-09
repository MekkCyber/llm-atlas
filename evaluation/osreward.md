# OSReward

*Depth — a standardized benchmark for VLM-as-judge on computer-use agent (CUA) trajectories, plus an open reward-model release trained on its data.*

**TL;DR:** OSReward (2026) is the first realistic, human-annotated benchmark that grades VLM judges on the actual signal CUAs need: given a full agent trajectory across screenshots and actions, did the agent complete the user's instruction? Findings: even frontier VLM judges share a systematic leniency bias (call failed runs successes); reliable judges are too expensive for RL scale. Companion releases include OSReward-Hard (challenge subset), OSReward-Multi (fine-grained scoring), OS-Shepherd-100K (open training corpus), and OS-Shepherd 9B/35B — open judges that match commercial ones at 30–60× lower cost.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../post-training/vlm-as-judge.md](../post-training/vlm-as-judge.md), [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

Three connected artifacts:

| Artifact | Purpose |
| --- | --- |
| **OSReward** | Main benchmark: human-labeled CUA trajectories across diverse platforms and backbones; a judge's score is its accuracy on the ground-truth verdicts |
| **OSReward-Hard** | Subset concentrating genuinely difficult cases (near-misses, partial completions, subtle failures) |
| **OSReward-Multi** | Fine-grained rubric: efficiency and alignment scoring, not just success / fail |
| **OS-Shepherd-100K** | Open training corpus of reasoning-annotated trajectory judgments |
| **OS-Shepherd 9B / 35B** | Open VLM reward models trained on OS-Shepherd-100K |

The benchmark exists because the field was quietly using frontier VLM APIs as judges without measuring how good those judges actually are.

---

## How it works

**Trajectory sourcing.** Diverse CUA backbones run human-verified instructions on multiple platforms. Each trajectory: `(instruction, [screenshots, actions, DOM diffs])`.

**Annotation.** Multi-stage human labeling produces a ground-truth verdict per trajectory. The paper reports iterative disagreement resolution rather than a single-annotator pass.

**Judge evaluation.** A VLM-as-judge candidate consumes the trajectory + instruction and emits a verdict. Score = agreement rate with the human ground truth on the full benchmark, on OSReward-Hard, and on OSReward-Multi's fine-grained axes.

**Open reward model.** OS-Shepherd is trained on OS-Shepherd-100K — trajectories paired with reasoning-augmented judgments — targeting the same verdict interface as the frontier judges it replaces. Cost/quality is reported per API dollar.

## Why it matters

- **Names a systematic failure of the current field default.** Leniency bias is not a per-model quirk — it's shared across frontier VLM judges. Any RL run using such a judge as reward is at risk of the policy learning to *look* successful.
- **Makes CUA RL affordable.** OS-Shepherd's 30–60× cost reduction is the difference between "we can afford a small CUA RL study" and "only frontier labs can."
- **Standardization pressure.** Before OSReward, CUA papers reported success under whatever judge they chose. A shared benchmark makes cross-paper comparisons real.

## Gotchas & tricks

- **Not a substitute for rule-based checks when they exist.** For tasks with a machine-checkable final state (URL match, DOM equality, file presence), rule verifiers are still cheaper and unhackable. Use VLM judges only where rules can't.
- **Benchmark ↔ RL-reward gap.** A judge that scores well on OSReward is not automatically a great RL reward — it may be well-calibrated but easy to Goodhart during optimization. Include KL to a reference judge and periodic frontier-judge audits.
- **Fine-grained OSReward-Multi is the more discriminating axis.** Success/fail is easier than efficiency/alignment; comparing judges on Multi surfaces differences that Overall accuracy hides.
- **Trajectory-level judging beats final-screenshot judging.** Judges that see only the final screenshot are cheaper and *more* leniently biased. The full-trajectory pipeline is worth the cost.

## Sources

- Paper: *OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models* — Sun et al., HKU / OS-Copilot, 2026 — [arXiv:2607.28609](https://arxiv.org/abs/2607.28609). Companion release at os-copilot.github.io/OSReward-Home.

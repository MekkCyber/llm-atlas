# Self-Improvement Ladder for Reasoning RL

*Taxonomy — how reasoning RL scales as human supervision recedes from the loop.*

**TL;DR:** As reasoning RL scales past the tasks any single human can verify, the remaining question is *which parts of the learning loop are still under human control*. A five-level ladder (L0–L4) from Yang et al. (2026) organizes the space along two coupled axes: the **reward axis** (per-instance human judgments → reusable verifiers → verifier-free rewards) and the **experience axis** (curated tasks → self-generated curricula → autonomous co-evolution). Each rung has characteristic failure modes; evaluation moves from policy scores alone to a triad of **capability × feedback fidelity × experience quality**.

**Related taxonomies:** [_rl.md](../_rl.md) · [_rewards.md](../_rewards.md) · [_post-training.md](../_post-training.md)
**Depth files covered here:** [long-cot-rl.md](long-cot-rl.md) · [prm.md](prm.md) · [orm.md](orm.md) · [mcts.md](mcts.md) · [../rlvr.md](../rlvr.md) · [../rl-prompt-curation.md](../rl-prompt-curation.md) · [../rejection-sampling.md](../rejection-sampling.md) · [../rubric-as-reward.md](../rubric-as-reward.md)

---

## The problem

Reasoning RL under RLVR ([../rlvr.md](../rlvr.md)) worked because math and code come with cheap ground-truth verifiers. Extending to open-ended and agentic tasks breaks this: reliable rewards are harder to obtain, and direct human-in-the-loop supervision cannot keep pace with the scale and complexity of model-generated experience. So training either stalls at "what humans can verify" — or continues, but on rewards and experience that humans authored less and less.

## The shared pattern

Every rung on the ladder pairs a *reward source* with an *experience source*, and each rung buys more autonomy at the cost of new failure modes. Advances at a given rung tend to fix one axis while relying on the previous rung's discipline on the other.

## Variants

| Rung | Reward axis | Experience axis | Where it lives today | Characteristic risk |
| --- | --- | --- | --- | --- |
| L0 | Per-instance human labels | Human-curated prompts | RLHF-era preference RM training | Labeler noise, cost |
| L1 | Reusable verifier (math/code test) | Curated benchmark prompts | RLVR ([../rlvr.md](../rlvr.md)), long-CoT RL ([long-cot-rl.md](long-cot-rl.md)) | Verifier gaming inside the answer set |
| L2 | Learned RM / rubric ([../rubric-as-reward.md](../rubric-as-reward.md)) / PRM ([prm.md](prm.md)) | Semi-curated + rejection-sampling ([../rejection-sampling.md](../rejection-sampling.md)) | Frontier reasoning fine-tuning today | Reward-model drift, feedback poisoning |
| L3 | Verifier-free / self-consistency / model-as-judge | Self-generated curricula, prompt evolution ([../rl-prompt-curation.md](../rl-prompt-curation.md)) | Emerging: DIEM-style dynamic curricula, agent-generated tasks | Curriculum collapse, sycophantic judges |
| L4 | Autonomous co-evolving reward + environment | Constructed environments; agent-agent interaction | Speculative | Reward hacking against a moving target, environment errors compound |

Rungs are cumulative: an L3 pipeline still uses L1-style verifiers where they exist and L0-style human oversight for evaluation, just not as the primary training signal.

## How to choose

- **You have a verifier that catches all correct answers?** Stay at L1. RLVR is the cheapest supervision-per-signal ratio and the failure modes are the best-understood.
- **Tasks are open-ended but you have a curated corpus of good outputs?** Move to L2 with rubric-as-reward ([../rubric-as-reward.md](../rubric-as-reward.md)) or PRM ([prm.md](prm.md)). Keep human eval as spot-check.
- **Data is running out and you're bottlenecked on task variety?** L3 experience-side moves (self-generated tasks, DIEM-style dynamic mining) — but layer them on top of L1/L2 rewards, not on a self-judged reward, or you'll spiral.
- **Both rewards and experience are self-generated?** You are at L4 and the paper's central warning applies: rewards can co-evolve with the policy toward local optima that no human ever authored. Independent eval is not optional.

## Adjacent but distinct

- **Preference learning ([../dpo.md](../dpo.md))** is an L0/L1 alternative to RLHF rather than a rung of the ladder; it's about how you use the labels, not where they come from.
- **On-policy distillation ([../on-policy-distillation.md](../on-policy-distillation.md))** looks like moving up the reward axis (teacher-free variants exist) but is not really about supervision reduction — the objective is different.
- **MCTS-based search ([mcts.md](mcts.md))** interacts with the ladder by amplifying whatever reward you have; it doesn't itself change the rung.

## Sources

- Paper: *Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence* — Yang et al. — HKUST / Tencent / HKU / CUHK / …, 2026 — arxiv.org/abs/2608.31075. Origin of the L0–L4 ladder and the capability × fidelity × experience-quality triad.

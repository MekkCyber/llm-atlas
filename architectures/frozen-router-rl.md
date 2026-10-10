# Frozen MoE Router during RL
*Depth — a stability trick for MoE post-training that freezes the gating network while policy weights update.*

**TL;DR:** During RL post-training of an MoE model, the **gating / router network** (which experts each token goes to) is held fixed while the rest of the model trains. In standard MoE RL, the router drifts under policy-gradient updates because its gradients are small, sparse, and interact badly with the exploration/exploitation balance of RL — causing load-balance degradation and expert collapse. Freezing the router sidesteps this; the policy specialises *within* each expert instead of reshuffling which expert sees what. Used in MiMo-V2.6 (2026) and expected to generalise to other MoE RL pipelines.

**Prereqs:** [_moe.md](_moe.md), [../post-training/_rl.md](../post-training/_rl.md), [load-balancing-loss.md](load-balancing-loss.md)
**Related:** [aux-loss-free-balancing.md](aux-loss-free-balancing.md) · [deepseek-moe.md](deepseek-moe.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

A one-line intervention in MoE post-training: set `router.requires_grad = False` for the duration of RL. All other weights — attention, expert FFNs, embeddings, norms, biases — update as normal under the RL objective (GRPO, PPO, DPO, etc.).

Not a replacement for load-balancing losses used *during pretraining*; those still shape the router that is then frozen. It's a post-training-only choice.

## How it works

Setting: an MoE model with gating $g(h) \in \Delta^E$ over $E$ experts, pretrained with auxiliary load-balancing. RL post-training then applies a policy-gradient-style loss over sampled completions.

Without freezing, three failure modes accumulate:

- **Load-balance drift.** Router gradients are sparse (only chosen experts receive gradient), so small biases amplify over many RL steps. Some experts receive most tokens; others starve.
- **Expert collapse.** Starved experts' FFN weights don't update, becoming less useful, which further reduces router preference for them.
- **Exploration coupling.** The router's drift interacts with the policy's exploration — some tokens change expert mid-training, so the policy's effective capacity moves non-monotonically.

Freezing the router makes the FFN pool a fixed *compositional* resource: every token's expert set is pinned by the pretrained router, so RL only reshapes *what each expert knows how to produce*, not *which experts contribute to each token*. Load balance is preserved for free.

## Why it matters

- **Portable.** No reward design, no new loss term — just a flag.
- **Avoids a known failure mode.** MoE RL stability issues (load degradation, mysterious benchmark drops in later RL stages) have been reported anecdotally; freezing is a cheap guard.
- **Preserves the pretrained routing policy.** The router was shaped by load-balance-aware pretraining; keeping it means the post-training inherits that balance.
- **MiMo-V2.6's named stability lever.** Reported as one of two explicit stability tricks (the other being the reward-hacking defense).

## Gotchas & tricks

- Freezing the router means RL can't *re-specialise* experts. If a task domain is wildly different from pretraining, the fixed routing may misroute tokens; mix in a small learning rate on the router instead of hard-freezing in that case.
- Still include the pretraining-era aux-loss-free balancing bias — it's part of the "router" in some implementations; freeze consistently.
- Works best when experts were already load-balanced at the end of pretraining. If pretraining left imbalance, freezing locks it in.
- Doesn't help if the primary RL stability issue is reward hacking or KL drift — those need their own fixes.

## Sources

- Paper: *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement* — Xiaomi LLM-Core Team, 2026 — [arXiv:2610.11959](https://arxiv.org/abs/2610.11959). Names "freezing the MoE router during RL" as one of the explicit stability tricks.

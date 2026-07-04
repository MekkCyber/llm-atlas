# Task-Agnostic Pretraining for VLAs
*Depth — self-supervised motor-prior pretraining for Vision-Language-Action models via inverse dynamics.*

**TL;DR:** Vision-Language-Action (VLA) models are bottlenecked by expert demonstration triplets — {vision, language, action} data is scarce, expensive, and slow to collect. **TAP** (Task-Agnostic Pretraining) splits the objective in two: first learn a **motor prior** from cheap unlabeled interaction via self-supervised inverse dynamics (predict action from before/after observations), then ground the motor prior in language with a small labeled corpus. Matches VLAs trained on 1M+ expert trajectories using orders of magnitude less labeled data; +10 pp over behavior cloning on SIMPLER; 25 % real-robot success under camera perturbations where internet-scale baselines fail completely.

**Prereqs:** [../pre-training/README.md](../pre-training/README.md)
**Related:** (no other VLA depth pages yet in the graph)

---

## What it is

A VLA model maps (visual observation, language instruction) → action. Training data are triplets — a demonstration where a human wrote the instruction, controlled the robot, and the video and actions were recorded. Every triplet is expensive.

The paper's central observation: this bundles two very different learning problems together:
- **Physical competence.** Given the current observation, what actions are physically reasonable? This should be learnable from *any* interaction data, expert or not.
- **Semantic alignment.** Given a language instruction, which physically-reasonable action is the *right one*? This needs labels.

Task-Agnostic Pretraining decouples them: pretrain physical competence on cheap unlabeled interaction; fine-tune semantic alignment on scarce expert data.

## How it works

**Stage 1 — Task-agnostic pretraining (self-supervised).** Collect unlabeled interaction data: sequences of $(o_t, a_t, o_{t+1})$ where $o_t$ is the observation and $a_t$ is *any* action (random, teleop, prior policy). Train an **inverse dynamics** head: given $(o_t, o_{t+1})$, predict $a_t$. The learned representation captures "what motor commands cause what observation changes" — a transferable physical prior — with no language, no reward, no expert demonstrations.

**Stage 2 — Language grounding (supervised).** With the motor prior frozen (or lightly adapted), fine-tune a language head on a small expert-triplet dataset. Because the physical prior already knows *how to move*, the fine-tune only needs to learn *when a given instruction wants which move*.

## Why it matters

- **Scarcity of expert triplets is the real bottleneck.** Unlabeled interaction is abundant (any tele-operated demo, any prior-policy rollout, any simulator run). Turning that into transferable motor priors closes the data gap.
- **Matches 1M+ expert trajectories with orders less labeled data.** On SIMPLER benchmark, TAP hits the same success rate as VLAs trained on ≥1M expert triplets — using dramatically fewer labeled examples.
- **Robust to real-world perturbations.** On real-world WidowX robots under camera-position perturbations, TAP maintains **25 %** success; internet-scale baselines fail completely. The motor prior generalizes past the visual distribution shift that language-only pretraining doesn't cover.
- **Composable with any VLA backbone.** The pretraining head can be attached to any vision-language backbone; only the inverse-dynamics head is task-agnostic-pretraining-specific.

## Gotchas & tricks

- **Inverse dynamics only works if actions are recoverable from observation deltas.** For high-dimensional or highly-actuated systems (many degrees of freedom, mostly redundant), $(o_t, o_{t+1})$ underdetermines $a_t$; the head learns a smoothed / averaged action.
- **Interaction data quality matters.** Random-action interaction is broadly informative but leaves state-space corners uncovered; mixing in prior-policy rollouts is helpful.
- **Language head can undo the prior.** If Stage 2 fine-tunes too aggressively, the motor prior is overwritten by task-specific quirks. Freeze or use small learning rates on the prior.
- **VLA is a superset of imitation learning.** TAP's idea (self-supervised motor pretraining + language grounding) applies beyond VLAs — any pipeline that maps language to control should benefit.

## Sources

- Paper: *Learning to Move Before Learning to Do: Task-Agnostic Pretraining for VLAs* — Shi et al., 2026 — [arXiv:2607.02466](https://arxiv.org/abs/2607.02466).
- Related: *RT-2 / OpenVLA* — end-to-end VLA baselines this contrasts with.
- Related: *Inverse dynamics models* — Christiano et al., 2016 (original IDM in embodied RL) — the self-supervised pretraining objective.

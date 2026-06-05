# Self-Distilled Policy Gradient (SDPG)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** SDPG augments GRPO-style RLVR with an auxiliary **on-policy self-distillation loss**: the same model, conditioned on privileged context, acts as a teacher and supervises the no-context student via a full-vocabulary reverse KL. The dense token-level signal complements the sparse verifier reward and stabilizes training. Introduced by Liu et al. (2026).

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [_rl.md](_rl.md)
**Related:** [_rewards.md](_rewards.md), [ppo.md](ppo.md)

---

## What it is

[RLVR](rlvr.md) with [GRPO](grpo.md) is the modern default for verifiable-reward post-training, but the reward signal is *terminal and sparse* — one scalar per response, zero on most tokens. Many rollouts give no informative gradient. SDPG injects an extra dense supervision channel by treating the policy itself as a teacher: a version of the model that sees privileged context (the gold answer, or a hint) supervises the production student that does not.

This is a special case of *on-policy self-distillation* but plugged into the RL update, not used as a standalone SFT stage.

## How it works

For a prompt $q$ with privileged context $c$ (e.g. the verified answer) and a student rollout $o$ from $\pi_\theta(\cdot \mid q)$:

1. **Verifier reward.** Compute $r_i$ for each of $G$ rollouts $o_i$ via the rule-based verifier.
2. **Group-relative advantage** with std normalization — the standard GRPO advantage with z-score.
3. **Teacher logits.** Run a forward pass of the *same* model on the same rollout conditioned on $q, c$ (privileged). This gives per-token next-token distributions $\pi_\theta(\cdot \mid q, c, o_{<t})$ — the teacher.
4. **Reverse KL self-distillation loss.** Apply a full-vocabulary reverse KL from student to teacher at every token of the rollout:
   $$L_{\text{SD}} = \sum_t \mathrm{KL}( \pi_\theta(\cdot \mid q, o_{<t}) \,\|\, \pi_\theta(\cdot \mid q, c, o_{<t}) )$$
5. **Combined objective.** $J_{\text{SDPG}} = J_{\text{GRPO}} - \alpha L_{\text{SD}} - \beta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$.

All three terms are exact, on-policy (the teacher uses the *current* model, just with extra context), and require only one extra forward pass per rollout.

## Why it matters

- **Dense signal under sparse reward.** Every token gets a usable gradient from the SD term, even on responses where the verifier scores zero.
- **No extra model.** Unlike PPO's value network or a separate distillation teacher, the SDPG teacher *is* the policy. Cost is one extra forward pass, no separate optimizer state.
- **Stability.** The authors report better training stability than vanilla RLVR baselines, particularly early in training when reward is rare.

It is plug-compatible with any GRPO-style RL pipeline: swap in the auxiliary loss and provide the privileged-context construction (often trivial in RLVR settings where the gold answer is available at train time).

## Gotchas & tricks

- **Choice of privileged context matters.** Showing the gold answer is the most informative but may cause the teacher to be *too* good (large KL, unstable updates). A partial hint is often better.
- **Direction of KL.** Reverse KL ($\pi_\theta \,\|\, \pi_{\text{teacher}}$) is *mode-seeking* — the student picks the highest-probability mode of the teacher. Forward KL would be mode-covering and tends to blur multimodal teachers; reverse KL is the right default for crisp distillation.
- **Loss-weight schedule.** Down-weighting $\alpha$ as training progresses helps — early SDPG signal is most valuable when the verifier is sparse; later, full reliance on the verifier matters.
- **Compatible with reference-policy KL.** The $\beta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ anchor is independent and should be kept.

## Sources

- Paper: *Self-Distilled Policy Gradient* — Liu, Zhang, Zhang, Gu, 2026 — [arXiv:2606.04036](https://arxiv.org/abs/2606.04036).
- Code: github.com/lauyikfung/SDPG.
- Related: GRPO (Shao et al., 2024); on-policy distillation literature (Agarwal et al., 2023).

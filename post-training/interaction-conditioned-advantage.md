# Interaction-conditioned advantage (ARC)
*Depth — conditioning group-relative advantage on interaction type so open-ended rollouts are compared fairly.*

**TL;DR:** GRPO computes advantages by comparing rollouts inside a *group* that share a prompt. In open-ended agent interaction a rollout can legitimately answer, ask for clarification, request a progress update, or confirm before acting — different interaction types that shouldn't compete on the same reward scale. ARC (Advantage Regularization via Conditioning) partitions the group by interaction type and normalizes advantages **within** each type.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [ppo](ppo.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

An extension to group-based RL (GRPO family) for open-ended agent training. Instead of one shared advantage baseline per prompt group, each rollout is tagged with the *type* of interaction it took (answer / clarify / progress-update / confirm / …) and the advantage is normalized against peers of the same type.

## How it works

- **Tag each rollout** with its visible interaction category (from the first agent-visible action).
- **Partition the group** into sub-groups per interaction type.
- **Normalize advantage inside each sub-group** — mean-subtract, optionally variance-scale — using only rollouts of the same type as the baseline.
- **Reunify** the normalized advantages back into the standard GRPO loss.
- Separately, ARC decouples *visible communication* from *internal reasoning* so long reasoning traces don't dilute the reward-comparison signal.

## Why it matters

A common failure mode of GRPO-on-agents: the reward looks noisy because one rollout that "asks for clarification and gets it" is scored against a peer that "answered directly and got it right." The two are *both correct behaviors*; comparing them yields useless advantage signal. ARC restores the invariant that group-relative comparisons require peers to be attempting the same kind of thing.

## Gotchas & tricks

- The interaction-type tagger has to be reliable; a mis-tagged rollout gets normalized against the wrong sub-group and injects noise. A simple regex/verifier is usually enough at training time.
- Sub-groups can end up small; add a minimum-count fallback (revert to full-group normalization when a sub-group is degenerate).
- Reported side-effect: shorter responses. When rollouts no longer have to over-explain to compete with each other, the policy naturally emits terser interactions.

## Sources

- Paper: *ARC: Fair Relative Advantage Comparison in Open-Ended Real-World Interaction* — Tong et al., 2026 — [arXiv:2608.13622](https://arxiv.org/abs/2608.13622)

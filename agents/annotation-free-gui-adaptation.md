# Annotation-Free GUI Adaptation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Closed-loop pipeline that adapts a mobile-GUI agent to a new target app without any human-written tasks, demonstrations, or reward labels. The pipeline mines tasks from app exploration, runs rollouts inside the app, harvests hierarchical feedback signals from the rollouts themselves, and uses them to drive policy optimization. MobileForge (Kwai, 2026) is the reference implementation.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [README.md](README.md) · [_gui-agents.md](_gui-agents.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Mobile apps update faster than humans can label tasks for them, so any agent that requires hand-written tasks or rewards lags real apps by months. Annotation-free adaptation removes both: the loop is fully self-supervised on the target app's own affordances.

Four stages run inside one substrate:

1. **Target-app exploration** — the agent crawls the app, building a graph of screens and affordances.
2. **Curriculum mining** — task candidates are auto-generated from the affordance graph (e.g. "open settings → toggle X → return to home"), filtered for feasibility and progressive difficulty.
3. **Rollout execution** — the policy attempts mined tasks inside the app.
4. **Hierarchical feedback-guided policy optimization** — feedback signals harvested from rollouts (UI affordance match, sub-goal progression, episode success) shape the RL update.

## How it works

The novel piece is **hierarchical feedback**. Past annotation-free GUI methods used flat episode-end rewards (success/failure), which is sparse and slow to learn from. MobileForge instead extracts feedback at three granularities and combines them into a per-token credit assignment:

| Granularity | Signal source | What it rewards |
| --- | --- | --- |
| Step-level | UI affordance match (did the tap land on a valid element?) | Action plausibility |
| Sub-goal | progress along the mined curriculum's intermediate states | Movement toward the target |
| Episode | final task success | Overall correctness |

These signals are weighted and fed as the reward into a GRPO-style policy update. Because the curriculum is mined from the same affordance graph that scores step-level feedback, the signal is internally consistent — no human in the loop.

## Why it matters

- Continuous adaptation: when a target app updates, the loop reruns and the policy follows the new affordance graph; no relabeling required.
- The hierarchical-reward idea ports beyond GUI: any environment that leaks intermediate state (terminal sessions, browser DOM, game logs) admits a multi-granularity reward shape and benefits from the same treatment.
- Lowers the data cost of agent specialization, which is otherwise a bottleneck for niche or long-tail apps.

## Gotchas & tricks

- The affordance graph is only as good as the explorer; pathological apps with heavy dynamic content (chat, feeds) fool naïve exploration.
- Mined curricula bias the policy toward easily-discoverable tasks. Real user goals may live off the affordance graph entirely — the recipe doesn't solve discoverability of latent capabilities.
- Feedback weights are environment-specific. The paper reports good defaults but expect to retune per app family.

## Sources

- Paper: *MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization* — Liu et al., Kwai, 2026 — [arXiv:2606.19930](https://arxiv.org/abs/2606.19930).

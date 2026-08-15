# Online Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding used **inside** an RL rollout loop, with the draft model updated online as the target policy evolves. Fixes the stale-draft problem that kills naïve speculative decoding in RL: as the target model changes during training, a fixed draft's acceptance rate collapses and speculative decoding stops paying off. Named as a rollout-throughput technique in Intern-S2-Preview.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md), [../systems/partial-rollouts](../systems/partial-rollouts.md), [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

**Speculative decoding** speeds up inference by having a small "draft" model propose the next $k$ tokens, then verifying them in a single parallel forward pass of the "target" model. Accepted drafts commit immediately; the first rejection resets. The effective speedup is roughly the mean acceptance-run length.

In an **RL post-training loop**, the target policy is being updated every iteration. A draft model trained against yesterday's target is systematically wrong about today's target's next-token distribution — its **acceptance rate collapses** as training progresses, and speculative decoding degrades from a speedup to overhead.

Online speculative decoding keeps the draft **synchronized with the target**: the draft is updated during training (from the target's activations, distillation objectives, or on-the-fly finetuning on target rollouts) so its acceptance rate stays high across iterations. This turns speculative decoding into a viable rollout-throughput lever *inside* RL, not just at deployment.

## How it works

**Standard speculative decoding (single-shot).**

1. Draft $\pi_d$ proposes tokens $x_{t+1}, \dots, x_{t+k}$.
2. Target $\pi_t$ evaluates all $k+1$ positions in one parallel pass.
3. For each position $i$, accept if $r_i = \min\left(1, \frac{\pi_t(x_{t+i} \mid \cdot)}{\pi_d(x_{t+i} \mid \cdot)}\right)$ drawn against a uniform.
4. On first reject, resample from an adjusted target distribution and continue.

**Online extension (used in RL rollouts).**

1. Draft $\pi_d^{(n)}$ tracks target $\pi_t^{(n)}$ via one of:
   - **Distillation from target rollouts** — the draft is trained on tokens the target generated during rollouts, so its next-token distribution matches recent target behavior.
   - **Shared-parameter draft** — the draft reuses layers of the target (MTP-head style: a shallow head over target hidden states, updated jointly).
   - **Periodic full-update** — replace the draft with a distilled snapshot of the target every $N$ iterations.
2. During each rollout iteration, use $(\pi_d^{(n)}, \pi_t^{(n)})$ for speculative decoding as normal.
3. After the trainer produces $\pi_t^{(n+1)}$, refresh $\pi_d$ accordingly.

The refresh policy is where the design lives: too frequent kills training throughput (draft updates cost), too infrequent lets acceptance rate decay between refreshes.

## Why it matters

- **RL rollouts are the throughput bottleneck** in modern long-CoT / agentic RL. Anything that shortens rollout wall-clock scales training directly.
- Static speculative decoding is **incompatible with RL** — a fixed draft's acceptance rate collapses. That barrier previously kept speculative decoding to deployment-time only.
- Online speculative decoding **preserves the speculative-decoding speedup across RL iterations**, without the stale-draft cost. Intern-S2-Preview cites it as a rollout-efficiency lever in its unified post-training pipeline.
- Composes cleanly with **partial rollouts**: partial rollouts handle long-tail latency across iterations, online speculative decoding handles per-token throughput within an iteration.

## Gotchas & tricks

- **Refresh cadence is the tuning knob.** Refresh cost vs. draft staleness — the sweet spot depends on how fast the target moves per iteration. Faster-moving targets (early training) need more frequent refresh.
- **Draft KV cache and target KV cache must stay coherent.** If they share activations (MTP-head style), memory accounting is simpler; if they run as separate models, KV cache sizes double.
- **Rejection distribution needs the correct adjusted resampling** to preserve the target's output distribution — using the raw target distribution on rejection biases the trajectory. Standard speculative-decoding theory applies.
- **Interacts with sampling temperature.** Higher temperature spreads the target distribution; drafts have harder time matching. Speedup drops at high temperatures.
- **On-policy vs. off-policy rollouts.** Online speculative decoding works cleanly for on-policy rollouts (the draft targets the current policy). Off-policy rollouts complicate the draft-training signal — which target should the draft chase?

## Sources

- Paper: *Intern-S2-Preview: Scientific Agentic Foundation Model* — Shanghai AI Laboratory, 2026 — [arXiv:2608.13505](https://arxiv.org/abs/2608.13505) — names online speculative decoding in the unified post-training pipeline.
- Predecessor: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — the deployment-time speculative decoding algorithm.
- Related: [../pre-training/mtp.md](../pre-training/mtp.md) — MTP heads are a natural fit for shared-parameter online drafts.

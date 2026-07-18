# Spectral Rewiring (SAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-hoc, training-free editor for RL post-training updates. Given a delta $\Delta W = W' - W$ from RL fine-tuning (e.g. GRPO), project $\Delta W$ into the top-singular subspace of the *base* model's weights $W$; keep only the aligned component and discard the orthogonal residue. Empirically, the aligned component carries the reasoning capability while the orthogonal component drives interference and reasoning suppression. SAR extracts compact "reasoning cores" that are ~0.58% of parameters yet preserve >99% of post-training performance, and turns model merging across experts into a robust operation.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [../pre-training/model-souping.md](../pre-training/model-souping.md) · [../pre-training/_model-merging.md](../pre-training/_model-merging.md)

---

## What it is

Given a post-trained checkpoint $W'$ and its base $W$, SAR splits the update into two components:

- **Aligned component** — the projection of $\Delta W$ onto the top singular directions of $W$.
- **Orthogonal component** — everything else.

SAR keeps the aligned component and discards the orthogonal one. The output is a rewired $W_{\text{SAR}} = W + \Pi_{\text{top}}(\Delta W)$. No training. No labels.

Two claims motivate the split:

1. **Reasoning lives in the spectral core.** For dense-full-parameter RL post-training, the reasoning-effective part of $\Delta W$ concentrates in the base model's top singular directions.
2. **Interference lives in the orthogonal residue.** Suppressed reasoning at test-time scaling, and cross-domain interference in multi-domain training, are driven by orthogonal noise.

## How it works

1. SVD the base weights: $W = U \Sigma V^\top$, keep the top-$k$ singular directions $U_k, V_k$.
2. Project the update: $\Delta W_{\text{aligned}} = U_k U_k^\top \, \Delta W \, V_k V_k^\top$.
3. Emit $W_{\text{SAR}} = W + \Delta W_{\text{aligned}}$.

Choice of $k$ is the only knob. At the compact end, $k$ chosen so the aligned component uses ~0.58% of total parameters still preserves >99% of post-training performance. Larger $k$ preserves more of the RL update; smaller $k$ purifies more aggressively.

Because the operation is linear and per-matrix, it composes cleanly with model merging: two RL experts $A$ and $B$ can be SAR-rewired against a shared base $W$ and then averaged, and the merged model surpasses both individual experts on cross-domain benchmarks — the orthogonal noise that would normally cancel out the merge is gone before the average.

## Why it matters

- **Reasoning purification.** Removes the orthogonal residue that causes premature test-time-scaling saturation and unlocks higher-k exploration in math and coding.
- **Compact deltas for storage and delivery.** A ~0.58% delta is small enough to ship as a patch, not a full checkpoint — practical for expert routing and LoRA-adjacent workflows.
- **Model merging goes from folklore to tool.** Naive averaging of expert deltas is brittle; SAR-then-average routinely surpasses both single-domain experts *and* prior merging baselines.
- **Empirical claim about post-training geometry.** SAR is evidence that successful RL updates in dense full-parameter fine-tuning are near-low-rank in the base's spectral coordinates — feeds interpretability research on where capability lives in weight space.

## Gotchas & tricks

- **SVDs are per-matrix.** Choose which matrices to rewire (attention Q/K/V/O, MLP up/down/gate) — usually all of them, but keep vocabulary embeddings and layernorm scales unrewired.
- **Base model is the reference.** SAR is defined against $W$, not $W'$. Using $W'$'s spectrum instead leaks post-training bias into the projection.
- **Not a substitute for good post-training.** SAR filters an existing $\Delta W$; it can't rescue an update that had no useful signal in the first place.
- **k selection is empirical.** Sweep $k$ on a held-out reasoning benchmark. The "sweet spot" depends on the RL recipe — RLVR updates are typically lower-rank than preference-based updates.
- **Interaction with LoRA.** LoRA-adapted deltas are already low-rank; SAR still helps because "low-rank in LoRA" is not "aligned with the base's spectrum" — the two projections are different objects.

## Sources

- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — Yu et al., SIA-Lab (Tsinghua AIR × ByteDance Seed), 2026.
- See also: [../pre-training/model-souping.md](../pre-training/model-souping.md) for the naive weight-averaging baseline SAR strengthens.

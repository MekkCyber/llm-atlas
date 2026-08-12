# Counterfactual Evidence Disentanglement (Evidence-RL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training reward augmentation for vision-language models: audit whether the model's answer **causally depends on the local image evidence**. Neutralize an object-centric Evidence Region and check the support drop against matched non-evidence regions — if the answer collapses on Evidence but not on non-Evidence, grounding is real. Combined with correctness inside GRPO, this rewards VLMs that answer *from* the image instead of *around* it.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [README.md](README.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

VLMs are notorious for **shortcut learning**: they answer from language priors, dataset regularities, or irrelevant visual context rather than the specific image region a human would call the "evidence." Prior perception-aware post-training methods encourage image use through global perturbations or attention proxies, but they don't test whether a *sampled answer* causally depends on the *specific local evidence* the model claims to be using.

Counterfactual Evidence Disentanglement (CED) reframes grounding as a **counterfactual test**: if a region really is the evidence, ablating it should collapse answer support — while ablating a matched non-evidence region shouldn't. The gap between the two ablation drops is a grounding signal that plugs directly into GRPO.

## How it works

For a VLM response `y` to a `(question, image)` pair:

1. **Identify Evidence Region.** Weak object-level proposals (from an off-the-shelf detector) generate candidate regions. The region most consistent with `y` per the paper's scoring is the Evidence Region `R_E`.
2. **Match a Non-Evidence Region.** Sample `R_N`, a region of similar size/appearance but semantically distinct from what `y` claims to describe.
3. **Neutralize.** Ablate `R_E` from the image (blur / mask / replace with dataset mean) and re-run the VLM to measure support drop `Δ_E`. Repeat for `R_N` to get `Δ_N`.
4. **Compute reward.** The **evidence-grounding reward** is a function of the gap `Δ_E − Δ_N`: large gap = answer genuinely depends on the Evidence, not on a shortcut.
5. **Combine with correctness inside GRPO.** The overall reward is `r = r_correct + λ · r_grounding`, plugged into standard GRPO.

Two properties matter: (a) no question-specific evidence annotations are needed (weak object proposals suffice), and (b) all counterfactual work happens at *training time* only — inference stays vanilla.

## Why it matters

- **Grounding becomes a training-time signal.** Turns a fuzzy correctness property ("did you actually look at the image?") into a measurable, differentiable-via-RL reward.
- **Plugs into any GRPO stack.** Not a new algorithm — a reward augmentation. Directly composable with RLVR pipelines already used for VLM post-training.
- **No extra inference cost.** All the counterfactual queries are training-time only.
- **Broad wins.** Across 9 VLM benchmarks and 4 backbones, Evidence-RL outperforms prior RL-based perception-aware post-training methods, with ablations verifying the object-centric signal is doing the work.

## Gotchas & tricks

- **Weak proposals are enough, but not any proposals.** Off-the-shelf object detectors work; scene-level proposals (whole regions of the image) don't provide enough resolution.
- **Matched non-evidence sampling matters.** If `R_N` is trivially different from `R_E` (background patch vs foreground object), the gap always looks large — grounding signal degenerates. Pair regions by size, texture, and salience.
- **Ablation choice affects the signal.** Masking with the dataset mean is safer than pure blackout (which is out-of-distribution and gives spuriously large drops).
- **Interaction with correctness reward.** Grounding reward without correctness → model learns to write answers that *look* grounded but are wrong. Correctness without grounding → shortcut learning. Both terms are load-bearing.
- **Object-centric assumption.** The method works because most VQA answers reference object-level entities. For questions about scene properties (weather, mood, style), the Evidence Region concept is less crisp.

## Sources

- Paper: *Evidence-RL: Towards Evidence-intensive Visual Reasoning* — Huang, Yu, Xu, Chen, Yang, He, Yang, Zhang, Hu, 2026 — NUS / Zhejiang / Fudan / Tsinghua / Tencent.
- Related: [../post-training/grpo.md](../post-training/grpo.md) — the RL algorithm this augments.

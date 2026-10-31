# Weakening-Induced On-Policy Self-Distillation (WISD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Self-distillation for reasoning models when no trustworthy *stronger* teacher is available. Instead of trying to construct a stronger self-teacher (voting, longer sampling, extra tools), WISD constructs a *weaker* self-counterpart by **quantizing** or **pruning** the model, then uses the contrast between the full model and its weakened copy as an on-policy training signal. No labels, no privileged info, no external teacher. Works across architectures and scales; recovers up to +12.8 pp on HMMT 2025.

**Prereqs:** [_rl](../_rl.md), [rejection-sampling](../rejection-sampling.md)
**Related:** [long-cot-rl](long-cot-rl.md), [long2short](long2short.md), [livecodebench](../../evaluation/livecodebench.md), [aime](../../evaluation/aime.md)

---

## What it is

Self-distillation needs some asymmetry between a "teacher" view and a "student" view of the same model. Common approaches build the teacher: self-consistency over $k$ samples, longer CoT, tool-augmented rollouts, or external verifiers. All of them need either privileged info (ground-truth labels, external tools) or many expensive rollouts to construct trustworthy teachers from the model's own uncertain predictions.

WISD inverts this. Keep the model as the teacher; build the *student* by weakening a copy of the model — the weakening is cheap, deterministic, and makes the gap trustworthy even without labels:

$$
\pi_{\text{weak}} = \text{Weaken}(\pi_\theta), \quad \text{Weaken} \in \{\text{low-bit quantization},\; \text{magnitude pruning}\}
$$

The on-policy training signal is the contrast $(\pi_\theta, \pi_{\text{weak}})$ on the same prompts — pulling $\pi_\theta$ toward responses on which it beats $\pi_{\text{weak}}$.

## How it works

- **Weakening operators.** Low-bit quantization (e.g., 4-bit) and magnitude pruning are both tested. Pruning tends to produce larger gains, likely because it degrades the model more uniformly across capabilities.
- **On-policy rollouts.** Prompts come from unlabeled math / coding tasks. The model samples $\pi_\theta$'s responses; $\pi_{\text{weak}}$ scores or re-samples the same prompts.
- **Contrast signal.** The exact form (preference pair, reward diff, DPO-style) is implementation-detail; the invariant is "cases where $\pi_\theta$ clearly beats $\pi_{\text{weak}}$ become positive examples; cases where it doesn't become negatives or are discarded."
- **No external teacher, no labels, no verifier.** The weakening is the entire asymmetry source.

## Why it matters

The dominant alternative — reasoning RL with a verifier — requires a verifier and a lot of compute. The dominant *supervised* alternative — distillation from a stronger reasoning model — requires that stronger model to exist. WISD removes both bottlenecks for the long tail of reasoning tasks where ground truth is hard to produce but the model has latent capability to improve.

Headline numbers:

- Average across benchmarks: **+4.0 pp** (quantization-based) and **+5.5 pp** (pruning-based).
- **Qwen3.5-4B**: 57.7 → 70.5 on HMMT 2025 (+12.8); 39.1 → 56.7 on LiveCodeBench v6 (+17.6).
- **Qwen3.6-35B-A3B** (MoE): 68.5 → 74.6 on HMMT 2025 (+6.1); 61.8 → 69.5 on LiveCodeBench v6 (+7.7).

Repurposes quantization and pruning — ordinarily *deployment* tricks — as *supervision* sources.

## Gotchas & tricks

- **Weakening strength is a hyperparameter.** Too weak a weakening gives no signal; too strong (e.g., 2-bit) collapses the student to noise and signal degenerates.
- **Pruning > quantization for most settings** in the paper's results, but both work. Pruning's uniform degradation seems to produce more learnable contrasts.
- **MoE friendly.** Reported on Qwen3.6-35B-A3B (active-parameter MoE) with substantial gains — the weakening operator applies to the active expert path.
- **No verifier.** The signal is implicit in the pair, not in correctness. Downside: the method cannot push past the model's own uncertainty on prompts where the full model is also wrong.
- **Overlaps with contrastive decoding.** Contrastive decoding (Li et al. 2023) also uses a weakened copy for inference. WISD extends the idea to training.

## Sources

- Paper: *Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models* — Song Han, Han Cai et al., MIT · NVIDIA, 2026 — arXiv:2610.38681.
- Related: *Contrastive Decoding* — Li et al., 2023 — same strong/weak gap, used at inference not training.
- Code: https://github.com/jet-ai-projects/WISD

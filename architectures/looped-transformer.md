# Looped Transformer (via Distill-to-Loop)
*Depth — depth-wise weight sharing in a Transformer stack, produced by distillation rather than from-scratch training, with an inference-time loop-count knob.*

**TL;DR:** A looped Transformer reuses the same small block stack N times along depth, so parameter count is roughly constant in N. Training looped Transformers from scratch has historically been unstable. Distill-to-Loop (Dooop) *distills* a pretrained non-looped Transformer into a looped student: the student shares its blocks along depth and is trained to match the teacher via a loop-aware distillation loss. The loop count N becomes an inference-time knob. On SD3-medium, a student with ~37% of the teacher's parameters matches the teacher at 2–4 loops.

**Prereqs:** [transformer-block.md](transformer-block.md)
**Related:** [mla.md](mla.md)

---

## What it is

Looped (or "universal") Transformers share weights along depth: a single block (or a short stack of blocks) is applied N times. This trades parameters for compute. The trouble is training from scratch — looped models are harder to optimize and usually underperform flat Transformers at matched compute. Distill-to-Loop skips that pain: start with a pretrained flat model (teacher), have a looped student imitate it. The student doesn't learn from scratch; it learns to be an efficient *compression* of the teacher's forward pass along depth.

## How it works

1. **Student architecture.** Pick a short block stack (e.g., 4–8 blocks) and loop count `N` during training. Total "effective depth" is stack_size × N, matching the teacher.
2. **Loop-aware distillation loss.** At each teacher layer `L_t`, pick a matching student loop index `L_s = ⌊L_t · N / teacher_depth⌋` and penalize the mismatch between the teacher's hidden state and the student's hidden state at that loop. Optional feature-level and logit-level distillation on top.
3. **Loop guidance (optional, inference-time).** During sampling, use earlier loops' outputs as a classifier-free-guidance-style prior for later loops — strengthens coherence as the loop deepens.
4. **Variable loop count at inference.** Because loops share weights, the model runs at any `N` ≥ 1. Users pick quality-vs-compute at serve time.

The reported experiment is on SD3-medium (image diffusion). Student parameters are ~37% of the teacher; 2 loops approximate the teacher's quality; 4 loops meet or exceed it.

## Why it matters

- **Distillation unlocks loops.** From-scratch looped training has been consistently disappointing. Distilling into a loop sidesteps the optimization difficulty and still captures the parameter savings.
- **Inference-time compute/quality knob.** One model, many quality points. Serve a budget-tight user with N=2 and a quality-sensitive user with N=6.
- **Memory-constant along depth.** Looped models have constant weight memory in depth; the KV cache still grows per layer, but the parameter memory is bounded by the stack size, not depth.
- **Works for diffusion.** Diffusion models are often parameter-bound at serve time; shrinking the DiT by 3× while preserving quality is practically valuable.

## Gotchas & tricks

- **Loop guidance is load-bearing at deep N.** Without it, the model's late-loop outputs drift away from the early-loop signal. Classifier-free-style guidance against the earlier-loop output restores coherence.
- **Student stack size ↔ `N` tradeoff.** A 1-block stack looped 32 times is unstable; an 8-block stack looped 4 times is easy. The paper lands near 4–8 block stacks.
- **Not a pruning method.** The student has new weights (trained from scratch under the distillation objective), not a pruned subset of the teacher's.
- **Teacher must be strong.** The student cannot exceed the teacher by more than small margins. If the teacher is weak, loop more doesn't help.
- **Training signal scales with `N`.** Deeper loops get less supervision per weight-step unless the matching schedule is tuned; expect to tune the per-layer loss weights.
- **Serves orthogonally to quantization.** You can quantize the looped student and keep the loop-count knob. The two compress different axes.

## Sources

- Paper: *Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth* — An et al., 2026 — [arXiv:2611.31977](https://arxiv.org/abs/2611.31977).

# SpecFold
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding for *diffusion language models*. SpecFold (2026) exploits that child branches in a draft tree share most of their computation with their parent — especially under diffusion dynamics where many positions are near-fixed between denoising steps. The method identifies tokens whose hidden state barely changes and reuses the parent-branch computation; divergent tokens are recomputed. A Triton kernel converts this selective reuse into real wall-clock gains: up to 1.64× over Spiffy and 1.99× over vanilla decoding across five models and five benchmarks.

**Prereqs:** [token-level-routing.md](token-level-routing.md), [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [sparse-decoding.md](sparse-decoding.md)

---

## What it is

Classical speculative decoding uses a cheap draft model to propose candidate tokens that the target model verifies in one parallel pass. For multi-branch drafting (several candidate continuations at once), each branch is verified independently and the shared ancestry is wasted. Diffusion language models amplify the waste because the per-position update between adjacent denoising steps is often tiny; recomputing all of it per branch burns FLOPs that produce the same bits.

SpecFold "folds" the shared computation across branches: for each child branch, test whether the hidden state at each position has materially changed relative to the parent; if not, reuse the parent's computation.

## How it works

- **Parent-child residual check.** For every token position in a child branch, compare its would-be hidden state against the parent's. Below a chosen threshold, mark the position as reusable.
- **Selective recomputation.** Only the positions that changed enough to matter run through the target model's forward pass. The rest inherit the parent's cached activations.
- **Fused Triton kernel.** The gather/scatter pattern of selective reuse would ordinarily be bandwidth-bound; the paper fuses it into a Triton kernel that keeps the reused regions resident in SRAM and executes the recomputations in-place.
- **Verification unchanged.** Draft acceptance uses the same probability-ratio criterion as standard speculative decoding, so correctness is preserved.

## Why it matters

- Diffusion LMs are gaining traction (ScaleFlow, LLaDA-style). Serving them well needs speculative-decoding analogues; SpecFold is one of the first to extract real speedup from the branch-sharing structure.
- No retraining required; drops into existing diffusion-LM serving stacks.

## Gotchas & tricks

- The reuse threshold is a quality knob — too permissive leaks parent errors into children, too strict reduces the speedup to parent-dominated branches.
- Works best when draft trees are wide (many sibling branches); narrow trees have little to fold.
- Memory pressure for cached parent activations is nontrivial; the Triton kernel's working set has to fit within SRAM budgets.

## Sources

- Paper: *SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models*, 2026 — [arXiv:2610.04875](https://arxiv.org/abs/2610.04875).

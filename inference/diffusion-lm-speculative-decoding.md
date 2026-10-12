# Speculative Decoding for Diffusion LMs (SpecFold)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion LLMs can verify a parent branch plus several draft branches in one forward pass, but drafts share most tokens with the parent, so most of the extra compute is redundant. SpecFold adds a token-level residual gate: positions whose draft hardly changed reuse the parent's attention and FFN activations, each branch keeping its own residual stream. A Triton kernel turns the reuse into real speedups — up to 1.64× over Spiffy and 1.99× over vanilla.

**Prereqs:** [README.md](./README.md), [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Speculative decoding adapted to **diffusion language models**, where multiple candidate branches are verified in a single denoising forward pass rather than drafted by a separate small model. SpecFold is a drop-in redundancy-reduction layer on top of that pattern.

## How it works

Each branch has a per-position residual stream. SpecFold introduces a **token-level residual gate** between the parent and each draft:

- If the draft's residual at position *t* is below a threshold δ from the parent's, the position *reuses* the parent's attention output and FFN output verbatim.
- If the draft meaningfully diverges at *t*, the position runs the normal computation.

Each branch keeps a distinct residual stream so divergence can grow across subsequent layers even where it was gated early. A Triton kernel implements the gated-skip at the warp level — crucial, because branchy control flow on GPUs usually negates the arithmetic savings.

## Why it matters

Multi-branch verification in diffusion LMs looked like a free-lunch idea until the extra branches started paying nearly full compute. SpecFold's observation — most positions are *shared* with the parent, not new — closes most of that gap. It's orthogonal to temporal caching (reuse across denoising steps) and composes with any speculation strategy the host system uses (pick-k, tree-of-drafts, etc.).

## Gotchas & tricks

- The gate threshold δ is a quality/speed knob. Too low and you reuse divergent positions; too high and you compute everything.
- Gains scale with the number of parallel branches; two-branch setups see modest wins, five-branch setups approach the 1.99× ceiling.
- Reused activations are *per-layer*; a draft that diverged at layer 3 can still gate layer 5 if it re-converges. This is why keeping per-branch residual streams matters.
- Benchmarks (2 model families, 5 models, 5 benchmarks) are mostly text; multi-modal diffusion LMs are untested in the paper.

## Sources

- Paper: *SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models* — 2026 — [arXiv:2610.04875](https://arxiv.org/abs/2610.04875)

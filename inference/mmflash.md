# MMFlash
*Depth — visual-dependence-weighted diffusion drafter for speculative decoding on vision-language models.*

**TL;DR:** A speculative-decoding drafter tailored to VLMs. Identifies which draft tokens *depend* on the visual input, up-weights them in the drafter's training loss so acceptance rates stay high on grounded tokens, and uses a diffusion-LM drafter with sparse attention so long video contexts stay tractable. Up to 3.58× lossless speedup over the VLM baseline, beating EAGLE-3 by up to 195.4% and DFlash by up to 17.1%.

**Prereqs:** [./speculative-decoding.md](./speculative-decoding.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Speculative decoding gives large text LLMs 2–4× latency wins. On VLMs the gain is smaller because drafters mis-predict exactly the tokens that depend on image/video context — colors, counts, object names — so the verifier rejects them. MMFlash fixes the drafter's training objective so it *learns to be right on the grounded tokens* and uses a diffusion-LM drafter that refines many positions in parallel.

## How it works

1. **Visual-dependence metric.** For each training token, compare the drafter's predicted distribution with the visual input intact vs. perturbed (masked / shuffled image). The KL gap scores *how much that token's prediction depends on vision*.
2. **Weighted drafter training.** Reweight the drafter's loss by the visual-dependence score, concentrating capacity on exactly the tokens where speculative decoding currently fails.
3. **Diffusion-LM drafter.** The drafter is a diffusion LM producing `k` positions in parallel via iterative refinement rather than left-to-right autoregression; sparse attention keeps long video contexts affordable.
4. **Standard verifier.** The target VLM scores the `k` proposed tokens in one forward and accepts the longest consistent prefix under the usual speculative-decoding acceptance rule.

## Why it matters

Reopens speculative decoding for VLMs — a modality where the technique was underperforming. The visual-dependence weighting is modality-agnostic: the same recipe (identify tokens whose prediction depends on side-information X, weight the drafter loss by that dependence) should transfer to audio-conditioned or retrieval-augmented models.

## Gotchas & tricks

- Visual-dependence scoring is expensive — perturbing every image adds ~2× cost over plain drafter training. Report uses a subsampled schedule.
- Diffusion drafters give larger `k` but need a few refinement steps per draft; the net-wall-clock win comes from parallelism in the refinement, not from doing more work than an AR drafter.
- Sparse attention on long video contexts is key: the drafter would otherwise dominate end-to-end latency on long-video inputs.
- Sampling temperature mismatch between drafter and target still hurts acceptance — align temperatures as usual.

## Sources

- Paper: *MMFlash: Faster Speculative Decoding for Vision-Language Models* — Chen, Wang, Fang, Wang (NUS), 2026 — [arXiv:2611.18482](https://arxiv.org/abs/2611.18482)
- Related: EAGLE-3, DFlash speculative-decoding baselines.

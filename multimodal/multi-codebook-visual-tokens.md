# Multi-codebook visual tokens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Discrete single-codebook visual tokenizers (VQ-VAE lineage) align cleanly with discrete text LMs but have a hard **representational-capacity ceiling** per position (log₂|V| bits). Multi-codebook tokenizers represent each visual position as a **group of codes from complementary codebooks**, raising capacity without leaving the fully discrete regime. Pair with a lightweight **AR head over codes within a position** on top of a diffusion-LM backbone that processes positions in parallel.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

Three recipes compete for unified text + image LMs:

| Recipe | Visual rep | Backbone objective | Problem it has |
| --- | --- | --- | --- |
| Single-codebook discrete AR | One VQ code per patch | Shared next-token prediction | Capacity bottlenecked at log₂|V| per patch |
| Continuous latent (DiT) | Continuous latent | Diffusion loss over images, CE over text | Heterogeneous objectives, modality interference |
| Hybrid (semi-discrete) | Mixed | Mixed | Complexity + both the above problems |

**Multi-codebook discrete** adds a fourth option: keep the discrete-token / shared-CE objective, but represent each position with $k$ codes from $k$ *complementary* codebooks. Total capacity per position: $k \cdot \log_2 |V|$ bits — scalable by adding codebooks.

## How it works

### Multi-codebook tokenizer

For each visual position (patch, 3D voxel, video spacetime token), encode with $k$ parallel encoders / decoders that produce $k$ discrete codes:

$$
\text{pos}_j \mapsto (c_{j,1}, c_{j,2}, \ldots, c_{j,k}),\ c_{j,i} \in \{1, \ldots, |V_i|\}
$$

Codebooks are trained to be *complementary* — a loss that discourages redundancy across the $k$ codes for the same position (orthogonal residuals, or VQ-of-residuals, depending on recipe).

### Input embeddings

At the LM input, embed each code and **concatenate** at the position: $e_j = [E_1(c_{j,1}), E_2(c_{j,2}), \ldots]$. The backbone sees one embedding per position, same shape as a text token embedding, no change to attention geometry.

### Output: AR head over codes within a position

At the output, the backbone predicts position embeddings in parallel (diffusion LM style — all positions updated per step). A **small AR head** then unrolls the codes within each position sequentially, in a fixed codebook order:

$$
p(c_{j,1}, \ldots, c_{j,k} \mid h_j) = p(c_{j,1} \mid h_j) \prod_{i=2}^{k} p(c_{j,i} \mid h_j, c_{j,<i})
$$

Autoregressive **only along the codebook axis**, parallel along the position axis. Keeps the sampling cost dominated by the diffusion backbone's parallel updates.

### Stability-aware sampling

To hit high throughput, Intern Lumina U2 uses a **confidence-plateau unmasking** schedule (unmask positions once their next-token confidence plateaus) and refreshes caches to keep already-decided positions consistent. Reported 4.45× generation speedup.

## Why it matters

- **Keeps the shared discrete objective.** All of text, image, video, 3D decode under the same CE / masked-token loss over discrete codes — no two-headed training, no modality-dependent loss weighting.
- **Breaks the capacity ceiling.** Single codebooks need $|V| \sim 2^{16}$ to compete with continuous latents; multi-codebook recovers capacity at modest per-codebook sizes.
- **Backbone stays diffusion-parallel.** The AR head is per-position (short), so position-wise parallelism remains the dominant factor.
- **Trains cross-platform.** Intern Lumina U2 reports full-stack training on both Ascend and NVIDIA — the architecture plays well with vendor-neutral code paths.

## Gotchas & tricks

- **Complementarity is the design point.** If codebooks aren't trained to decorrelate, you pay for $k$ codes but encode <$k$×log₂|V| bits of real information.
- **AR head order matters.** The autoregressive factorization inside a position is a modeling choice; some orders (coarse-to-fine) are measurably easier to learn than others.
- **Masked-token training on $k$ codes.** You can mask whole positions or individual codes within a position; mixed masking helps the AR head learn robustly.
- **Discrete loss weighting.** Despite the shared CE objective, per-modality weights (text vs image vs video) remain a knob — text is cheaper and risks dominating if not controlled.
- **Not the same as RQ-VAE.** Residual quantization is a *sequential* multi-code scheme stacked on one residual stream; multi-codebook here uses *parallel* complementary codebooks. Related but distinct.

## Sources

- Paper: *Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation* — Qiu, Yan, Li, Qin, et al. (Shanghai AI Laboratory + partners), 2026 — [arXiv:2610.20818](https://arxiv.org/abs/2610.20818).
- Code: https://github.com/InternLM/InternLumina-U2
- Related tokenizer family: RQ-VAE (Lee et al., 2022); FSQ (Mentzer et al., 2024); AToken unified 4D representation.

# NarrowSpec
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A speculative-decoding trick: the **draft's residual-stream width can be $\frac{1}{2}$ or $\frac{1}{4}$ of the target's** with largely unchanged acceptance. For the DFlash family, the trained output adapter can be *folded into the target's `lm_head`* so logits are bitwise-equivalent after folding. Measured on Qwen3-8B and GPT-OSS-20B across a B200 and a DGX Spark (GB10): **1.06–1.12× speedup on B200, 1.13–1.44× on GB10** at batch 1, temp 0. Speedups persist at larger batches (up to 1.21×).

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md)
**Related:** [vspark.md](vspark.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

A narrow-draft variant of hidden-state-based speculative decoders (EAGLE-3, DFlash). The draft operates on a residual stream of width $d/2$ or $d/4$ where $d$ is the target's hidden dimension. A linear "output adapter" projects the draft's narrow hidden state back to the target's vocabulary. For DFlash, this adapter is linear in the lm-head space and can be absorbed post-hoc, leaving the target's logits unchanged.

## How it works

1. **Narrow draft.** Attention heads, MLPs, and layer-norms all run at width $d' < d$. Parameter count and FLOPs scale with $d'^2$, so a $d/2$ draft is ~4× cheaper in matmul and roughly $4\times$ smaller.
2. **Adapter.** A trained linear map $W_{\text{adapt}} \in \mathbb{R}^{d \times d'}$ lifts the narrow hidden state to the target's width before the lm-head.
3. **Fold-in (DFlash only).** The target's lm-head $W_{\text{lm}} \in \mathbb{R}^{|V| \times d}$ composes with $W_{\text{adapt}}$ into $W_{\text{lm}} W_{\text{adapt}} \in \mathbb{R}^{|V| \times d'}$ — a new *narrow* lm-head applied directly on the draft's hidden state. Logits produced by the folded head match the unfolded two-stage pipeline exactly.
4. **Verify as usual.** The target model's own forward pass on the proposed tokens remains unchanged; speculative-decoding acceptance rule is unaffected.

## Why it matters

- **Pure engineering win.** Narrower draft = smaller artefact, less VRAM, faster matmul — no architectural surgery to the target.
- **Edge / consumer host.** 1.44× on DGX Spark (GB10) matters most where VRAM is tight; narrowing the draft leaves more memory for KV cache and batch concurrency.
- **Invalidates "match the target width" folklore.** EAGLE/Medusa originally assumed draft width = target width because of residual-stream compatibility; NarrowSpec shows the compatibility was an artefact of the adapter, not a requirement.

## Gotchas & tricks

- **Acceptance drops a little.** $d/4$ drafts do give up some acceptance versus $d/1$; the win is in per-step cost, not acceptance length. Sweep empirically.
- **Fold-in requires linear adapter.** Nonlinear adapters (an MLP in the adapter path) can't be absorbed; keep the adapter affine for logit parity.
- **Not fully evaluated at high batch size.** Speedups persist to 1.21× at larger batches but spec decoding's advantage narrows as decode becomes compute-bound.
- **Still needs a trained narrow draft.** Not a free retrofit — retraining or distilling a narrower draft is required, though the narrow draft trains faster than the full-width one.

## Sources

- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — corresponding author `avalliappan@nvidia.com` (NVIDIA), 2026 — primary source.
- Related: EAGLE / EAGLE-3 (Li 2024), DFlash — the draft families NarrowSpec specialises.

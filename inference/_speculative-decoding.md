# Speculative Decoding

*Taxonomy — the family of inference techniques that trade a cheap drafter for parallel verification to amortise an expensive target model.*

**TL;DR:** Autoregressive decoding runs one forward pass per token. Speculative decoding runs a **cheap drafter** forward, proposes K tokens in a row, then verifies all K with the **target model** in a single parallel forward pass. Accepted tokens come for free; rejected tokens roll back to the first disagreement. Variants differ in how they draft (small model, early-exit, tree, lookahead, MTP) and when they verify (fixed K, confidence-gated loop).

**Related taxonomies:** [_rl](../post-training/_rl.md) *(unrelated; pointer for navigation)*
**Depth files covered here:** [looped-speculative-decoding](looped-speculative-decoding.md)

---

## The problem

For a target model $M_T$ with per-token latency $t_T$, naïve decoding takes $N \cdot t_T$ wall-clock for $N$ tokens. GPUs are massively underutilised during this: they are compute-bound on a prefill of many tokens but memory-bound (bandwidth-limited) on a single-token decode. Serving systems want to **batch forward passes along the sequence dimension**, not just along requests, which plain autoregressive decoding forbids.

## The shared pattern

Every variant has three stages:

1. **Draft** — produce $K$ candidate tokens cheaply.
2. **Verify** — run the target model once over the K-token draft, in parallel, scoring each position.
3. **Accept/reject** — keep the longest accepted prefix (per the acceptance rule, typically speculative sampling preserving the target's distribution). Rejected tokens are discarded; the next draft starts from the last accepted position.

The acceptance rule is **lossless** when it preserves the target's token distribution. The speedup is $(1 + \alpha K) / (1 + K \cdot t_D/t_T)$ for acceptance rate $\alpha$ and drafter latency $t_D$.

## Variants

| Technique | How it drafts | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Small-model (classical) | A tiny model with the same tokeniser | Needs a separately trained drafter | Simple deployments; large target |
| Medusa | Extra heads on the target itself | Needs training; limited depth | No separate drafter |
| EAGLE / EAGLE-3 | Shallow transformer re-using target hidden states | Specialised training | Highest acceptance rate per draft length |
| MTP ([mtp](../pre-training/mtp.md)) | Multi-token prediction built into pretraining | Not a bolt-on; architecture-level | Models trained with it from scratch (DeepSeek-V3) |
| Lookahead | Jacobi iteration; no drafter | Draft length limited | Zero-training retrofit |
| [looped-speculative-decoding](looped-speculative-decoding.md) (DLoop) | Keep drafting while confident; verify once | Needs confidence calibration | Stacks on top of any drafter |

## How to choose

- **Have an aligned small model with the same tokeniser?** Classical spec-dec. Simplest baseline.
- **Training the target from scratch and compute allows?** Build in MTP. Best structural win.
- **Shipping a current frontier model?** EAGLE-3 is the current SOTA bolt-on.
- **Already running EAGLE/Medusa/MTP and want 20–40% more?** Stack DLoop on top.

## Adjacent but distinct

- **Prefix caching** — reduces prefill, not decode; orthogonal.
- **KV offloading / persistence** — see [kv-persistence](kv-persistence.md); different axis (memory, not latency).
- **Continuous batching** — scheduler-level; composes with spec-dec.

## Sources

- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022.
- *Medusa* — Cai et al., 2024.
- *EAGLE / EAGLE-3* — Li et al., 2024–2026.
- *DLoop: Looped Speculative Decoding* — Gu et al., 2026 — [arXiv:2610.07659](https://arxiv.org/abs/2610.07659)

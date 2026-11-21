# VSpark
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding adapted to **autoregressive Vision-Language-Action (VLA) models** — robots that emit actions one token at a time. Three changes make a draft model practical for VLAs: (i) train the draft on rollouts with *calibrated action noise*, (ii) *align the draft's hidden features* to the VLA's, and (iii) *reuse predictions from rejected drafts*. Reports **lossless 1.28–2.10× end-to-end speedup** on 4 VLAs × 4 manipulation benchmarks.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md)
**Related:** [narrowspec.md](narrowspec.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

VSpark is a draft-and-verify scheme for autoregressive VLAs (RT-2-style action-token generators). The draft produces candidate action-token chunks; the target VLA verifies them in a single forward pass as in standard speculative decoding. The novelty is the training recipe that makes the draft accept-friendly for action streams, plus a reuse rule for rejected drafts.

## How it works

Three recipe changes versus text-LM speculative decoding:

1. **Action-noise-calibrated rollouts.** The draft is trained on trajectories where small Gaussian noise is added to the VLA's own action tokens, with scale calibrated so the induced token distribution matches what a drafter actually sees at inference time (slightly off-policy w.r.t. the VLA). Without this, the draft overfits to clean teacher-forced sequences and misses the drift it has to speculate over.
2. **Hidden-feature alignment.** An auxiliary MSE loss aligns the draft's last-layer hidden states with the VLA's at matched positions, so acceptance selects on semantically compatible features, not just token IDs. Lifts acceptance length on continuous-control action vocabularies.
3. **Rejected-draft reuse.** On a rejection at position $j$, the draft's logits for positions $> j$ still carry information. VSpark re-uses them as the *starting point* for the next speculation window, rather than discarding them as vanilla speculative decoding does.

## Why it matters

- **Latency is a control-loop constraint.** VLAs drive real hardware at policy frequencies; a 1.5× decode speedup directly translates to faster control.
- **Lossless.** Policy quality unchanged — verified on 4 VLA families × 4 manipulation benchmarks.
- **Draft recipe generalises.** The three changes (noise-calibrated rollouts, feature alignment, rejected-draft reuse) are not VLA-specific; they're plausible defaults for any non-text autoregressive target (music, video tokens).

## Gotchas & tricks

- **Noise scale calibration.** Too little noise → draft looks like teacher-forced text LM, misses off-policy drift; too much → draft diverges and acceptance collapses. The paper reports a schedule that bakes the scale into the training curriculum.
- **Hidden-feature alignment loss can dominate.** If the MSE term's weight is too high, the draft's next-token prediction degrades. Treat as regulariser with a small coefficient.
- **Rejected-draft reuse must still verify.** The reused logits only prime the next draft; the target still verifies, so losslessness holds.
- **Not tested at scale on long-horizon tasks.** All benchmarks are manipulation-scale; long-horizon planning is left open.

## Sources

- Paper: *VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models* — Suho Yoo, Youngjoon Jang, Hyebin Cho, Taeyeop Lee, Jaewook Jeong, Joon Son Chung (KAIST AI), 2026 — primary source.
- Code: https://github.com/suho-yoo/VSpark
- Project page: https://suho-yoo.github.io/VSpark/

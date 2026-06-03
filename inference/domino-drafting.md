# Domino — Causal-Corrected Parallel Drafting
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A speculative-decoding drafter that decouples causal-dependency modeling from expensive autoregressive draft execution. A parallel backbone produces base logits for the entire draft block in one forward pass; a lightweight **Domino head** (GRU summarizer + low-rank residual head) then injects causal information as a **logit-space residual**, avoiding the per-token full-LM-head computation that bottlenecks autoregressive drafters like EAGLE-3.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md), [draft-on-policy-distillation](draft-on-policy-distillation.md)

---

## What it is

Speculative decoding's quality–cost tradeoff splits drafters into two camps. **Autoregressive** drafters (EAGLE-3) model `q(x_{t+i} | x_{<t+i})` with explicit causal dependency between draft tokens — high acceptance length, but $\gamma$ sequential draft-LM-head calls per cycle. **Parallel/block** drafters (DFlash, DART) emit `q(x_{t+1:t+γ} | x_{≤t})` in one forward pass — cheap, but intra-block dependency is lost and acceptance length drops. Domino keeps the parallel backbone and adds a thin causal-correction branch in *logit space* to recover most of the acceptance gain at a fraction of the cost.

---

## How it works

### Architecture

```
prefix x_{≤t} ─▶ parallel backbone (DFlash-style) ─▶ block hidden states H_{t:t+B-1}
                                                            │
                                                            ▼
                                              frozen target LM head
                                                            │
                                                            ▼
                                            base logits L_i^base for all i

draft tokens so far  ──▶  GRU causal encoder  ──▶  causal state S_{i-1}
                                                            │
                          (H_i, S_{i-1}) ─▶ low-rank MLP (rank r) ─▶ ΔL_i

                                        final logits L_i = L_i^base + ΔL_i
                                            sample x_{i} ~ softmax(L_i)
```

Two components: a parallel backbone (one transformer forward → block hidden states), and a small **Domino head** with (1) a GRU summarizing previously sampled draft tokens into causal state $S_{i-1}$ and (2) a low-rank correction MLP producing a logit-space residual $\Delta L_i = W_2\,\sigma(W_1[H_i;S_{i-1}])$. The correction lives in *logit space*, not hidden space, so the expensive full LM head runs once on the whole block instead of $\gamma$ times.

Reported sizes for Qwen3-8B: GRU hidden 1024, low-rank bottleneck $r=256$, draft block size 16. Total head overhead: +56M params (+5.3%) and +2.8% draft-then-verify latency vs DFlash.

### Training

Two design choices, both load-bearing:

1. **Teacher forcing of the causal encoder.** Feed the GRU ground-truth tokens, not self-generated prefixes. Self-generated prefixes are noisy early in training, and only draft positions whose previous tokens are *accepted* by verification contribute to acceptance length — so the relevant regime is exactly the teacher-forced one. Empirically +0.16 acceptance length over training-time-test (TTT) sampling.
2. **Base-anchored curriculum.** With clean prefixes, the correction branch can shortcut the parallel backbone. Mix base- and final-logit losses with $\mathcal{L} = (1-\lambda_t)\mathcal{L}_{final} + \lambda_t\mathcal{L}_{base}$, annealing $\lambda_t: 1 \to 0$. Forces the backbone to learn strong base logits before the head takes over residual correction. Adds another +0.23 acceptance length.

---

## Why it matters

- **5.49× end-to-end speedup** on Qwen3-8B under Transformers (Domino vs vanilla decoding), 5.8× throughput under SGLang; +16.6% acceptance length vs DFlash with negligible extra latency.
- **Beats both camps simultaneously** on same-data ablation: out-throughputs autoregressive drafters (EAGLE-3) by removing the sequential draft loop, and out-acceptance-lengths parallel drafters (DFlash, DART) by re-injecting causal information.
- **Logit-space residual is the new design pattern.** Hidden-space correction would require re-running the full LM head; the logit-space residual sidesteps this. Likely template for future drafter designs and compatible with vocabulary-reduction tricks (FR-Spec, SpecVocab).
- **Composable with [Draft-OPD](draft-on-policy-distillation.md)** for training, with paged-attention serving, and with tree-structured verification.

---

## Gotchas & tricks

- **Don't put the correction in hidden space.** Re-introduces the per-token LM-head cost the parallel backbone was designed to avoid. The whole point of Domino is the logit-space residual.
- **Teacher forcing alone collapses the backbone.** Skip the base-anchored curriculum and the backbone loss stalls while the correction branch overfits clean prefixes. Anneal $\lambda$, don't drop it cold.
- **Bottleneck rank matters but isn't critical.** $r=256$ for $d=4096$ vocab projections in Qwen3-8B; smaller works at small quality cost.
- **GRU, not transformer.** A lightweight GRU is enough to summarize at most 16 preceding draft tokens; using a transformer would inflate the head latency without helping acceptance.
- **Currently SGLang-only kernel.** Other serving frameworks need port work.

---

## Sources

- Paper: *Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding* — Huang, Zhang, Zhang, Lin, Xu, Zhang — SJTU EPIC Lab / Huawei, 2026 — [arXiv:2605.29707](https://arxiv.org/abs/2605.29707).
- Code: https://github.com/jianuo-huang/Domino
- Background: *DFlash: Block Diffusion for Flash Speculative Decoding* — Chen et al., 2026 — the parallel-backbone baseline Domino corrects.
- Background: *EAGLE-3* — Li et al., 2025 — the autoregressive drafter Domino out-throughputs.

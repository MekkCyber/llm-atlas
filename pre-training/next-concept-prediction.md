# Next Concept Prediction (NCP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretraining objective that augments standard next-token prediction (NTP) with prediction of the **next discrete concept** — a code from a product-quantized codebook built directly from the model's own hidden states. Concepts span multiple tokens, so the model gets a coarser-grained forecasting target alongside NTP. Predicted concepts are fed back into the token stream, so autoregressive token generation is preserved. Demonstrated at 8.9B / 5.73T tokens (NCP-ArchPreview), reaching OLMo-3-7B's final pretraining loss at ~51% of the tokens.

**Prereqs:** [attention](../fundamentals/attention.md), [mtp](mtp.md)
**Related:** [_tokenization](../fundamentals/_tokenization.md)

---

## What it is

An extra head trained jointly with NTP. Where MTP predicts the *token* $k$ steps ahead, NCP predicts a *concept* — a discrete code representing a longer-range summary of the next $k$ hidden states — from a codebook that is itself learned by product-quantizing the backbone's hidden states. Concepts are then embedded back into the token stream so the main NTP head can use them as extra context.

## How it works

1. **Codebook construction.** During training, snapshots of hidden states from mid-to-late backbone layers are product-quantized (split the hidden vector into groups, k-means each group) into a discrete concept vocabulary. The codebook is refreshed periodically so it tracks the model's evolving representation.
2. **Concept Module.** A small transformer head takes the backbone's hidden state at position $t$ and predicts the concept code that summarizes hidden states over positions $[t+1, t+k]$. Loss: cross-entropy over the PQ codebook (one factor per PQ group).
3. **Feedback loop.** The predicted concept is embedded and injected back into the token stream — either concatenated to the token embedding at step $t+1$ or attended to as an extra key/value — so the token head sees the concept it just predicted when it produces $x_{t+1}, \ldots, x_{t+k}$.
4. **Joint objective.** $L = L_{\text{NTP}} + \lambda \cdot L_{\text{NCP}}$, end-to-end. Autoregressive text generation is unchanged at inference; the concept head can be dropped or used as a scaffold.

## Why it matters

- **Denser + longer-horizon supervision.** NTP is myopic; MTP extends horizon by a few tokens; NCP extends it by a *concept*. Bigger training signal per position without leaving the AR paradigm.
- **Token-efficient pretraining.** NCP-ArchPreview reaches OLMo-3-7B's final loss at ~51% of the tokens and beats it by +2.45 macro-avg (+5.99 on GSM8K) after a full run.
- **Compatible with existing infra.** The codebook is derived from the model's own hidden states — no external encoder, no VAE. Inference is standard AR decoding.

## Gotchas & tricks

- **Codebook drift.** Hidden states move fast early in training; refresh the codebook frequently in the first few percent of tokens, then anneal.
- **PQ groups vs. dimension.** More PQ factors = richer concept space but harder cross-entropy target. Balance so per-factor codebook stays learnable.
- **λ schedule.** Similar to MTP — keep NCP weight higher early, taper toward the end so the terminal checkpoint is NTP-optimized.
- **Feedback must respect causality.** The predicted concept fed back at position $t+1$ must have been produced from the causal state at $t$; leaking future hidden states into the codebook target defeats the point.
- **Don't confuse with MTP.** MTP predicts the next few *tokens* using additional AR heads on top of the backbone. NCP predicts the next *concept* — a discrete code over hidden-state summaries — and feeds it back into the token stream. MTP's target is the model's own token vocabulary; NCP's target is a jointly-learned codebook.
- **Concept span length.** Bigger $k$ = longer-horizon target but noisier hidden-state summary. NCP-ArchPreview keeps $k$ modest (a few tokens).

## Sources

- Paper: *NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction* — Intern-NCP Team, 2026 — [arXiv:2609.10715](https://arxiv.org/abs/2609.10715)
- Related: MTP (Gloeckle et al. 2024) — the token-level ancestor.
- Related: Large Concept Model (Meta, 2024) — a concept-level LM that operates *entirely* in a fixed sentence-embedding space, in contrast to NCP's jointly-learned codebook + AR token head.

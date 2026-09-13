# Next Concept Prediction (NCP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretraining objective that runs a **discrete concept-prediction head alongside standard next-token prediction**. At each position, the backbone predicts token $t{+}1$ (as usual) *and* a discrete "concept" summarizing the short span $t{+}1..t{+}k$ in a learned latent vocabulary. Denser per-position supervision without changing the token-level generation contract at inference. NCP-ArchPreview (8.9B) matched OLMo-3-7B pretraining loss on ~51% of the tokens and beat it by 2.45 pts on downstream macro-average.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md), [mtp.md](./mtp.md)
**Related:** [mid-training.md](./mid-training.md), [../data/dolma.md](../data/dolma.md)

---

## What it is

The standard LM objective learns one prediction per position: $P(x_{t+1} \mid x_{\le t})$. NCP adds a *concept* prediction that summarizes a longer horizon in a **compressed discrete space**, trained jointly with the token loss.

Compared to Multi-Token Prediction ([mtp.md](./mtp.md)), which predicts the exact identity of tokens $t{+}2, t{+}3, \ldots$, NCP predicts a *categorical concept* over a learned codebook of ~$K$ atoms. The concept target is much lower-entropy than the raw token span, so the auxiliary loss becomes a cleaner signal about long-range structure.

## How it works

Two heads, one backbone. At each position the backbone hidden state $h_t$ feeds:

1. A standard LM head → $P(x_{t+1} \mid h_t)$ — the primary objective.
2. A concept head → $P(c_t \mid h_t)$ over a learned codebook $C = \{c_1, \ldots, c_K\}$, where the target $c_t^*$ is the concept assigned to the future span $x_{t+1..t+k}$ by an auxiliary encoder trained jointly (VQ-style).

The training loss is $L = L_{\text{NTP}} + \lambda L_{\text{NCP}}$ with $\lambda$ decayed over training (higher early, lower late), matching MTP's schedule pattern.

At **inference time only the LM head runs** — the concept head is discarded, so decoding cost, KV cache and serving contracts are identical to a plain NTP model.

## Why it matters

- **Token efficiency at pretraining scale.** NCP-ArchPreview matches OLMo-3-7B's final pretraining loss with only 51.3% of the tokens, then beats it by +2.45 macro-avg pts on downstream benchmarks. Same order of magnitude as MTP's speedup but from a representation signal, not an inference one.
- **Latent-space language modeling foothold.** A discrete concept space alongside tokens is a step toward reasoning happening in a compressed latent while fluency stays at the token level — a much cleaner factorization than pure "latent CoT" proposals.
- **Zero inference cost.** Unlike MTP-with-speculation, NCP doesn't need any inference-time infrastructure change; the concept head simply isn't loaded.

## Gotchas & tricks

- **Concept codebook design is the whole game.** Vocabulary size $K$, whether the codebook is jointly trained or frozen, how targets are assigned (nearest-code vs. Gumbel routing) — abstract does not disclose specifics for NCP-ArchPreview; treat as unknown.
- **$\lambda$ schedule matters.** Same pattern as MTP: keep $\lambda$ high early where dense supervision helps representations most, decay late so the final checkpoint optimizes primarily for next-token.
- **Concept head is discardable.** Because the concept head isn't used at inference, it can be a full-size transformer block without any serving cost. This is different from MTP's speculative-decoding path where the extra module must run.
- **Baseline is OLMo-3-7B.** All efficiency claims are relative to that recipe; unclear whether the gain compounds on top of MTP or replaces it.

## Sources

- Paper: *NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction* — Intern-NCP Team (Shanghai AI Lab), 2026 — arXiv:2609.10715.
- Related: *Better & Faster Large Language Models via Multi-token Prediction* — Gloeckle et al., 2024 — the token-level ancestor. See [mtp.md](./mtp.md).

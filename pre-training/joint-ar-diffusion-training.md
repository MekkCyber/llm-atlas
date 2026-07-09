# Joint AR + diffusion pretraining
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a single language model with a **joint next-token (AR) + masked-block reconstruction (diffusion) objective**, so at inference the same weights can be run in either mode — or in a hybrid "self-speculation" mode where diffusion drafts and AR verifies. The two objectives are complementary: diffusion teaches parallel lookahead and planning, AR teaches left-to-right linguistic priors. Nemotron-Labs-Diffusion (3B/8B/14B) shows this scales and delivers a single-checkpoint tri-mode LM competitive with pure-AR baselines.

**Prereqs:** [mtp.md](./mtp.md)
**Related:** [../inference/self-speculation-decoding.md](../inference/self-speculation-decoding.md), [../inference/_speculative-decoding.md](../inference/_speculative-decoding.md)

---

## What it is

Diffusion LMs (SEDD, Score Entropy DL, Mask-Predict, LLaDA) train by masking a random subset of tokens and having the model fill them in — a parallel, iterative denoising process. AR LMs train by predicting the next token given all previous. Historically these have been *alternative* training regimes with different serving profiles: AR gives strong linguistic quality at 1 token/pass; diffusion gives parallel multi-token filling but weaker in-domain quality on long-form text.

Joint AR + diffusion training folds both into a single model. The loss on any given batch is a mixture of an AR next-token loss on some sequences (or some positions) and a diffusion masked-reconstruction loss on others. Same weights, both signals.

## How it works

**Objective.** On each training example, choose a mode (or a mixture):

- **AR mode** — standard next-token loss $\sum_t -\log P_\theta(x_{t+1} \mid x_{\le t})$ under a causal attention mask.
- **Diffusion mode** — sample a mask $M \subset [1..n]$, replace $x_i$ with a special mask token for $i \in M$; loss is $\sum_{i \in M} -\log P_\theta(x_i \mid x_{[1..n] \setminus M})$ under a block-bidirectional attention mask.

Both use the same transformer weights but different attention patterns (causal vs bidirectional over the unmasked context). Positional / segment embeddings must handle both patterns without collapsing one.

**Complementarity.** In Nemotron-Labs-Diffusion's ablations:

- **Diffusion improves lookahead planning** — filling multiple future tokens forces the model to plan globally rather than one-step.
- **AR provides left-to-right linguistic priors** — natural next-token flow, correct discourse structure.

Removing either objective hurts the other mode.

**Inference tri-mode.**

- Pure AR — one token per forward pass; used under high concurrency where memory dominates.
- Pure diffusion — a block of tokens per forward pass; used when parallel lookahead pays off.
- Self-speculation — diffusion drafts a block, AR verifies. See [self-speculation-decoding.md](../inference/self-speculation-decoding.md).

## Why it matters

- **Kills the "AR vs diffusion for text" debate** — you don't have to choose. One model, both inference profiles.
- **Serving flexibility.** Same weights adapt to concurrency: high-load → pure AR; low-load → diffusion or self-speculation for latency.
- **Better self-speculation than external drafters.** No drafter–verifier distribution gap because the drafter *is* the verifier. Beats MTP-style parallel drafters on acceptance rate at 8B–14B.
- **Speed-of-light headroom.** 76.5% more tokens per forward pass than self-speculation under an optimal sampler — evidence the current implementations leave large gains on the table.
- **Scales.** Nemotron-Labs-Diffusion at 3B/8B/14B outperforms open AR and open diffusion LMs on both accuracy and speed. This is not a toy-scale demonstration.

## Gotchas & tricks

- **Attention-mask plumbing.** The engine has to support both causal and block-bidirectional masks efficiently. FlashAttention supports both, but custom masking overhead can eat the diffusion mode's throughput win.
- **Mixing ratio between AR and diffusion loss matters.** Nemotron doesn't fully disclose the ratio; too much diffusion hurts linguistic quality, too little means the diffusion mode is undertrained. Expect it to be a schedule (early diffusion-heavy, late AR-heavy) rather than a constant.
- **Mask token vocabulary slot** — the diffusion mode needs a dedicated mask token in the vocab, not repurposed from a real token. Adds one embedding.
- **Positional encoding under bidirectional attention** must still be well-behaved at the target length; RoPE/YaRN/DCA choices carry over from the AR side.
- **Not a drop-in retrofit.** You can't take a pure AR checkpoint and add diffusion mode cheaply — the joint objective needs to be trained from (near) scratch, or via expensive continued pretraining.
- **Distinct from MTP.** MTP is an AR training objective with a small side head predicting future tokens. Joint AR+diffusion trains the *whole model* to operate in either mode. MTP is a specialization; this is a broader unification.

## Sources

- Paper: *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding* — Whalen, Garg, Wu, et al., NVIDIA, 2026 — [arXiv:2607.05722](https://arxiv.org/abs/2607.05722).
- Related: *SEDD / Score Entropy Discrete Diffusion* — Lou et al., 2024 — pure-diffusion LM baseline.
- Related: *LLaDA* — Nie et al., 2025 — large-scale mask-diffusion LM.

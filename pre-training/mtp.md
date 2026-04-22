# Multi-Token Prediction (MTP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train the model to predict not just the next token but the next **k** tokens at each position, via one or more lightweight auxiliary "MTP modules" on top of the main backbone. Gives a denser training signal (more loss per position), a modest downstream quality lift, and — kept at inference — enables **speculative decoding** with an ~1.8× speedup. DeepSeek-V3 uses depth-1 MTP (predicts one extra token beyond next).

**Prereqs:** [transformer-block](../architectures/transformer-block.md)
**Related:** none

---

## What it is

The standard language-model objective predicts token `t+1` at position `t`:

```
L_main = -Σ_t  log P(x_{t+1} | x_{≤t})
```

MTP adds additional prediction heads that, at position `t`, predict tokens `t+2`, `t+3`, …, `t+k` — each through its own small transformer module sitting on top of the main backbone. The extra losses are summed with a coefficient into the training objective; the extra modules can be discarded at inference for the base model, or kept for **speculative decoding**.

Two motivations:
1. **Denser supervision per position.** One forward pass of the backbone produces `k` predictions instead of 1, so each token contributes more to the loss. The backbone's hidden state is forced to encode enough structure for a downstream module to predict two tokens ahead — a stronger representation requirement.
2. **Free speculative decoding.** If you keep the MTP head at inference, you have a cheap "draft" for the next token that the main model can verify in parallel, cutting decoding steps roughly in half.

---

## How it works

### Depth-D MTP architecture (D = prediction depth)

```
                main backbone
                 (L layers)
                      │
                      ▼
              h_t^main = backbone(x_{≤t})
                      │
                      ├─► LM head  ──► P(x_{t+1}|...)       ← main loss
                      │
                      ▼
                  MTP module 1
          (takes h_t^main + embed(x_{t+1}))
                      │
                      ▼
              h_t^{MTP,1}
                      │
                      ├─► LM head  ──► P(x_{t+2}|...)       ← MTP loss 1
                      │
                      ▼
                  MTP module 2                 (omitted in depth-1)
          (takes h_t^{MTP,1} + embed(x_{t+2}))
                      │
                      ...
```

DeepSeek-V3 uses **D = 1**: a single MTP module predicting `x_{t+2}`. The module is a single transformer block.

### The MTP module

Inputs: the previous step's hidden state `h_t^{MTP,k-1}` (or `h_t^main` for the first module) and the embedding of `x_{t+k}`. These are combined via RMSNorms and a learned linear map, then passed through a transformer block:

```
h_t' = W_m · [ RMSNorm(h_t^{MTP,k-1}) ; RMSNorm(Embed(x_{t+k})) ]
h_t^{MTP,k} = TransformerBlock(h_t')
P(x_{t+k+1}|...) = softmax( W_head · h_t^{MTP,k} )
```

Crucially, **the embedding layer `Embed` and the output head `W_head` are shared with the main model.** Only the fusion matrix `W_m` and the transformer block's parameters are MTP-specific. This keeps the parameter overhead small.

### The loss

Sum the main and MTP losses with a coefficient `λ`:

```
L = L_main + λ · (1/D) Σ_{k=1..D}  L_{MTP,k}
```

In DeepSeek-V3:
- `λ = 0.3` for the first **10T tokens** of training
- `λ = 0.1` for the final **4.8T tokens**

The decay keeps MTP influence strong early (where the denser supervision helps representation learning most) and tapers it at the end so the final checkpoint is optimized primarily for the real next-token task.

### Causal masking correctness

To predict `x_{t+2}` from position `t`, the MTP module takes `h_t^main` *plus* `Embed(x_{t+2-1}) = Embed(x_{t+1})`. The `x_{t+1}` is not "future information" — at training time the entire sequence is known, and the module is only asked to predict the *next* token in its own stream. Just make sure the causal mask applies correctly so the main backbone at position `t` never sees `x_{>t}`.

### Inference: two modes

**Mode A — discard MTP module.** Standard autoregressive decoding. No cost, but you still benefit from the stronger backbone representation that training under MTP produced (confirmed in DeepSeek-V3 ablations: keeping the main loss only but removing the MTP objective during training costs ~0.5 points on average benchmarks).

**Mode B — speculative decoding.** Use the MTP head to *draft* token `t+2` cheaply, then let the main model verify both `t+1` (normal) and `t+2` (check). If the main model's prediction at `t+2` matches the draft, accept both; else accept only `t+1` and re-decode `t+2` normally.

DeepSeek-V3 reports **85–90% acceptance rate** for the second token, giving an effective speedup of **~1.8× tokens per second** on their serving stack.

---

## Why it matters

- **Stronger representations.** The backbone's hidden state at position `t` must encode enough signal for a one-block module to predict two tokens out. This is a regularizer toward more "global" representations, empirically worth 0.3–0.6 points on mixed benchmarks.
- **~1.8× inference throughput** with self-speculative decoding — no external draft model, no separate training pipeline, the draft is a byproduct of pretraining.
- **Minimal parameter overhead.** One transformer block per MTP level (for depth-1, one extra block out of 61 = ~1.6% params) plus the small fusion matrix. Negligible.
- **Natural curriculum.** Early training benefits more from dense supervision (`λ = 0.3`), late training from focus on the real objective (`λ = 0.1`). The schedule matters.

---

## Gotchas & tricks

- **Share embeddings and LM head.** Don't give the MTP module its own embedding or head. You'll double parameters without a quality gain, and the shared head is what makes speculative decoding well-calibrated (draft logits are on the same scale as main logits).
- **Fusion matrix size.** The `W_m` that combines `h_t^{MTP,k-1}` and `Embed(x_{t+k})` needs to project `2·d_model` down to `d_model`. Keep it simple — a single linear, no activation.
- **Depth-1 is the sweet spot.** DeepSeek-V3 stopped at D=1. Earlier work (Gloeckle et al. 2024) tried up to D=4 and found diminishing returns, plus training overhead grows linearly in D.
- **Decay `λ` late in training.** The final 4.8T tokens with `λ = 0.1` matters — dropping `λ` to 0 too early loses the representation regularization; keeping it high to the end leaves the main objective under-optimized.
- **Speculative decoding requires compatible inference infra.** The main model must accept a batch of `1 + D` candidate tokens for verification in a single forward pass. Standard decoding kernels don't support this out of the box; DeepSeek released serving code that does.
- **Don't confuse with "predict multiple tokens" from seq2seq literature.** Those are usually decoder-side beam search variants; MTP is a *training* objective with *optional* inference use.
- **MTP modules see the main model's *final* hidden state.** Not intermediate layer states. This keeps the MTP prediction close to the model's own best representation.
- **Not the same as parallel decoding / Medusa.** Medusa trains *multiple* prediction heads off a frozen base with a lightweight tuning phase. MTP trains the heads and the backbone together from scratch. Medusa is a post-hoc add-on; MTP is a pretraining objective.

---

## Sources

- Paper: *Better & Faster Large Language Models via Multi-token Prediction* — Gloeckle et al., Meta, 2024 — introduces the multi-token-prediction objective with ablations at up to D=4.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — depth-1 MTP at 671B scale with the `λ` schedule and speculative-decoding integration.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024 — the post-hoc multi-head alternative.

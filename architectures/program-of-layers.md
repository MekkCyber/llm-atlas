# Program-of-Layers (PoLar)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Treat pretrained transformer layers as *callable modules* and learn a per-input **execution program** over them — skip some, loop others, run the rest in order. A lightweight predictor network takes the input embedding and emits the skip/loop schedule; the base model weights stay frozen. For most inputs, a shorter program matches or beats the full forward pass, and a different program can correct predictions the original forward pass got wrong.

**Prereqs:** [transformer-block.md](transformer-block.md), [_normalization.md](_normalization.md)
**Related:** [../inference/README.md](../inference/README.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Standard transformer inference is rigid: every input goes through every layer once, in order. The PoLar paper's empirical observation: for the same pretrained model, alternative layer programs (subsets of layers possibly with repetitions) exist that

- match the original forward pass on most inputs while executing fewer layers, **and**
- correct some inputs the original forward pass got wrong.

Together these results mean fixed-depth execution is leaving capacity on the table. PoLar formalizes the discovery as a learnable per-input dispatcher.

---

## How it works

### Programs over a frozen layer pool

The base model's $L$ pretrained layers $\{f_1, \ldots, f_L\}$ act as a typed module pool: each layer maps residual-stream state to residual-stream state. A *program* is a sequence $(f_{i_1}, f_{i_2}, \ldots, f_{i_T})$ chosen from these layers with skip / loop allowed.

### Predictor network

A small predictor network takes the input embedding (or an early hidden state) and outputs the program. The paper uses a lightweight head — orders of magnitude smaller than the base model — emitting per-block decisions:

- **pass** — execute this block once (default).
- **skip** — bypass; residual stream is unchanged.
- **loop $k$** — execute the same block $k$ times in succession.

The predictor is trained on a task signal (e.g. math reasoning accuracy) with the base model frozen.

### Identity and stability

For training-free discovery, the paper shows valid programs are surprisingly common even with random search over the {skip, pass, loop} program space — pretrained layers admit non-identity execution orders without collapse.

The predictor stabilizes the discovery: rather than searching per input at deploy time, the predictor learns to map inputs to good programs offline.

---

## Why it matters

- **Adaptive compute.** Inputs that need less reasoning execute fewer layers; harder inputs trigger loops on specific blocks. Serving cost matches input difficulty.
- **Out-of-distribution robustness.** The gains hold on OOD inputs in the paper's eval, suggesting the predictor learns a transferable signal rather than memorizing patterns.
- **Free fixed-budget alternative to test-time scaling.** Where long-CoT RL spends compute on longer outputs, PoLar spends compute on adaptive depth — both buy capability for inputs that need it.
- **Interpretability surface.** Which programs the predictor picks for which inputs is a new diagnostic for what the model is doing.

---

## Gotchas & tricks

- **Layer ordering vs. layer set.** Programs that reorder layers (rather than just skip/loop) can break: residual stream geometry is calibrated to a specific ordering during pretraining. PoLar's skip/loop space is conservative on purpose.
- **Looping deep layers is risky.** Late layers are calibrated to a near-final residual-stream norm; looping them can drift output distributions. The paper restricts loops to mid-stack blocks.
- **Pre-norm vs. post-norm matters.** Pre-norm transformers (most modern stacks) tolerate skips and short loops more gracefully than post-norm.
- **Predictor must condition on a *fast* signal.** Using a late-layer hidden state to choose the program defeats the speedup. Restrict to input embedding / early layer states.
- **Training the predictor without RL is preferred.** RL'ing the predictor against a reasoning reward works but is expensive; a supervised proxy (program correctness on a held-out task) is cheaper and reported to suffice.
- **Mixed-precision and KV caches.** Skipping a block changes the KV-cache layout for that position; integrating with paged-attention serving stacks needs care.

---

## Sources

- Paper: *Skip a Layer or Loop It? Learning Program-of-Layers in LLMs* — 2026 — [arXiv 2606.06574](https://arxiv.org/abs/2606.06574).
- Background: early-exit / adaptive-depth literature (DeeBERT, MoD, MoE depth gates) for related but coarser dynamic-depth designs.

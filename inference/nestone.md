# NestOne
*Depth — nested-depth joint training for adaptive early-exit decision LLMs.*

**TL;DR:** "System One" decision LLMs (answer with few or zero generated tokens) usually run every layer at inference even when a shallower model would suffice. NestOne jointly trains a full model and several shorter prefix models that share the full model's final layers and output head; a lightweight gate reads intermediate hidden states and triggers early exit through the shared head. One checkpoint gives a continuum of depth-cost tradeoffs and matches or beats separately trained fixed-depth baselines at every depth.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

A training + inference recipe for decision LLMs with adaptive depth. The idea has been attempted before (early-exit for BERT, Universal Transformers, various cascade setups), but the usual failure mode is that an "early-exit" head trained post-hoc never matches a model that was trained *as* a shallow model from the start. NestOne's fix is joint training: the full model and its truncations share weights and co-train so that every depth is a competent model, not an interpolation.

## How it works

- **Nested parameter sharing.** Define a sequence of prefix models of increasing depth, all sharing the full model's final layers and output head. Weight-tying the shared head is essential — the shorter model's early layers produce hidden states that the shared head must decode, so the head has to be competent on all of them.
- **Co-training objective.** During training, every mini-batch is scored at every depth (or a sampled subset of depths), and the loss is a weighted sum. Each depth is a first-class objective, not a regularizer.
- **Learned exit gate.** At inference, a small MLP reads each intermediate hidden state and predicts whether the shared head will produce the correct decision from there. The first depth above threshold exits. The gate is trained on held-out data after the main model is frozen.

## Why it matters

Matches dedicated depth-tuned baselines at each depth while serving all of them from one checkpoint. For the System One pattern — one LLM used as a fast decision model inside agent routing, retrieval, planning — the per-query depth is now a knob, not a model-selection problem. Also removes the main empirical objection to early exit, which was that it underperformed in practice.

## Gotchas & tricks

- The shared head is the load-bearing component. Un-tying it (per-depth heads) collapses the gains because the joint training no longer forces representation alignment across depths.
- Gate training needs calibrated exit thresholds per deployment. A gate trained for average-case latency over-exits on hard questions.
- The nested set of depths is a hyperparameter. Too many points slow co-training; too few leave gaps in the Pareto curve.
- Loss weighting across depths matters — weighting all depths equally over-emphasizes the shallow ones (they dominate the sum). The paper tunes per-depth weights so the full-depth model doesn't degrade.
- Serving requires inspecting intermediate activations, which breaks some KV-cache layouts. The gate has to be cheap enough that polling it every layer is negligible.

## Sources

- Paper: *From Large Language Models to Efficient System One Models via Adaptive Computation* — 2026 — [arXiv:2611.16321](https://arxiv.org/abs/2611.16321)

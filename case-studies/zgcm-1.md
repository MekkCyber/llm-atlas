# Case Study: ZGCM-1

*A fully open 7B foundation model at 256K context, positioned for math reasoning and agentic search. The premise: 7B parameters cannot memorize the open web, so the model is trained from the start to **think internally** and **retrieve/act externally** as complementary capabilities rather than an inference-time bolt-on. Architecture couples sliding-window attention with periodic full-attention layers; pretraining stages context length upward through a progressive curriculum.*

**Related concepts:** [mla](../architectures/mla.md) · [rope](../fundamentals/rope.md) · [context-length-curriculum](../pre-training/context-length-curriculum.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [rlvr](../post-training/rlvr.md) · [grpo](../post-training/grpo.md)

---

## What this is

ZGCM-1 is a 7B-parameter fully open foundation model released September 2026 by Microsoft Research Asia. Its stated design premise is that **compact models cannot passively memorize the open web**, but can overcome parametric capacity limits by **coupling deliberate internal thinking with active external tool use**. The paper releases the checkpoint, the training recipe, and the tool-use scaffolding as one package.

The interesting part is not the parameter count but the design commitments that follow from the premise: an architecture chosen for cheap long context, a progressive-context pretraining curriculum, and post-training that treats tool use as a first-class capability rather than a fine-tuning afterthought. The result is claimed competitive with substantially larger models on math reasoning and agentic search — a data point on the tool-use / parametric-capacity Pareto frontier.

---

## Architecture at a glance

```
Base:  7B parameters, decoder-only transformer
Context window: 256K tokens (native)

Attention:
  ├─ sliding-window layers  (window W)
  └─ periodic full-attention layers interleaved for long-range fusion

Positional encoding: RoPE
```

The sliding-window / full-attention interleave is the load-bearing architectural choice: most layers stay cheap at long context because attention reads scale with the window rather than the sequence length, while the periodic full-attention layers refuse to lose long-range signal.

The paper positions the choice as a deliberate alternative to latent-compression variants like [MLA](../architectures/mla.md): both cut the per-token attention cost at long context, but sliding-window bets on locality and full-attention layers for the long tail, while MLA compresses per-token KV without changing reach.

---

## Training recipe

### Progressive-context curriculum

Pretraining runs through a sequence of stages that **expand the context window across the run** — starting with shorter sequences and stepping up to the final 256K. Each stage trains the model with the current maximum length; RoPE frequencies and any window-size configuration are adjusted per stage so the model learns to use the new long-range span rather than treating the extra tokens as padding.

The pattern is documented in the depth file [context-length-curriculum](../pre-training/context-length-curriculum.md). The empirical claim is that this graduated exposure is what actually bakes long-context capability into the base model, rather than length-extrapolation tricks applied after the fact.

### Data mix and tool-use grounding

The paper's premise implies a specific data-mix choice: pretraining not only on text-to-be-memorized but on **traces of internal reasoning + external tool use** — retrievals, calculations, code execution outcomes — so the model learns the tool-use interface at pretraining scale. Exact mix ratios are not extractable from the abstract; the recipe emphasizes that tool use is not left to a post-training bolt-on.

### Post-training

Post-training combines instruction tuning with RL on verifiable rewards over agentic-search and math-reasoning tasks. The [RLVR](../post-training/rlvr.md) / [GRPO](../post-training/grpo.md) primitives are used at trajectory scale so the reward signal covers tool-mediated sequences — the same shape as the "verifiable experience" recipe surfaced in Atria Dawn's tech report, applied here at 7B scale.

---

## Key results

- **Competitive with substantially larger models** on math reasoning and agentic search tasks in the paper's evaluation suite.
- **Fully open** checkpoint at 7B / 256K context — the size class where the tool-use-vs-parametric-capacity trade-off is most contested.
- Positions the sliding-window + full-attention + tool-use pretraining recipe as a reproducible baseline for the community.

Exact benchmark numbers are not fully extractable from the fetched abstract; the paper's own results tables should be consulted for math (competition-math, AIME-class) and agentic-search benchmarks.

---

## Why it matters

- **A reproducible small-model baseline for the "compact reasoner that searches" hypothesis.** The dominant open-source position for 7B-class models has been "small + strong on knowledge" — trained to memorize as much as possible. ZGCM-1 argues the opposite: small models should be trained to **not** memorize, and to reach for tools instead. If the hypothesis holds at scale, this becomes the template for open efficient bases.
- **A concrete example of architecture-following-premise.** Sliding-window + periodic full-attention is not a novel primitive on its own, but its adoption here is deliberate — it exists because 256K context has to be cheap for tool-use pretraining to be feasible.
- **Progressive-context curriculum documented as a first-class technique.** Long-context capability is usually treated as an add-on (position-interpolation, ALiBi, YaRN). Stage-wise pretraining exposure is a distinct approach worth naming in the KG.

---

## Gotchas & open questions

- **Fully-open at frontier scale is unproven here.** The 7B / 256K / open checkpoint is a clean baseline, but the "compact reasoner + tool use" thesis needs to be tested against similarly-sized bases without the tool-use pretraining to isolate the contribution.
- **Sliding-window ratio.** The paper's choice of full-attention layer frequency (e.g. 1-in-4, 1-in-5) directly trades throughput for long-range quality; the ablation on this ratio is where the recipe's cost/quality curve lives.
- **Tool-use benchmark contamination.** Any model trained on agentic-search traces at pretraining scale needs strong decontamination against the benchmarks it will be evaluated on; the paper's decontamination story is the one to read.

---

## Sources

- Paper: *ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search* — Jiyan He, Guang Liang, Yifei Shen, Shuxin Zheng, Tie-Yan Liu, et al. — Microsoft Research Asia — 2026 — [arXiv:2609.13356](https://arxiv.org/abs/2609.13356).
- Background: *Longformer* (Beltagy et al., 2020) and *Mistral 7B* (Jiang et al., 2023) for the sliding-window primitive; *DeepSeek-V3* (2024) for the MLA long-context alternative.

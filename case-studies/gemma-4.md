# Case Study: Gemma 4

*Google DeepMind's fourth-generation open-weight Gemma family. The interesting story is a triple architectural bet on one release: (1) MoE variants pushed into the small-open-model regime alongside dense siblings (2.3B–31B), (2) a 12B model that drops separate vision/audio encoders and ingests **raw image patches + raw audio** directly through a shared front end, and (3) an integrated **thinking mode** so reasoning traces are a first-class output at inference. The tech report positions Gemma 4 as competitive with much-larger frontier open models on human-rated tasks.*

**Related concepts:** [../architectures/_moe.md](../architectures/_moe.md) · [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md) · [encoder-free-multimodal.md](../architectures/encoder-free-multimodal.md) · [../post-training/reasoning/thinking-mode.md](../post-training/reasoning/thinking-mode.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../multimodal/README.md](../multimodal/README.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What this is

**Gemma 4**, released July 2026 by Google DeepMind. arXiv 2607.02770. A model family of **dense and Mixture-of-Experts** open-weight LMs spanning **2.3B to 31B parameters**, natively multimodal (vision + audio + text) across all sizes.

Three headline choices:

1. **Dense + MoE inside the same open-weight family.** Previous Gemma releases were dense-only. Gemma 4 adds MoE variants; the family covers both regimes at open-weight scale so users can pick per compute/quality point.
2. **Encoder-free multimodal at 12B.** Every other Gemma 4 size uses "improved vision and audio encoders." The 12B variant drops those separate encoders entirely and ingests raw image patches and raw audio directly into the LM front end. This is Google's version of the encoder-free thesis (Chameleon, Fuyu) at the largest open-weight scale so far.
3. **Native thinking mode.** All Gemma 4 models integrate a thinking mode: the model emits a reasoning trace before its final answer. Not a post-hoc CoT wrapper — the mode is trained in, and the trace is a first-class output the model conditions its answer on.

Positioning claim: Gemma 4 "establishes a leap" on STEM, multimodal, and long-context benchmarks, and rivals larger, frontier open models in human-rated tasks.

---

## Architecture at a glance

Parameter range and variants:

```
Family footprint (all natively multimodal: text + vision + audio):
  2.3B  — dense
  ...   — dense and/or MoE (exact sizes not enumerated in abstract)
  12B   — dense, ENCODER-FREE multimodal (raw patches + raw audio)
  ...   — dense and/or MoE
  31B   — dense or MoE (flagship)

  Some intermediate sizes are MoE; abstract does not enumerate which.
```

**Modality front end (non-12B sizes).** Improved vision and audio encoders per model size. Encoded features project into the LM's token space (standard cross-modal projection layer pattern).

**Modality front end (12B).** No separate vision/audio encoder. Raw image patches and raw audio are tokenized into the shared input stream and processed by the LM directly. Sits in the encoder-free multimodal lineage (Chameleon, Fuyu) at previously-untried scale for the class.

**Thinking mode.** Integrated at training time. At inference, the model produces a reasoning trace segment before the final answer, conditioned on the same input.

**Long-context.** "Critical design choices" improve long-context ability; specifics not enumerated in the abstract. Likely a mix of extended positional encoding scaling + attention efficiency tricks — details in the full report.

---

## Pre-training

*Pre-training data mix, token count, and training-hardware footprint are not disclosed in the abstract. Details forthcoming from the full report.*

Structural expectations, from the abstract:
- Native multimodality means vision, audio, and text tokens are present throughout pre-training, not bolted on afterward.
- For MoE variants: the abstract does not name a routing algorithm; DeepMind's prior open work leans toward standard top-k routing with load-balancing losses.

---

## Post-training

*The full recipe (SFT + RL stages, data sizes) is not in the abstract. What is stated:*

- **Thinking-mode training.** The model is trained to emit a reasoning trace before the final answer. This is a post-training capability — the model learns *when and how* to generate reasoning versus answering directly. See [../post-training/reasoning/thinking-mode.md](../post-training/reasoning/thinking-mode.md).
- **Reasoning focus.** The report emphasizes compute-efficient reasoning as a design goal. Reasoning-specific RL (GRPO or a variant with verifiable rewards) is the standard 2026 recipe; whether Gemma 4 uses it exactly is TBD from the abstract.

---

## Key results

*Numeric benchmark tables are not extracted from the abstract. What is stated at claim-level:*

- **STEM benchmarks:** "leap in performance" over prior Gemma generation.
- **Multimodal benchmarks:** same claim.
- **Long-context benchmarks:** same claim.
- **Human-rated tasks:** rivals larger, frontier open models.

Concrete numbers will be filled in when the full report is available.

---

## What's interesting

1. **Encoder-free multimodal at 12B is a serious bet.** Every other size in the family uses separate encoders; the 12B stands alone dropping them. If this variant matches or beats its encoder-equipped siblings, DeepMind is quietly saying the encoder-less route is the correct direction and the encoder pattern is transitional infrastructure. If it *doesn't* match, they can point at 6 other model sizes that use encoders. Low-risk-high-signal design.

2. **MoE inside a "small open" family.** MoE in open weights has largely been the domain of Mixtral, DeepSeek, Qwen. Gemma landing MoE variants at 2.3B–31B scale — where compute constraints hit hardest — is a distributional shift toward MoE-by-default in this size range.

3. **Native thinking mode ≠ post-hoc CoT wrapper.** The mode is trained in and the trace is a first-class output. This aligns with the o1 / R1 / Kimi-k1.5 lineage but the abstract emphasizes *integration* — thinking mode is on/off, not a separate model variant.

4. **Three architectural bets in one release.** Encoder-free + MoE + thinking mode are three independent axes each of which is a research paper's worth of engineering. Landing all three in one open release is aggressive and forces users to disentangle which bet drove which improvement.

---

## What's opaque (from the abstract alone)

- **Which sizes are dense and which are MoE.** Only the 2.3B–31B range and the "dense + MoE" statement are given.
- **MoE routing algorithm and expert counts.**
- **Pre-training data volume, mix, and infrastructure.**
- **Post-training recipe details** (SFT scale, RL algorithm, reward setup).
- **Thinking-mode training data and any budget/gating on reasoning traces at inference.**
- **Long-context specifics** (target context length, positional encoding scheme, attention modifications).
- **Vision/audio encoder architectures** for non-12B sizes.
- **Human-eval methodology** ("rivals larger, frontier open models" — vs which models, on which tasks).

---

## Key takeaways

1. **Encoder-free multimodal moves up the size ladder.** Fuyu / Chameleon showed it works at small scale; Gemma-4-12B is the largest open-weight encoder-free multimodal model to date. The story of vision as a separate artifact from the LM continues to erode.

2. **MoE is arriving in the small-open-model regime.** Below the frontier-flagship band, MoE variants alongside dense siblings gives users a compute-efficient path without going to closed-model scale.

3. **Thinking mode as a first-class output** is now the standard for reasoning-capable open models (o1 / R1 / Kimi-k1.5 / Gemma 4). The design pattern has consolidated: emit reasoning trace, condition final answer on it, expose the trace to the user.

4. **Three axes at once is an unusual release strategy.** Most tech reports lead on one architectural change; Gemma 4 stacks three. Expect follow-up papers to isolate the contribution of each.

---

*Pairs well with:* [qwen2-5.md](./qwen2-5.md) for contrast on model-family scaling with a fixed dense recipe, and [deepseek-v3.md](./deepseek-v3.md) for the frontier-scale MoE + systems-innovations lineage that Gemma 4's MoE variants inherit some conventions from.

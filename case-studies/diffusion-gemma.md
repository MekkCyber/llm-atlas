# Case Study: DiffusionGemma

*A 25.2B-total / 3.8B-active MoE text diffusion LM from Google DeepMind. Fine-tuned from Gemma 4 with under 10% of the AR model's pretraining budget, it decodes ~1,500 tokens/sec on a single H100 — establishing a new speed × capability Pareto frontier for open-weight LLMs.*

**Related concepts:** [text-diffusion-lm](../architectures/text-diffusion-lm.md) · [sampler-distillation](../post-training/sampler-distillation.md) · [_moe](../architectures/_moe.md) · [mtp](../pre-training/mtp.md) · [rlvr](../post-training/rlvr.md) · [on-policy-distillation](../post-training/on-policy-distillation.md)

---

## What this is

**DiffusionGemma**, released July 2026 by Google DeepMind (43-author "DiffusionGemma Team"). An experimental **open-weight discrete text diffusion** language model built by fine-tuning the mixture-of-experts **Gemma 4** backbone (25.2B total / 3.8B activated). Its central claim: at competitive quality it produces **~1,500 output tokens/sec on a single H100** — substantially faster than autoregressive Gemma of comparable size *even with state-of-the-art speculative decoding*.

Why it matters as a case study, beyond throughput: it is the first serious open-weight text-diffusion release from a frontier lab, and it validates a two-stage recipe — SFT for bidirectional denoising, then RL + sampler distillation — for converting a pretrained AR MoE into a diffusion LM at *under 10% of the original pretraining token budget*.

---

## Architecture at a glance

```
Base backbone         Gemma 4 MoE
Total parameters      25.2B
Activated per token   3.8B
Decoding mode         Discrete diffusion over blocks
Block size            256 tokens
Effective tokens/step ~20 (average)
Retained capabilities thinking mode, multimodal inputs, long context
Bonus                 still capable of AR decoding with minor degradation
```

The model is *architecturally* still Gemma 4 — DeepMind did not redesign the transformer. What changed is the objective, the training data shape (fully-masked → progressively-revealed blocks), and the sampler at inference. Bidirectional attention is enabled within a diffusion block; blocks themselves are decoded left-to-right, so long-context and streaming semantics survive.

See [text-diffusion-lm](../architectures/text-diffusion-lm.md) for the class-level mechanics.

---

## Training recipe

### Stage 1 — Bidirectional denoising SFT

- Take fully-trained Gemma 4 MoE weights as initialisation.
- Adapt the model to a **discrete diffusion objective** over 256-token blocks: sample a random masking ratio $r \in [0, 1]$, mask that fraction of tokens in a block, train the model to denoise them jointly using bidirectional attention within the block.
- Loss: masked cross-entropy over the corrupted positions, weighted to emphasise higher-noise ratios (the harder cases).
- Supervised on the same broad corpus mix as Gemma 4 fine-tuning; no new pretraining data.

### Stage 2 — RL + sampler distillation (jointly)

Two objectives applied together on the Stage-1 model:

- **Reinforcement learning for quality.** Standard preference / verifiable-reward RL applied over diffusion-sampled outputs. Rewards target answer correctness on reasoning tasks and preference wins on open-ended tasks.
- **Sampler distillation for speed.** Distil a many-step diffusion sampler into a few-step sampler by training the student sampler to match a teacher sampler's block-output distribution while using fewer denoising iterations. This is the piece that drives the ~20-tokens-per-forward-pass average and lets DiffusionGemma cross the speed × quality Pareto frontier.

The joint objective explicitly balances *don't lose quality* (RL) and *use fewer steps* (distillation). See [sampler-distillation](../post-training/sampler-distillation.md).

### Compute

Under **10% of the AR Gemma 4's pretraining token budget** — the entire diffusion training is a fine-tuning-scale run. The specific token count is not published in the abstract; the "<10%" figure is the paper's headline compute claim.

---

## Inference: parallel block refinement

```
prompt  ──▶  [block of 256 masked positions]  ──▶  ~T diffusion steps  ──▶  next block
                       ▲                                      │
                       └────── previous decoded blocks ───────┘
```

Each block:
1. Initialise 256 positions as `<mask>`.
2. Run the model forward with bidirectional attention over the block, conditioned on the left-context of already-decoded blocks.
3. Sample or argmax a subset of masked positions per step; commit the confident ones, keep the rest masked.
4. Repeat for a small number of denoising steps until the block is fully decoded.
5. Move to the next block.

Averaged across the evaluation suite: **~20 committed tokens per forward pass**, **~1,500 output tokens/sec on one H100**. Compares favourably to AR Gemma even with state-of-the-art speculative decoding (paper's baseline).

The model also **remains AR-capable** — a diffusion checkpoint can decode left-to-right with only "minor" quality degradation. This enables **hybrid diffusion-AR decoding**: use fast diffusion for bulk generation, fall back to AR for the last few high-precision tokens or for streaming interactions.

---

## Retained capabilities

Diffusion fine-tuning did not require sacrificing:

- **Thinking mode.** The model still supports Gemma 4's extended-thinking format.
- **Multimodal inputs.** Image/text inputs still work at the encoder side; only the decoder is diffused.
- **Long context.** Block-wise decoding streams naturally; the model preserves Gemma 4's context window.
- **AR compatibility.** Same weights can run AR-mode with small quality loss.

This is genuinely surprising — prior text-diffusion work generally required either training from scratch or losing significant capability. Two design choices explain it: keeping the transformer architecture and initialisation intact, and preserving bidirectional attention only within blocks (not globally).

---

## Key takeaways

1. **Text diffusion is a live open-weight option now.** DeepMind is not the first to try discrete text diffusion (SEDD, MDLM, Coconut all pre-date it), but is the first frontier lab to publish an open-weight model where quality holds up and throughput is a real win.

2. **AR → diffusion adaptation works.** Under 10% of the original training budget is enough to convert an AR MoE into a competitive diffusion LM. The "train text diffusion from scratch" cost story is over.

3. **Sampler distillation is what makes it fast.** The joint RL + distillation stage is where the ~20 tokens/step average comes from. Without it, block diffusion is slow because it needs many denoising steps to reach quality.

4. **Block size 256, not full-sequence.** Full-sequence diffusion loses left-to-right causality (bad for streaming, bad for long-context KV reuse). Block-diffusion keeps both while paying the parallelism dividend inside each block.

5. **Hybrid diffusion-AR decoding is on the table.** Since the model retains AR capability, you can mix decoding modes per-request or per-token — fast bulk generation with AR fallback for high-precision spans.

6. **MoE + diffusion composes.** Fine-grained MoE routing works fine over diffused blocks; the routing is per-token per-position, orthogonal to the denoising objective.

---

## What's still opaque

- **Concrete benchmark scores** vs Gemma 4 AR baseline — the abstract references an "evaluation suite" but per-benchmark numbers aren't in the intro summary.
- **Number of denoising steps per block** — averaged effective count is ~20 tokens/step (so ~13 steps per 256-token block), but the schedule is not detailed.
- **RL reward composition** — how much rule-based vs preference-based, and what benchmarks the RL targeted.
- **Sampler-distillation teacher** — whether it's the Stage-1 model, a longer-step diffusion of it, or Gemma 4 AR itself.
- **Multimodal quality** — retained "in principle," but not benchmarked in the abstract.
- **Training cost dollars.** "<10% of AR budget" is a ratio, not a number.

---

*Pairs well with:* the DeepSeek-V3 case study for contrast — V3 shows the frontier AR-MoE recipe (MLA + fine-grained MoE + FP8 + DualPipe); DiffusionGemma shows what happens when you take an AR MoE and switch the objective. Same architectural substrate, radically different inference-time behaviour.

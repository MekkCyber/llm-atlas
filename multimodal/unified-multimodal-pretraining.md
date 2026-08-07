# Unified Multimodal Pretraining

*Depth — the "early unification" recipe for training a single model on text + vision + generation from step 0, and the "vision laziness" failure of late-integration alternatives.*

**TL;DR:** Meta FAIR + Oxford (Tong et al., 2026) run a controlled physics-style sweep over unified multimodal pretraining and distill four findings: (i) knowledge flows asymmetrically between language, visual-understanding, and visual-generation; (ii) shared attention + normalization with modality-specific FFN layers is the architectural sweet spot for synergy; (iii) unifying modalities from the very early stages beats late alignment or sequential training — delayed integration produces **"vision laziness"** where the model relies on language priors; (iv) efficient recipes hit strong generation performance at **5% of the compute budget** of naive baselines. Validated at scale with 13.5B MoE models on 2T tokens.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../pre-training/README.md](../pre-training/README.md)
**Related:** [../pre-training/mid-training.md](../pre-training/mid-training.md) · [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md) · [README.md](README.md)

---

## What it is

The choice of *when* and *how* to combine vision and language during pretraining. Three canonical alternatives:

- **Late alignment.** Pretrain a language model, pretrain a vision encoder separately, then align them with a small projection layer + adapter tuning (LLaVA-style).
- **Sequential.** Text pretrain first, add vision tokens and continue training.
- **Early unification.** From the very first optimizer step, the model sees interleaved vision and language tokens and jointly predicts both. This paper's recommended default.

## How it works — the four findings

1. **Knowledge flow (asymmetric transfer).** Language capability transfers into visual generation more strongly than into visual understanding. Visual understanding transfers into generation, but not vice versa. These asymmetries define which modality benefits from joint training and which is left underserved by naive mixing.
2. **Synergy vs. competition (architecture matters).** Shared attention and normalization layers across modalities *plus* modality-specific FFN layers is the sweet spot. Fully-shared FFN → competition; fully-separate attention → no synergy. Findings hold across visual tokenizer choices, so the pattern is architectural, not tokenizer-specific.
3. **Early unification (avoid vision laziness).** Sequential training produces a model that has already learned to solve tasks using language alone, and then treats vision as auxiliary — "vision laziness." Joint from step 0 forces the model to learn cross-modal representations before language shortcuts entrench.
4. **Efficient recipes.** With early unification + shared-attn + modality-FFN + calibrated data complexity, strong generative multimodal performance is achievable at **~5% of naive-recipe compute**.

Validated at scale: multiple 13.5B **MoE** models trained on **2T tokens** hold to the recipes derived on smaller sweeps.

## Why it matters

- **First mechanism-level design guide for unified multimodal pretraining.** Prior work (Chameleon, Show-o, Janus) was largely trial-and-error; this paper isolates the actual axes.
- **Data complexity as the synergy-vs-competition dial.** The paper's controlled experiments show *data complexity* — not just modality mixture ratio — drives whether modalities synergize. That's a design axis prior recipes underweighted.
- **20× compute savings** if the recipes hold in your setting. Real budget freed for scale or data.
- **Architectural recommendation is portable.** Shared-attn + shared-norm + modality-FFN is a small change to most existing transformer implementations.

## Gotchas & tricks

- **Vision laziness is silent.** Late-integration models can score fine on shallow benchmarks and still ignore image evidence when it contradicts language priors — spot-check on adversarial image-text tasks.
- **Shared-attn requires care with attention masks** for interleaved modality tokens; positional encodings and modality-typed embeddings become non-optional.
- **Data complexity is fuzzy.** The paper uses their own operationalization — porting the recipe to a new dataset requires re-measuring "complexity" per their definition.
- **MoE validation is at 13.5B active.** Whether the recipes hold at 100B+ active is untested here.
- **Efficient recipes assume the paper's data mix.** 5% compute savings are relative to a naive baseline in a comparable data regime — not a promise across arbitrary pretraining setups.

## Sources

- Paper: *Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes* — Tong, Fan, Chen, Torr, Kokkinos, Lewis, 2026 — [arXiv 2608.05000](https://arxiv.org/abs/2608.05000). FAIR at Meta, Reality Labs, University of Oxford.

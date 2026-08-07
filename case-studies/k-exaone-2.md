# Case Study: K-EXAONE 2.0

*A 750B-total / 37B-active MoE built by upcycling K-EXAONE — three times the active capacity of its parent — with a distinctive hybrid local/global attention stack, SWA-only RoPE for 256K context, and a two-stage GrouPER preference-optimization pipeline. Released open-weight under Apache 2.0 by LG AI Research, 2026.*

**Related concepts:** [moe-upcycling](../architectures/moe-upcycling.md) · [swa-only-rope](../architectures/swa-only-rope.md) · [_moe](../architectures/_moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [mid-training](../pre-training/mid-training.md) · [grouper](../post-training/grouper.md) · [_post-training](../post-training/_post-training.md) · [dpo](../post-training/dpo.md) · [safety-case](../safety/safety-case.md)

---

## What this is

K-EXAONE 2.0, released August 2026 by LG AI Research. A Mixture-of-Experts decoder transformer: **750B total parameters, ~37B activated per token**, with a hybrid local + global attention stack expanded from the earlier K-EXAONE via structured *upcycling* rather than from-scratch training. Long context: **256K tokens**. Multilingual: **10 languages** (up from 6), with distinctive strength on Korean. Apache 2.0 license.

The paper matters for three separable contributions:

1. **MoE upcycling with rotation-noise symmetry breaking** — a norm-preserving way to duplicate experts without them staying tied.
2. **SWA-only RoPE** — apply RoPE only to sliding-window layers, omit it from global-attention layers, for cheap staged long-context extension.
3. **GrouPER (Group-wise SimPER)** — reference-free preference optimization over ranked response groups, applied in two sequential stages (multi-task then safety).

Any one of these would be a standalone paper; K-EXAONE 2.0 bundles them into a validated frontier-scale system.

---

## Architecture at a glance

```
78 transformer layers = 19 × LLLG block
  LLLG block:  L = local sliding-window attention
               L
               L
               G = global full attention

RoPE:            local L layers only  (SWA-only RoPE)
                 global G layers have NO positional encoding

MoE:             256 routed experts per MoE layer   (up from 128)
                 ~37B activated / 750B total        (up from ~12B active parent)
                 routing:  inherited from K-EXAONE (bias-based balancing)

Tokenizer:       inherited unchanged from K-EXAONE
Context length:  256K tokens (staged: 8K → 64K → 256K)
```

The parent model was 48 layers (12 LLLG blocks) with 128 experts. K-EXAONE 2.0 does depth expansion (12 → 19 blocks by repeating middle-of-stack blocks) *and* width expansion (each expert duplicated, 128 → 256) simultaneously.

See [moe-upcycling](../architectures/moe-upcycling.md) for the block-repeat + rotation-noise expert-duplication mechanics, and [swa-only-rope](../architectures/swa-only-rope.md) for why the global layers have no positional encoding.

---

## Training pipeline

The three-phase pipeline is: **continual pre-training → difficulty-focused mid-training (2 stages) → post-training (SFT + 2 GrouPER stages)**.

### Continual pre-training

- **Init:** upcycled from K-EXAONE. Experts duplicated with norm-preserving random rotation applied to the copy (breaks symmetry without disturbing activation statistics). Router biases and tokenizer inherited unchanged; load stays balanced across the new 256-expert set with no intervention beyond the inherited bias-update schedule.
- **Continued training:** **8T tokens** on the K-EXAONE 2.0 pre-training data mixture after a short stabilization phase.

### Difficulty-focused mid-training (two stages)

- **Mid-Stage 1:** 400B tokens at 64K context. Data emphasizes long-form reasoning trajectories, cross-file dependencies in code repositories, and multi-step tool-use workflows. Internal classifiers are retrained to identify STEM corpora containing advanced knowledge and reasoning signals; additional challenging knowledge data is constructed with an internal search agent.
- **Mid-Stage 2:** 400B tokens at contexts exceeding 64K, extending up to 256K. Continues the reasoning / repo-level code / agentic-workflows emphasis at higher context lengths.

The mid-training phase is where the "difficulty-focused" label lives: data is filtered to be harder than typical web mix, and the model is trained at the full target contexts rather than short then extended.

### Long-context recipe

Staged extension **8K → 64K → 256K** across the two mid-training stages. Long-context data is diverse end-to-end corpora with minimal truncation: complete documents, large code repositories, tool-use trajectories, and synthetically-constructed multi-hop data requiring retrieval and synthesis across distant context positions.

Because RoPE lives only on local (sliding-window) layers, only local-layer RoPE frequencies need rescaling between stages; global layers require no positional reconfiguration. Verified by NIAH: perfect retrieval scores across evaluated needle positions up to 256K.

### Post-training

**SFT:** 350B tokens across target domains, with **router parameters frozen**. Jointly trains "thinking" and "non-thinking" modes (an inference-time switch, similar to Qwen3 / Kimi-K1.5 style dual-mode outputs).

**Preference optimization (two GrouPER stages):**

- **Stage 1 — multi-task GrouPER:** groups mix reasoning, agentic, and chat prompts. Rewards are stage-specific per prompt:
  - Math/code: verifiable signals (unit tests, answer match) + LLM-as-a-judge
  - Chat: instance-specific rubrics
  - Agentic: correctness of actions + response quality, depth, comprehensiveness
- **Stage 2 — safety-aware GrouPER:** groups are safety-oriented prompts scored under domain-specific safety reward criteria drawn from the K-AUT-V2 taxonomy.

See [grouper](../post-training/grouper.md) for the group-wise SimPER mechanics.

---

## Korean sociocultural safety grounding

A distinguishing feature relative to peer tech reports. Safety data and evals are grounded in Korean institutional sources and cultural context:

- **Data:** high-quality Korean materials from K-DATA, NIA, National Institute of Korean Language, and Northeast Asian History Foundation.
- **Taxonomy K-AUT-V2:** expanded from 226 → **296 risk areas** across four domains — Universal Human Values (69), Social Safety (89), Korean Sensitivity (87), Future Risk (51). Coverage extends to geopolitical / historical risks (North Korea, constitutional order, historical revisionism).
- **Expert loop:** four-week closed loop with a Safety Teacher Advisory Council of 46 UNESCO-Global-Citizenship-trained teachers doing red-teaming and criterion revision.
- **Evals:** KGC-Safety (in-house) and ROK-Fortress.

**Headline safety numbers:** 99.8 on KGC-Safety, 89.5 on ROK-Fortress. This is where the paper carves out a clear differentiation from other frontier open-weight releases.

---

## Key results (reasoning mode)

| Category | Benchmark | K-EXAONE 2.0 |
| --- | --- | --- |
| World knowledge | MMLU-Pro | 83.5 |
| | GPQA-Diamond | 82.2 |
| Math | AIME 2026 | 92.3 |
| Coding | SWE-Bench Verified | 68.2 |
| | Terminal-Bench 2.1 | 43.8 |
| Long context | OpenAI-MRCR | 94.4 |
| Korean | KMMLU-Pro | 69.1 |
| | CLIcK | 84.2 |
| Multilingual | MMMLU | 86.6 |
| Safety | KGC-Safety | 99.8 |

**Largest gains vs. K-EXAONE (parent):** +18.8 pts SWE-Bench Verified, +13.5 pts Terminal-Bench 2.1, +42.1 pts OpenAI-MRCR. The agentic-coding and long-context deltas dominate.

---

## Why this matters

- **Third open-weight tech report at the ~750B / ~37B-active MoE scale** after DeepSeek-V3 (671B/37B) and Kimi-K2-style releases — this scale point is becoming the dominant open-frontier design regime.
- **Upcycling as an alternative to from-scratch.** The K-EXAONE → K-EXAONE 2.0 path shows a plausible cheaper trajectory for organizations with an existing model: expand and continued-train rather than restart.
- **Long-context via architecture, not tricks.** Most 256K-context papers rely on RoPE frequency manipulation or YaRN-style extensions; SWA-only RoPE removes the need entirely for the global layers.
- **Preference optimization is settling on group-wise objectives.** GrouPER, GRPO, and their family are converging: the paired DPO/SFT era is giving way to group-wise reward-aggregation as the default.
- **Regional sociocultural safety framework** as a first-class deliverable. Beyond generic safety benchmarks, K-AUT-V2 shows how a model provider serving a specific region can operationalize governance in the training pipeline.
- **Not the endpoint.** The paper explicitly frames 750B/37B as a milestone en route to global frontier scale — expect a subsequent K-EXAONE at larger active parameters.

---

## Related depth pages (new with this case study)

- [moe-upcycling](../architectures/moe-upcycling.md) — block-repeat depth expansion + expert duplication + rotation-noise symmetry breaking.
- [swa-only-rope](../architectures/swa-only-rope.md) — RoPE on local layers only, global layers position-encoding-free, for cheap 256K staged extension.
- [grouper](../post-training/grouper.md) — group-wise SimPER for reference-free preference optimization, used in two stages (multi-task + safety).

---

## Sources

- Paper: *K-EXAONE 2.0 Technical Report — Journey to Global Frontier-Scale Foundation Models* — LG AI Research, 2026 — [arXiv 2608.04505](https://arxiv.org/abs/2608.04505).
- License: Apache 2.0. Weights released open.
- Predecessor: K-EXAONE (LG AI Research, 2024) — the parent model that was upcycled.

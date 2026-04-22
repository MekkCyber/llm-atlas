# Mid-Training (Late-Stage High-Quality Mix)
*Depth — a final pre-training phase on a small, curated mix with annealed learning rate.*

**TL;DR:** After the bulk of pre-training on a general web-scale corpus, switch to a **smaller, much higher-quality data mix** (math, code, filtered web, instruction-like text) and **anneal the learning rate toward zero** over the last 50–200B tokens. Downstream benchmark scores (math, code, reasoning) are dominated by this phase, not the prior trillions of tokens. Different labs call it "mid-training", "annealing phase", "Stage 2", or just "Dolmino" / "data cooldown" — it's the same idea.

**Prereqs:** [transformer-block](../architectures/transformer-block.md)
**Related:** [model-souping](model-souping.md), [wsd-schedule](wsd-schedule.md), [_lr-schedules](_lr-schedules.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

A structured two-phase pre-training curriculum:

| Phase | Data | Tokens | LR schedule |
|---|---|---|---|
| **Stage 1** | Broad, web-dominated, lightly filtered | 4–15T | Warmup → cosine to some plateau |
| **Stage 2 (mid-training)** | Small, heavily curated: filtered web, math, code, academic, synthetic, instruction-like | 50–300B | Anneal from plateau LR to near-zero |

The model that comes out of Stage 1 has **broad world knowledge** but weak benchmark performance. Stage 2 does not teach it new facts at scale — it **sharpens the distribution toward high-quality patterns** while the LR decays, locking in what it sees.

## How it works

### Why a second phase rather than just a better mix throughout

High-quality data is **expensive to produce and scarce**. You can't fill 10T tokens with curated math and code — there isn't that much on the open web, and generating synthetic data at that scale risks homogenizing the model. So the economical answer is:

- Use cheap, broad data for the bulk of training — enough tokens to span the knowledge distribution.
- Reserve the precious curated data for the tail of the run, where the LR is small and each step's update is weighted more heavily toward the final weights.

### Why annealing LR matters

The learning rate at the end of a cosine schedule is small — maybe 10% of peak or less. But the **effective update** per token late in training is still significant, because the model is close to a minimum and gradients point in the right direction.

More importantly, **the data mix during LR annealing disproportionately shapes the final weights**, because the optimizer is fine-tuning rather than exploring. Feeding the annealing phase the same web mix as Stage 1 wastes this leverage; feeding it math + code + reasoning text cashes it in as benchmark points.

Ablations in OLMo 2 and Llama 3 both show: a 100B-token Stage 2 on Dolmino recovers 5–10 points on GSM8K / MATH / HumanEval vs. 100B more tokens of Stage 1 data.

### LR schedule variants — where the anneal lives

"Anneal in Stage 2" is a convention used by three different underlying schedules. They look similar from a distance but differ in what Stage 1's LR is doing:

| Schedule | Stage 1 LR | Stage 2 LR | Used by |
|---|---|---|---|
| **Classical cosine** (no separate stage) | Cosine from peak → ~10% peak over whole run | — (no Stage 2) | GPT-3, early Llama |
| **Two-stage cosine** | Cosine from peak → *plateau* (e.g. 30% peak) | Plateau → ~0 over Stage 2 tokens | OLMo 2, Llama 3 |
| **[WSD](wsd-schedule.md)** (Warmup-Stable-Decay) | **Constant at peak** (no decay) | Peak → 0 over Stage 2 tokens | MiniCPM, some DeepSeek variants |

The important common property: **the aggressive LR decay is saved for Stage 2**, so the curated mix gets the leverage. The differences are:

- **Classical cosine** doesn't have a separate Stage 2 at all — LR is already near-zero at the end, and you've spent that leverage on generic web data. This is what modern labs moved away from.
- **Two-stage cosine** reaches a plateau at Stage 1's end, then continues decaying (steeper) through Stage 2. Continuous and smooth at the boundary.
- **[WSD](wsd-schedule.md)** keeps Stage 1's LR pinned at peak, then does *all* the decay in Stage 2. Conceptually cleanest — the "stable" endpoint is a reusable checkpoint you can fork with multiple Stage 2 variants.

So "during pre-training we don't anneal the LR" is only literally true under WSD. Under two-stage cosine (OLMo 2, Llama 3), LR is decaying throughout pre-training, just not aggressively — the aggressive part is reserved for mid-training.

### What goes in the Stage 2 mix

Typical ingredients (ratios vary by lab):

- **Heavily filtered web.** Same sources as Stage 1 but with much stricter [quality classifiers](../data/quality-filtering.md) — e.g., keep only top-decile FastText-classified pages.
- **Math.** Scraped math text, GSM-style problem-solution pairs, OpenWebMath, AutoMathText, sometimes synthetic chain-of-thought generations.
- **Code.** High-rated GitHub repos, Stack Exchange Q&A, competitive-programming solutions.
- **Academic.** arXiv, peer-reviewed prose, pes2o.
- **Instruction-like text.** FLAN-style templated prompts and responses — not full SFT, just instruction-flavored continuations.
- **Synthetic rewrites.** Teacher models paraphrasing hard examples, or generating problem-solution pairs from seed topics.

The Stage 2 mix is **always smaller than Stage 1's vocabulary of concepts**. Its job is to re-weight, not to broaden.

### Token budget

Common numbers across recent models:

- OLMo 2 7B: ~50B tokens Stage 2 on ~4T Stage 1. (~1.2%)
- OLMo 2 13B: ~100B on ~5T. (~2%)
- Llama 3 70B's annealing phase: ~40B on ~15T. (~0.3%)
- DeepSeek-V3: hundreds of billions of "high-quality" tokens late.

The exact percentage matters less than having the phase at all and getting the mix right.

## Why it matters

- **Benchmark scores are built here.** If you skip mid-training, your base model looks weak on math/code/reasoning relative to peers, even with the same Stage 1 compute.
- **Separates "teaching facts" from "teaching format".** Stage 1 absorbs world knowledge; Stage 2 aligns the model with high-quality reasoning patterns. Both matter, but they respond to different data.
- **Makes post-training cheaper.** A well mid-trained base model needs less SFT data to behave well. Starting point matters.

## Gotchas & tricks

- **Mid-training is not fine-tuning.** It's still next-token prediction on raw text, not instruction-response pairs. Instruction-style data is included as text, not as a supervised chat format. SFT comes later.
- **Contamination risk.** Heavy curated data overlaps with benchmark formats. [Decontaminate](../data/decontamination.md) the Stage 2 mix against MMLU, GSM8K, HumanEval etc. at least as aggressively as Stage 1.
- **Don't re-warmup LR.** The annealing continues from wherever Stage 1's schedule left off. Warming up LR at the Stage 1 → Stage 2 boundary disturbs the already-good weights.
- **Swap mix at a clean step boundary.** Don't interleave — step K on Stage 1 data, step K+1 on Stage 2. The optimizer state (momentum) carries over cleanly.
- **Validate on held-out Stage 2 data.** Loss on the curated mix drops *fast* at first (easy domain, small LR). Watch benchmark curves too — they're the real signal.
- **Over-training Stage 2 plateaus.** 50B is often enough; 500B rarely helps and can overfit to the mix's idiosyncrasies. This is not a scale-as-much-as-possible phase.
- **It pairs with [model souping](model-souping.md).** Running Stage 2 with multiple data-order seeds and souping the results is a double dip on free quality.

## Sources

- Paper: *2 OLMo 2 Furious* — AI2, 2024 — "Dolmino Mix 1124", most detailed public write-up.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — describes the annealing phase and its data mix.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — late-stage high-quality continuation.
- Paper: *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* — OpenBMB, 2024 — small-model variant of the same pattern, with stability analysis of the LR annealing choice.

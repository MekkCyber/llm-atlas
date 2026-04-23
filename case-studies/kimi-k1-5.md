# Case Study: Kimi k1.5

*Moonshot AI's long-CoT reasoning model, contemporaneous with DeepSeek-R1 (January 2025). A different recipe for the same problem — **online policy mirror descent** instead of GRPO, **length-penalized rewards** to fight overthinking, **partial rollouts** for 128k-context RL efficiency, and an explicit **long2short** distillation phase. Multimodal (text + vision). Not open-weights.*

**Related concepts:** [online-policy-mirror-descent](../post-training/reasoning/online-policy-mirror-descent.md) · [length-penalty](../post-training/reasoning/length-penalty.md) · [long2short](../post-training/reasoning/long2short.md) · [partial-rollouts](../systems/partial-rollouts.md) · [cot-reward-model](../post-training/cot-reward-model.md) · [rl-prompt-curation](../post-training/rl-prompt-curation.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [grpo](../post-training/grpo.md) · [ppo](../post-training/ppo.md) · [rope](../fundamentals/rope.md) · [deepseek-r1 case study](deepseek-r1.md)

---

## What this is

**Kimi k1.5**, released January 2025 by Moonshot AI. arXiv 2501.12599 (v1 Jan 2025, v4 Jun 2025). A long-CoT reasoning model matching or approaching OpenAI o1 on math and code reasoning, with a distinctive technical stack:

- **Online policy mirror descent** for the RL objective — an ℓ₂-regression surrogate on KL-regularized expected reward (a sibling to GRPO, derived from different first principles).
- **Length penalty** in the reward — asymmetric group-relative shaping that fights overthinking.
- **Partial rollouts** for system-level efficiency at 128k-context RL.
- **long2short** as an explicit post-processing phase — four methods to compress long-CoT capability into short-CoT models.
- **Multimodality** — text + vision, trained end-to-end including vision RL data.

The paper's framing is deliberately **"simplistic"**: no MCTS, no value network, no process reward model. Outcome reward plus careful engineering.

Kimi k1.5 is **not open-weights** (Appendix B explicit); the paper deliberately does not disclose parameter counts, pretraining tokens, or RL batch sizes. Architectural details are deferred to "future publications".

---

## The five-stage pipeline

From Sec. 2:

```
Pretraining
  ├─ vision-language pretraining           ← progressive multimodal introduction
  ├─ cooldown                              ← high-quality mix + synthetic rejection-sampled QA
  └─ long-context activation               ← 4,096 → 32,768 → 131,072; RoPE base = 1,000,000

Vanilla SFT
  ├─ ~1M text + ~1M text-vision
  └─ 1 epoch @ 32k, 1 epoch @ 128k

Long-CoT SFT cold-start
  └─ prompt-engineered warmup; planning/evaluation/reflection/exploration traces

Reinforcement Learning                     ← the heart of the paper
  ├─ online-policy-mirror-descent
  ├─ length-penalty reward
  ├─ partial rollouts infrastructure
  └─ curriculum + prioritized sampling

long2short
  └─ compress long-CoT capability into short-CoT via
     model merging / shortest rejection sampling / DPO / long2short-RL
```

Inheritance vs novelty: the base model architecture (a variant of the Transformer decoder) and prior Kimi-series training pipeline are inherited (Appendix B.3). Novel contributions per the paper's own framing: the RL algorithm, length penalty, partial rollouts, long2short, and vision-RL recipe.

---

## 1. Pretraining and multimodality

### Data (Appendix B.1, B.2)

Text: five domains (English, Chinese, Code, Math & Reasoning, Knowledge) through a 4-layer quality pipeline: rule-based → FastText classifier → embedding-similarity dedup → LLM-based scoring.

Multimodal: five categories — caption, image-text interleaving, OCR, knowledge, general QA.

### Three pretraining sub-stages (Appendix B.4)

1. **Vision-language pretraining.** Language-only first; gradual multimodal; vision tower trained in isolation (LM frozen), then unfrozen; eventual ratio ~30% V-L.
2. **Cooldown.** High-quality language + V-L data; synthetic QA rejection-sampled from math/knowledge/code, validated.
3. **Long-context activation.** 40% full-attention data + 60% partial-attention. **RoPE base = 1,000,000** (see [rope](../fundamentals/rope.md)). Max sequence length progressively extended 4,096 → 32,768 → 131,072.

### Vanilla SFT (Sec. 2.5.2)

- Text (~1M): 500k general QA, 200k coding, 200k math/science, 5k creative writing, 20k long-context.
- Multimodal (~1M): charts, OCR, image-grounded conversations, visual coding, visual reasoning.
- 1 epoch at 32k, then 1 epoch at 128k. LR decays 2e-5 → 2e-6 (stage 1); re-warms to 1e-5 then decays to 1e-6 (stage 2).

---

## 2. Long-CoT SFT cold-start (Sec. 2.2)

A "small yet high-quality long-CoT warmup dataset" (size not disclosed), generated via prompt engineering against the curated RL prompt set. Traces cover four cognitive processes explicitly: **planning, evaluation, reflection, exploration**. Light SFT — primes the model before RL without saturating.

Analogous to R1's Stage 1 cold-start SFT; emphasis on *breadth of cognitive modes* is Kimi-specific.

---

## 3. RL prompt curation (Sec. 2.1)

Before RL can run, the prompt pool is curated. Four rules — detailed in [rl-prompt-curation](../post-training/rl-prompt-curation.md):

1. **Difficulty labeling**: 10-sample pass-rate with the SFT model.
2. **Exclude hackable formats**: drop multiple-choice, true/false, proof-based.
3. **Easy-to-hack filter**: prompt for a direct answer with no CoT; if the model gets it right within **N = 8 attempts**, drop the prompt. A prompt that doesn't need CoT won't teach CoT.
4. **Domain tagging**: for balance and curriculum stratification.

The easy-to-hack filter is the distinctive piece — cheap, reusable, prevents a specific reward-hacking mode at the data layer.

---

## 4. The RL algorithm (Sec. 2.3.1–2.3.2)

The paper frames RL as **online policy mirror descent**. Full derivation and math in [online-policy-mirror-descent](../post-training/reasoning/online-policy-mirror-descent.md).

### KL-regularized objective (Eq. 2)

```
max_θ  E_{(x, y*) ~ D} [
    E_{(y, z) ~ π_θ} [ r(x, y, y*) ]  −  τ · KL( π_θ(·|x) || π_{θ_i}(·|x) )
]
```

Where `π_{θ_i}` is the reference policy at iteration `i`. The reference is **updated each iteration**; the optimizer state is reset accordingly.

### Gibbs-Boltzmann optimum

```
π*(y, z | x)  ∝  π_{θ_i}(y, z | x) · exp( r(x, y, y*) / τ )
```

Taking logs gives an identity relating the centered reward to `τ · log(π_θ / π_{θ_i})`.

### ℓ₂-regression surrogate

```
L(θ) = E [ ( r(x, y, y*) − τ log Z(x) − τ · log(π_θ(y,z|x)/π_{θ_i}(y,z|x)) )^2 ]
```

Rollouts sampled from the reference `π_{θ_i}`. In practice, `τ log Z(x)` is approximated by the **empirical mean reward** `r̄` over the group of `k` rollouts (like GRPO, minus the std normalization).

### The gradient (Eq. 3)

```
g = (1/k) Σ_j [
    ∇_θ log π_θ(y_j, z_j | x) · (r(x, y_j, y*) − r̄)
    − (τ/2) · ∇_θ (log(π_θ(y_j, z_j | x) / π_{θ_i}(y_j, z_j | x)))^2
]
```

Two terms: REINFORCE-with-mean-baseline + ℓ₂ regularization on log-ratios. No value network, no PPO clip, no importance-ratio clipping.

### Key design choices

- **No value network** (Sec. 2.3.2). Explicit justification: in long-CoT, exploring a wrong intermediate step that later recovers should be *reinforced*; a value network would penalize it.
- **No PRM, no MCTS** (Sec. 1, Sec. 4). The "simplistic framework" framing.
- **Optimizer reset per iteration**. Because each iteration has a new reference, the optimization landscape changes.

### Curriculum and prioritized sampling (Sec. 2.3.4)

- **Curriculum**: easier prompts first, progressively harder.
- **Prioritized sampling**: each prompt `i` with running success rate `s_i` is sampled proportional to `(1 − s_i)` — harder prompts get more weight.

---

## 5. Reward function

### Outcome reward (Sec. 2.3.1, Sec. 2.3.5)

`r(x, y, y*) ∈ {0, 1}`, from three sources:

- **Verifiable math**: rule-based answer match + a [CoT reward model](../post-training/cot-reward-model.md). Kimi reports **84.4% → 98.5%** spot-check accuracy going from a classic value-head RM to a CoT RM — a 14-point verifier-quality gap. Uses the CoT RM during RL.
- **Coding**: execution-based against **auto-generated test cases** via **CYaRon**. Pipeline: 50 test cases per problem; 10 ground-truth submissions validate; keep tests where **≥7/10 submissions agree**; admit problems where **≥9/10 submissions pass all selected tests**. From 1,000 online-contest problems → **323 admitted** with clean test suites.
- **Free-form answers**: a trained reward model (details undisclosed).

### Length penalty (Sec. 2.3.3)

The distinctive piece. Full math in [length-penalty](../post-training/reasoning/length-penalty.md).

```
λ(i) = 0.5 − (len(i) − min_len) / (max_len − min_len)

len_reward(i) = λ(i)         if r(x, y_i, y*) = 1    (correct)
              = min(0, λ(i)) if r(x, y_i, y*) = 0    (incorrect)
```

Correct responses get `±0.5` linearly interpolated by length; incorrect responses get `min(0, λ)` — penalized for length, never rewarded. The asymmetry prevents "give up quickly to save tokens" reward hacking. **Warm-up schedule**: disabled for initial RL, then constant-weight for the rest.

### Repetition penalty (Sec. 2.6.2)

Not a standalone reward term. Part of the partial-rollout infrastructure: repeat-detection terminates degenerate trajectories early, can apply a penalty.

---

## 6. System — partial rollouts (Sec. 2.6)

Full walkthrough in [partial-rollouts](../systems/partial-rollouts.md).

Problem: 128k-context RL has long-tail trajectory lengths; the longest trajectory blocks the whole step.

Mechanism:
1. Fixed per-iteration output token budget.
2. Trajectories that hit the cap are saved to a replay buffer and **continued next iteration**.
3. Rollout workers run asynchronously — fresh prompts start while long trajectories continue.
4. Responses assemble from segments across iterations; **only the final segment is on-policy** under the current reference; earlier stale segments are **masked out of loss**.
5. Repeat detection → early termination → optional repetition penalty.

### Hybrid Megatron + vLLM deployment (Sec. 2.6.3)

- **Megatron** for training, **vLLM** for inference, wrapped by a "checkpoint-engine" shim.
- Weights transferred via **Mooncake** (RDMA) between phases.
- Phase transitions: **<1 min training→inference**, **~10 s inference→training**.
- Collocated in a single Kubernetes pod; coordination via etcd.

### Code sandbox (Sec. 2.6.4)

- **crun** (not Docker), pre-created cgroups, overlay filesystem with tmpfs.
- Container startup: **0.12s → 0.04s**; throughput: **27 → 120 containers/sec** on 16-core machine.

---

## 7. long2short (Sec. 2.4, Sec. 3.4)

After long-CoT RL produces a verbose reasoner, four methods are compared to compress it into a short-CoT model. Full comparison in [long2short](../post-training/reasoning/long2short.md).

| Method | Cost | Quality (per Fig. 7) |
|---|---|---|
| **Model merging** | ~free | Weakest |
| **Shortest Rejection Sampling SFT** (n=8) | 1 SFT pass | Middle |
| **DPO** (shortest correct vs >1.5× long correct + incorrect) | 1 DPO pass | Middle-to-strong |
| **long2short RL** (second RL phase, length-capped rollouts + length penalty) | 1 RL phase | **Strongest** |

### Headline long2short numbers

- **k1.5-short w/ rl**: Pass@1 = **60.8 on AIME 2024** (avg over 8 runs) at **~3,272 tokens avg**.
- **k1.5-shortest**: Pass@1 = **88.2 on MATH500**.

---

## 8. Results (Sec. 3.2)

### Long-CoT (Table 2) — vs frontier reasoners

| Benchmark | QwQ-32B-Preview | o1-mini | o1 | **k1.5** |
|---|---|---|---|---|
| MATH-500 (EM) | 90.6 | 90.0 | 94.8 | **96.2** |
| AIME 2024 (Pass@1) | 50.0 | 63.6 | 74.4 | **77.5** |
| Codeforces (percentile) | 62 | 88 | 94 | **94** |
| LiveCodeBench (Pass@1) | 40.6 | 53.1 | **67.2** | 62.5 |
| MathVista-Test | — | — | 71.0 | **74.9** |
| MMMU-Val | — | — | **77.3** | 70.0 |

### Short-CoT (Table 3) — vs non-reasoning frontier

| Benchmark | Qwen2.5-72B-Inst | DeepSeek V3 | Claude-3.5-Sonnet | GPT-4o | **k1.5** |
|---|---|---|---|---|---|
| MMLU (EM) | 85.3 | 88.5 | 88.3 | 87.2 | 87.4 |
| IF-Eval (prompt-strict) | 84.1 | 86.1 | 86.5 | 84.3 | **87.2** |
| MATH-500 (EM) | 80.0 | 90.2 | 78.3 | 74.6 | **94.6** |
| AIME 2024 (Pass@1) | 23.3 | 39.2 | 16.0 | 9.3 | **60.8** |
| HumanEval-Mul (Pass@1) | 77.3 | **82.6** | 81.7 | 80.5 | 81.5 |
| LiveCodeBench (Pass@1) | 31.1 | 40.5 | 36.3 | 33.4 | **47.3** |

The abstract's "+550%" is AIME 2024 short-CoT vs GPT-4o: 60.8 / 9.3 ≈ 6.53× ≈ +553%.

### Benchmark context

See: [aime](../evaluation/aime.md), [math500](../evaluation/math500.md), [mmlu](../evaluation/mmlu.md), [livecodebench](../evaluation/livecodebench.md), [codeforces-benchmark](../evaluation/codeforces-benchmark.md), [humaneval](../evaluation/humaneval.md), [ifeval](../evaluation/ifeval.md).

### Ablations (Sec. 3.5, mostly qualitative — figures only)

- **Model size vs context length** (Fig. 8): larger model wins initially; smaller catches up with more CoT; larger has better **token efficiency**.
- **Negative gradients** (Fig. 10): k1.5 vs **ReST** (positives-only). k1.5 wins via negative gradients — "superior sample complexity." *Only explicit RL-method contrast in the paper.*
- **Curriculum sampling** (Fig. 9): warm-up all-difficulty, focus hard — "significantly enhances" vs uniform.
- **Length penalty ablation**: not reported numerically.
- **Partial rollouts ablation**: not reported numerically.

### Long-context scaling (Sec. 3.3, Figs. 5–6)

Accuracy and response length co-grow across RL training. Harder benchmarks show steeper length growth.

---

## 9. Explicit contrast with R1 — and the honest note

**The paper does not mention DeepSeek-R1 anywhere.** Not in v1 (January 2025, simultaneous with R1); not in v4 (June 2025, after R1 was published in *Nature*). The only DeepSeek reference is DeepSeek-V3 as a Table 3 baseline.

What the paper *does* explicitly contrast against:

- **MCTS, value networks, process reward models** — all declared unnecessary (Sec. 1, Sec. 4).
- **ReST** (Sec. 3.5) — ablation showing negative gradients matter.
- **Prompt-based CoT and planning-augmented CoT (MCTS-style)** — framed as two extremes k1.5 combines.

So R1-vs-k1.5 comparisons have to be constructed externally. A useful constructed-contrast:

| Aspect | DeepSeek-R1 | Kimi k1.5 |
|---|---|---|
| RL algorithm | [GRPO](../post-training/grpo.md) | [Online policy mirror descent](../post-training/reasoning/online-policy-mirror-descent.md) |
| Surrogate | PPO clip + KL penalty | ℓ₂ regression |
| Baseline | Group-mean with std normalization | Group-mean (no std normalization) |
| Reference update | Rolling | Per-iteration (optimizer reset each time) |
| SFT cold start | Yes (Stage 1 "thousands" of CoT samples) | Yes (small prompt-engineered set) |
| Pure-RL-from-base experiment | Yes (R1-Zero, 671B MoE) | Not reported at this scale |
| Rejection-sampled SFT round | Yes (Stage 3, ~800k samples) | Not in main pipeline (SRS is one of 4 long2short methods) |
| Long-CoT → short-CoT | Not in pipeline (handled via distillation into smaller models) | Explicit phase (long2short, 4 methods compared) |
| Length reward | None (R1 accepts the length growth cost) | Asymmetric group-relative length penalty with warm-up |
| Infrastructure highlight | MoE pretraining (DualPipe, FP8) | Long-context RL (partial rollouts, hybrid Megatron+vLLM) |
| Multimodal | Text-only | Text + vision, end-to-end |
| Open weights | Yes (R1, R1-Distill variants) | No |
| Headline comparability | AIME 2024 Pass@1 79.8%, MATH-500 97.3% | AIME 2024 Pass@1 77.5% (long), 60.8% (short-CoT); MATH-500 96.2% (long), 94.6% (short) |

Both papers argue for **outcome reward + scale over value/process supervision + search**. Different derivations, similar net philosophies.

---

## 10. Emergent behaviors

- Sec. 1: "planning, reflection, and correction" emerge from scaled context + RL, not from hand-engineered search. Consistent with R1's "aha moment" finding.
- Sec. 2.3.1: the model learns **trial-and-error** — outcome reward over long CoT rewards paths that took wrong turns mid-way but recovered, which is exactly the behavior a value-network-free RL can reinforce.

---

## What's still opaque

The paper deliberately doesn't disclose:
- Parameter counts of Kimi k1.5.
- Pretraining token budget.
- `k` (rollouts per prompt), RL batch size, sampling temperature.
- Number of RL iterations or steps.
- Size of the long-CoT SFT warmup set.
- Weighting coefficient on the length reward; length-penalty warm-up duration.
- Wall-clock / GPU-hour totals.
- Numeric ablation deltas for length penalty and partial rollouts in isolation.

"Details regarding model architecture scaling experiments lie beyond the scope of this report and will be addressed in future publications" (Appendix B.3).

---

## Key takeaways

1. **Mirror descent is a principled cousin to GRPO.** The ℓ₂-regression surrogate comes out of a clean derivation (KL-regularized RL → Gibbs optimum → log-ratio identity). Understanding the derivation illuminates GRPO's empirical PPO-clip-plus-group-baseline structure as one choice among several.
2. **Length penalties can be shaped to avoid give-up hacking.** The `min(0, λ)` asymmetry is a small but load-bearing design choice; naive penalties without the asymmetry collapse to "emit short wrong answers."
3. **Partial rollouts unblock long-context RL.** Long-tail trajectory lengths become a streaming problem, not a blocking one. A real system-level contribution, independent of the RL algorithm.
4. **long2short is a distinct phase, not a free boost.** Kimi treats short-CoT derivation as a separate pipeline stage with four specific methods; long2short RL wins. Contrast with R1, which gets short-CoT via distillation into smaller models rather than post-processing.
5. **CoT reward models are a real accuracy gain for verifier-bound RL.** 84% → 98% in Kimi's numbers. Generative verifiers with explicit reasoning beat scalar-head value RMs on math.
6. **Multimodal RL is viable at scale.** Kimi runs the full RL loop over text + vision prompts with the same reward machinery.
7. **The "simplistic" framing is deliberate.** No MCTS, no PRM, no value net. Both Kimi and R1 argue that outcome reward + strong base is enough; the convergence is notable given the independent recipes.

---

*Pairs well with:* the [DeepSeek-R1 case study](deepseek-r1.md) for direct comparison — R1 and k1.5 solved the same problem two different ways within the same week. Reading both end-to-end is the cleanest way to see the space of long-CoT-RL design choices.

# Case Study: DeepSeek-R1

*A 671B-total / 37B-active MoE trained — with no CoT SFT — into a frontier reasoning model via pure RL with verifiable rewards, then polished with a 4-stage pipeline. The paper that popularized "long-CoT RL" and the "distill the reasoner into small dense models" workflow.*

**Related concepts:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [orm](../post-training/reasoning/orm.md) · [prm](../post-training/reasoning/prm.md) · [mcts](../post-training/reasoning/mcts.md) · [_rewards](../post-training/_rewards.md) · [_rl](../post-training/_rl.md) · [deepseek-v3 case study](deepseek-v3.md)

---

## What this is

**DeepSeek-R1**, released January 2025 by DeepSeek-AI. Two artifacts, one pipeline:

1. **DeepSeek-R1-Zero.** RL-from-base. Starts from `DeepSeek-V3-Base` (671B total / 37B activated MoE) and runs [GRPO](../post-training/grpo.md) with only rule-based accuracy + format rewards. No SFT, no CoT demonstrations. Proof that reasoning emerges from pure RL on a strong base.
2. **DeepSeek-R1.** Production model. 4-stage pipeline — cold-start SFT → reasoning RL → rejection-sampled SFT → alignment RL — that fixes R1-Zero's language mixing and readability issues while matching OpenAI o1-1217 on reasoning benchmarks.

The paper (arXiv 2501.12948, published in *Nature* 645: 633–638) is the canonical reference for **long-CoT RL** — the idea that reasoning patterns (reflection, verification, length scaling) can be *elicited by RL* rather than learned from demonstrations. It's also the reference for **reasoning distillation**: SFT on R1-generated CoT traces uplifts small dense models (Qwen2.5, Llama 3) to near-o1-mini levels, and the paper shows distillation beats running RL directly on small models.

---

## R1-Zero — the "RL from base" experiment

### The recipe

- **Base model:** `DeepSeek-V3-Base` — same 671B-total / 37B-active MoE used for DeepSeek-V3, but **before** any SFT or RL.
- **Algorithm:** [GRPO](../post-training/grpo.md) (Group Relative Policy Optimization).
- **Reward:** rule-based only. Two components:
  - **Accuracy reward** — answer extracted from `<answer>...</answer>` tags, compared to reference (math: symbolic match; code: unit tests pass).
  - **Format reward** — +1 if the response is wrapped in `<think>...</think><answer>...</answer>`, else 0.
- **No neural reward models.** Deliberate, to prevent reward hacking.
- **Prompt template (verbatim from the paper):**
  > *"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within `<think></think>` and `<answer></answer>` tags, respectively."*
- **No SFT, no few-shot, no demonstrations.** The model is told *that* to think, not *how*.

### What emerges during training

Three phenomena, in rough order:

1. **Response length grows autonomously.** From hundreds to thousands of tokens across thousands of RL steps. No length reward. The model learns "thinking longer helps on hard problems" purely because longer traces score higher on the accuracy reward.
2. **Reasoning structure appears.** Numbered steps, sub-problems, intermediate checks — none of this is in the prompt or reward.
3. **Reflection / "aha moment."** Mid-to-late training the model begins writing things like *"Wait, wait. Wait. Let me reconsider step 3…"* and switches strategies mid-chain. The paper's own phrasing: *"This moment is not only an 'aha moment' for the model but also for the researchers observing its behavior."*

### Headline numbers (Nature-version R1-Zero)

| Benchmark | R1-Zero |
| --- | --- |
| AIME 2024 pass@1 | **77.9%** (v1 arXiv reports 71.0%) |
| AIME 2024 cons@64 | 86.7% |
| MATH-500 pass@1 | 95.9% |
| GPQA Diamond | 73.3% |
| Codeforces rating | 1444 |

Training trajectory on AIME: **15.6% → 77.9%** pass@1.

### R1-Zero's flaws

- **Poor readability.** Traces are stream-of-consciousness, run-on, minimal structure.
- **Language mixing.** Chinese and English interleave mid-CoT, especially on multilingual prompts.

These two problems are what the R1 pipeline exists to fix.

For the generalized concept, see [long-cot-rl](../post-training/reasoning/long-cot-rl.md).

---

## The R1 4-stage pipeline

```
DeepSeek-V3-Base
     │
     ▼
 Stage 1 — Cold-start SFT on "thousands" of clean CoT examples
     │     (few-shot prompting, R1-Zero outputs, human post-processing;
     │      format: |special_token|<reasoning>|special_token|<summary>)
     ▼
 Stage 2 — Reasoning-oriented RL (GRPO)
     │     rewards = accuracy + format + language-consistency  (direct sum)
     │     domains: math, code, science, logic
     ▼
 Stage 3 — Rejection sampling + SFT  (see rejection-sampling.md)
     │     ~600k reasoning samples (rule verifier + DeepSeek-V3 judge)
     │     ~200k non-reasoning samples (reuse V3 SFT dataset; synthetic CoT for some)
     │     total ~800k SFT samples, 2 epochs over V3-Base
     ▼
 Stage 4 — All-scenarios RL (GRPO)
     │     rule-based rewards for reasoning prompts
     │     preference RMs for general prompts:
     │       helpfulness RM scores the SUMMARY only
     │       harmlessness RM scores the ENTIRE response (including CoT)
     ▼
 DeepSeek-R1
```

### Stage 1 — Cold-start SFT

Purpose: give the model a *readable CoT prior* before RL. Collected "thousands" (paper's word, not tens of thousands) of long-CoT examples by:
- few-shot prompting with long CoT as examples
- prompting the model to reflect and verify
- taking **R1-Zero's raw outputs** and reformatting them
- **human post-processing** for readability

Format imposed: `|special_token|<reasoning_process>|special_token|<summary>` — reasoning followed by a clean user-facing summary. Filter: drop mixed-language, poor-markdown, unreadable samples.

### Stage 2 — Reasoning-oriented RL

Same GRPO + rule-based rewards as R1-Zero, plus one new term:

- **Language-consistency reward** = proportion of target-language words in the CoT.
- Combined with accuracy by **direct sum**.

The language term slightly hurts raw accuracy but produces much more user-friendly traces. The paper calls this out as a deliberate tradeoff — a real instance of **alignment cost** (see [_rewards](../post-training/_rewards.md) on shaping rewards).

Domains: math, code, science, logic. Run until **convergence on reasoning tasks** (exact step count not disclosed).

### Stage 3 — Rejection sampling + SFT

The pivotal stage. Use the Stage-2 checkpoint to generate a large SFT dataset via [rejection sampling](../post-training/rejection-sampling.md):

**Reasoning data (~600k samples):**
- Multiple rollouts per prompt; keep only **correct** ones per rule verifier.
- Where no verifier applies: use **DeepSeek-V3 as a generative judge** (compare model output to ground truth).
- Filter: drop mixed-language CoT, overlong paragraphs, stray code blocks.

**Non-reasoning data (~200k samples):**
- Writing, factual QA, self-cognition, translation, etc.
- Reused from the DeepSeek-V3 SFT pipeline.
- For some prompts, V3 generates a short synthetic CoT before answering; for trivial prompts (e.g., "hi"), no CoT.

**SFT:** ~800k samples total, **2 epochs** over `DeepSeek-V3-Base` (full re-SFT, not a continuation of Stage 2's RL checkpoint).

### Stage 4 — All-scenarios RL (alignment pass)

Dual reward stack:
- **Reasoning prompts:** same rule-based rewards as before.
- **General prompts:** **learned preference reward models** for helpfulness and harmlessness.
  - **Helpfulness RM** scores the **final summary only** — deliberately NOT the CoT, to prevent preference optimization from corrupting the reasoning trace.
  - **Harmlessness RM** scores the **entire response** including CoT, to catch hidden risky content.

Prompt distribution spans reasoning + general; the goal is a broadly usable instruct model.

---

## Headline evaluation — DeepSeek-R1

Decoding: temperature **0.6**, top-p **0.95**, max generation length **32,768 tokens**. Pass@1 averaged over k=4–64 samples.

| Benchmark | DeepSeek-R1 |
| --- | --- |
| AIME 2024 pass@1 | **79.8%** |
| MATH-500 pass@1 | **97.3%** |
| Codeforces rating | **2029** (~96.3 percentile) |
| LiveCodeBench | **65.9%** |
| GPQA Diamond | **71.5%** |
| MMLU | 90.8% |
| MMLU-Pro | 84.0% |

Roughly on par with OpenAI o1-1217 across reasoning; ahead of o1-mini and o1-preview on math.

---

## Distillation — and the key negative result

**Targets:** dense models SFT'd on Stage 3's 800k samples. No RL on the distilled models (explicitly left to the community).

| Distilled model | AIME 2024 | MATH-500 | LiveCodeBench | Codeforces |
| --- | --- | --- | --- | --- |
| R1-Distill-Qwen-**1.5B** | 28.9% | 83.9% | — | — |
| R1-Distill-Qwen-**7B** | 55.5% | — | — | — |
| R1-Distill-Qwen-**14B** | 69.7% | — | — | — |
| R1-Distill-Qwen-**32B** | 72.6% | 94.3% | 57.2% | 1691 |
| R1-Distill-Llama-**8B** | — | — | — | — |
| R1-Distill-Llama-**70B** | 70.0% | 94.5% | — | — |

The **R1-Distill-Qwen-1.5B** beats GPT-4o and Claude 3.5 Sonnet on AIME at 1.5B parameters — a small-model-wins-big result that hinges on distilling from a strong reasoner.

### RL-on-small-model control experiment

The paper runs an important ablation: train Qwen-32B-Base directly with **>10,000 RL steps** of math/code/STEM GRPO → *DeepSeek-R1-Zero-Qwen-32B*. Then compare to the SFT-distilled `R1-Distill-Qwen-32B`.

**Result: the distilled 32B comfortably beats the RL-from-scratch 32B.** The paper's two conclusions:

1. **Distilling from a stronger reasoner is more efficient** than running large-scale RL on a smaller base.
2. But **advancing the frontier** still requires powerful base models and large-scale RL — distillation cannot exceed the teacher.

This is arguably the most actionable engineering takeaway from the paper: **if you have budget for one thing on a 7B–32B model, it's SFT on R1's traces — not your own RL run.**

---

## Failed attempts the paper documents

### Process Reward Model ([PRM](../post-training/reasoning/prm.md))
Three reasons PRMs were abandoned:
1. *"It is challenging to explicitly define a fine-grain step in general reasoning."*
2. *"Correctness of intermediate steps is hard to judge"* — automated labels are unreliable; manual doesn't scale.
3. *"A model-based PRM inevitably leads to reward hacking."*

### Monte Carlo Tree Search ([MCTS](../post-training/reasoning/mcts.md))
Four reasons MCTS was abandoned:
1. *"Token generation presents an exponentially larger search space"* vs chess/Go.
2. Node-expansion caps cause local optima.
3. *"Training a fine-grained value model is inherently difficult."*
4. AlphaGo's value-model-driven iterative improvement *"proves difficult to replicate in our setup due to the complexities of token generation."*

### Language mixing
Occurred most with multilingual RL prompts. Mitigation = language-consistency reward in Stage 2. Residual issue on languages other than Chinese/English.

These failures are part of the paper's contribution: they argue that the frontier-reasoning-RL recipe **does not need** process rewards or search. Outcome-only rule-based rewards + strong base + [long-CoT RL](../post-training/reasoning/long-cot-rl.md) is the recipe.

---

## Key takeaways

1. **RL can elicit reasoning from base — no CoT SFT required.** The most important framing-shift in post-2024 reasoning work. See [long-cot-rl](../post-training/reasoning/long-cot-rl.md).
2. **Outcome-only rule-based rewards beat learned rewards for reasoning.** PRMs, learned ORMs, and MCTS-based search all underperform simple rule verifiers at scale. Reward hacking is the reason.
3. **The 4-stage pipeline separates capability from legibility.** RL gets you capability; cold-start SFT + rejection-sampled SFT + alignment RL get you legibility, coverage, and safety. Each stage fixes one thing R1-Zero left broken.
4. **Helpfulness RM scopes matter.** Scoring only the summary (not the CoT) with a helpfulness RM is a specific, load-bearing choice to prevent preference RL from degrading reasoning traces. Reusable idea.
5. **Distillation > small-model RL, up to the teacher.** Small dense models should be SFT'd on R1's traces, not RL'd from scratch. The frontier still needs large bases + large-scale RL; distillation doesn't push the ceiling.
6. **Shaping rewards have a named cost.** Adding the language-consistency term costs a small amount of benchmark score and is done deliberately. Honest reward-shaping papers should report this tradeoff.
7. **GRPO + rule rewards generalize.** Whether from base (R1-Zero), from cold-start SFT (Stage 2), or combined with preference RMs on general prompts (Stage 4), the same algorithm works across 4 different data regimes. See [grpo](../post-training/grpo.md).

---

## What's still opaque

- **Exact GRPO hyperparameters** — group size `G`, clip `ε`, KL coefficient `β`, optimizer settings, learning rate, batch size. Not disclosed for R1 or R1-Zero.
- **Training-step counts** for each stage. The paper says "until convergence" for Stage 2 and mentions ">10,000 steps" only for the 32B control experiment.
- **Stage 1 dataset size** is stated as "thousands" but not given precisely.
- **Preference RM details** (Stage 4) — data, size, architecture.
- **No open intermediate checkpoints.** R1 is open weights; R1-Zero and the Stage-2 checkpoint are not separately released.
- **Compute budget.** Training cost for R1 (beyond V3-Base) is not stated.
- **v1 vs Nature numbers.** The Jan 2025 arXiv v1 and the updated Nature version differ on several benchmark numbers (e.g., R1-Zero AIME pass@1: 71.0% → 77.9%), reflecting extended training between submissions.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for the base-model side — V3 produced the architecture and the pretrained weights R1 builds on; R1 is what you get when you run the reasoning-RL recipe on that base. V3's own chat post-training also uses R1 traces (SFT distillation) — the two papers are a loop: V3-Base → R1 → V3-chat distillation.

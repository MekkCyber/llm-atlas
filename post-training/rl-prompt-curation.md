# RL Prompt Curation
*Depth — the data-engineering step that decides which prompts enter RL, from the Kimi k1.5 recipe.*

**TL;DR:** RL training is bounded by the quality of the prompt pool. Kimi k1.5 (Sec. 2.1) specifies four curation rules: (1) **difficulty labeling via 10-sample pass-rate** using an SFT model, (2) **exclude easily-hacked formats** (multiple choice, true/false, proof-based), (3) **easy-to-hack filter** — prompt the model to *guess without CoT*; if correct within **N = 8 attempts**, drop the prompt, (4) domain tagging for balance. The easy-to-hack filter is the distinctive piece: **prompts the model can guess without reasoning don't teach reasoning** — they contribute noise to the RL signal. A small, cheap, reusable filter for any RLVR pipeline.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md), [_rewards](_rewards.md)
**Related:** [long-cot-rl](reasoning/long-cot-rl.md) · [rejection-sampling](rejection-sampling.md) · [_data-curation](../data/_data-curation.md) · [kimi-k1-5 case study](../case-studies/kimi-k1-5.md)

---

## What it is

A set of filtering rules applied to the **prompt pool** before RL training begins. Not a training technique — a data-engineering technique specific to RL post-training.

Why this matters separately from pretraining data curation:
- **Pretraining data** is measured in trillions of tokens; curation is about throughput, dedup, language balance.
- **RL prompts** are measured in hundreds of thousands to millions; curation is about **reward-signal quality** per prompt. Every prompt consumes `k` rollouts worth of compute per step; noisy prompts burn compute and contribute no learning signal.

The four rules in Kimi k1.5 Sec. 2.1 target three desiderata: **diverse coverage** (domain tagging), **balanced difficulty** (pass-rate labeling + curriculum), and **accurate evaluability** (exclude hackable formats + drop easily-guessable prompts).

---

## How it works

### The four rules

#### 1. Difficulty labeling via pass-rate

For each candidate prompt `x`:

```
samples = [SFT_model.generate(x, temperature=T_high) for _ in range(10)]
pass_rate(x) = mean(correctness(x, s) for s in samples)
```

The SFT model is used (not the RL checkpoint) so the label is stable and doesn't drift with training. `pass_rate` is a continuous difficulty proxy — 0% = no model can solve it, 100% = trivial.

This feeds **curriculum** (train easier prompts earlier) and **prioritized sampling during RL** (Kimi k1.5 Sec. 2.3.4: sample each prompt proportional to `(1 − s_i)`, where `s_i` is the running success rate — harder prompts get more weight).

#### 2. Exclude easily-hacked formats

Drop prompts of three types:
- **Multiple choice.** `r = 1 / #options` expected by chance; RL amplifies lucky guesses.
- **True/false.** Same, 50% expected by chance.
- **Proof-based.** Correctness is hard to grade automatically — either you need a theorem prover or an LLM judge, both of which are hacking surfaces.

The rule is domain-agnostic; it targets *format*. Well-posed, compute-cheap exclusion.

#### 3. Easy-to-hack filter

The distinctive piece:

```
for prompt x in candidate pool:
    for i in range(8):
        # Prompt the SFT model to guess the answer DIRECTLY, with no CoT
        guess = SFT_model.generate(x, prompt_style="no_cot")
        if correctness(x, guess) = 1:
            discard x
            break
```

If the model can get the answer right within **N = 8 attempts without reasoning**, the prompt is dropped. Rationale: *a prompt where the answer can be guessed without chain-of-thought is a prompt that doesn't need CoT to solve*. Keeping it in RL means the model gets rewarded for prompts that don't require the capability we're trying to train.

This is cheap — 8 forward passes per candidate, no CoT generation — and it removes a specific class of reward hacking where the model learns "emit plausible short answer, skip reasoning."

#### 4. Domain tagging for balance

Each prompt is tagged by domain (STEM / competitions / general reasoning / coding / science / logic). Used for:
- **Balancing** the RL mix (don't let one domain dominate).
- **Per-domain evaluation** of RL progress.
- **Combining with difficulty tags** for curriculum stratification.

Kimi covers text-only and image-text prompts for multimodal RL. Specifics per domain are not disclosed.

### Ordering

Apply in sequence (each pass operates on the survivors of the previous):

```
candidates → domain tagging → format exclusion → difficulty labeling → easy-to-hack filter → RL prompt pool
```

Alternative ordering puts easy-to-hack filter before difficulty labeling to save compute (drop hackable prompts before spending 10 samples each on pass-rate labeling). Either works; the paper doesn't specify.

---

## Why it matters

- **Reward signal quality is the ceiling of RL.** A noisy prompt pool limits how much the policy can learn, no matter how many steps you run. The rules above each remove a specific noise source:
  - Format exclusion → removes guessability noise.
  - Easy-to-hack filter → removes non-reasoning prompts that shouldn't count.
  - Difficulty labeling → enables curriculum / prioritization (downstream).
- **Cheap compared to the RL training it gates.** A 10-sample pass-rate labeling per prompt + 8 forward-pass hack check is ~18 forward passes per prompt. RL training is `k × steps × per-step-rollout-cost` — often millions of forward passes per prompt. Curation is negligible in context.
- **Prevents reward hacking at the source, not after.** Drop the bad prompts, not the bad behaviors. Once a prompt is in the RL pool, the policy will exploit any hackability it contains. Prevention > detection.
- **Reusable recipe.** The rules are task-agnostic at the level of "format" and "hackability" — they transfer to any RLVR pipeline (RL for code, RL for tool use, etc.) with small adaptations.

---

## Gotchas & tricks

- **The "no-CoT guess" prompt style is load-bearing.** If the easy-to-hack filter lets the model reason (CoT) before guessing, every reasoning-solvable prompt gets dropped. The filter's whole point is to differentiate "guess-from-surface-features" from "requires reasoning" — which requires a CoT-suppressing prompt (e.g., "Give the final answer only, no explanation.").
- **`N = 8` attempts is a paper choice.** Too few → keep prompts that are borderline-guessable. Too many → drop prompts that are solvable with reasoning but occasionally guess-correct. Kimi uses 8; tune per domain.
- **SFT model for labeling, not base model.** Labeling with a base model gives noisier pass rates (base models are bad at instruction-following). Labeling with the current RL checkpoint contaminates the labels (moves with training). The SFT model is the stable choice.
- **Difficulty labels shift over time.** A prompt labeled "60% pass rate" by the SFT model might become "90% pass rate" after RL. The labels are a static snapshot; re-labeling mid-training is possible but rare. For prioritized sampling, use the RL checkpoint's current success rate, not the fixed label.
- **Domain imbalance creates capability imbalance.** If 80% of prompts are math, the post-RL model is strong at math and weak elsewhere. Balance by domain; if you want the model to be strong on one domain, over-sample it *deliberately*, not accidentally.
- **Multi-modal prompts need their own curation.** Image-text prompts can't be guessed without looking at the image; the "easy-to-hack" filter needs to run in a setting where the image is actually presented (not stripped). Kimi notes they run vision-prompt curation separately.
- **Doesn't detect subtle hackability.** "The final answer is likely 4" kinds of surface patterns can survive the 8-attempt filter. If the model can't guess but can still pattern-match, the filter misses it. Use in combination with [decontamination](../data/decontamination.md) and manual spot-checks.
- **Proof-based prompts can be readmitted with an LLM judge.** Dropping proofs is conservative; a strong judge LLM can grade proofs (with some hackability). For a production RL pipeline with proofs, train a specialist CoT RM for proof grading (see [cot-reward-model](cot-reward-model.md)).
- **Don't ship the labeled prompt set.** The labels (pass rates, difficulty tags) are artifacts of your labeling model; shipping them publicly reveals your labeling choices. If you release a public RL prompt pool, release the prompts but not the labels (or label with a public off-the-shelf model).

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces the four curation rules (Sec. 2.1) and uses them to feed the RL pool.
- Related: [rejection-sampling](rejection-sampling.md) — uses a similar "sample N, filter by correctness" pattern, applied to SFT-data generation rather than prompt curation.
- Related: [rlvr](rlvr.md), [grpo](grpo.md), [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md) — the RL algorithms that consume the curated prompt pool.
- Related: [_data-curation](../data/_data-curation.md) — pretraining-side data curation; shares patterns (filter noise before training) but is far larger in scale.

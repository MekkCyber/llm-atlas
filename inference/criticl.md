# CritICL — Critique-Based ICL from Weaker-Model Failure Modes
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time reasoning-boost that feeds **critiques of a weaker model's characteristic failures** to a stronger model as in-context examples. The observation: LLM failure modes are family-structured — a small model in a family systematically slips in ways a larger model in the same family also tends to slip toward. Feeding *critiques* of those slips into the larger model's context nudges it away from the same pit. Two variants: **CritICL-dynamic** predicts input-specific failures and retrieves matching critiques; **CritICL-static** uses a global failure-mode profile. Consistently beats standard ICL and matches or beats test-time scaling at *significantly fewer generations and lower token cost* — no training required.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Test-time scaling for reasoning (majority vote, best-of-N, tree search) is dominated by "generate more, pick one." That's compute-heavy and only works when the model can produce correct responses at all. CritICL takes a different angle: instead of asking the strong model to try harder, tell it *what not to do*, sourced from a much cheaper weak model's mistakes.

The pitch is a lightweight "weak-to-strong" signal delivered *entirely at inference*. No training, no reward model, no rollouts. Just a critique bank and an ICL pass.

## How it works

**Prep (offline, one-time per task family):**

1. Run a weaker model in the same family on a set of representative problems.
2. Collect the mistakes and, for each, generate a natural-language critique: *"the model dropped a factor of 2 in step 3 because it forgot the derivative of sin was cos, not -cos."*
3. Optionally cluster critiques into a **failure-mode profile** — the recurring patterns across mistakes.

**At inference:**

- **CritICL-static** — prepend a fixed set of critiques (or the failure-mode profile) to every strong-model prompt as ICL examples. Cheapest.
- **CritICL-dynamic** — predict which failure modes are most likely for this specific input (from input embedding or a lightweight classifier), retrieve the matching critiques, and use them as ICL examples. More expensive setup, but better targeting.

Either way, the strong model produces a single response — no repeated sampling, no verifier, no reward model.

## Why it matters

- **Cheaper than test-time scaling at comparable quality.** Matches or beats majority-vote and best-of-N at a fraction of the generation budget.
- **Complementary to inference-time scaling.** Nothing stops you from stacking CritICL with self-consistency; the critiques improve the base distribution being sampled from.
- **Weak-to-strong at inference is unusual.** Most weak-to-strong work is training-time (weak supervisor → strong student). CritICL is a training-free instance.
- **Extracts value from cheap weak-model outputs.** The weak model's failures aren't wasted; they seed the critique bank.

## Gotchas & tricks

- **Weak and strong must share a family.** Cross-family transfer is weaker in the paper's ablations — the critiqued failure modes have to be modes the strong model actually shares.
- **Critique quality matters.** Auto-generated critiques from the weak model itself are worse than critiques written by a mid-tier model or a human; treat critique generation as a small ML problem, not a freebie.
- **Static profile can stagnate.** As the task distribution drifts, the static profile misses newly-common failure modes. Dynamic retrieval or periodic profile refresh helps.
- **Context-cost of critiques is real.** Each critique added to the prompt costs tokens; over-fetching in dynamic mode erodes the token-cost advantage vs test-time scaling.
- **Beware of anti-instruction ICL.** A critique like *"don't confuse X with Y"* can inadvertently prime the model to think about X-vs-Y and answer wrong. Prefer critiques phrased as positive procedures where possible.

## Sources

- Paper: *CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes* — Wu et al. (USTC / Tsinghua), 2026 — arXiv:2608.27455.

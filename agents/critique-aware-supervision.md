# Critique-Aware Supervision (CAST)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training method for long-horizon tool-calling agents that converts sparse task-outcome rewards into **per-action supervision**. A **critique model** is trained to produce structured rationales explaining whether each action in a trajectory was valid under domain policies; those critiques then become supervision for policy fine-tuning. Fine-tuning Qwen3-family models with CAST outperforms **GPT-OSS-120B by >10% pass⁴** on Retail tool-calling tasks and adds **+9% out-of-domain on Telehealth**.

**Prereqs:** [tool-calling.md](tool-calling.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

Long-horizon tool-calling agents operate under partial observability and multi-step, policy-governed dynamics. A single wrong action (refunding the wrong purchase, dispensing the wrong medication) can be irreversible. Task-level reward is sparse (success/failure at the end), and frontier LLMs — used as critique agents in prompt-only pipelines — struggle to explain *why* an action was wrong in a way that transfers to training signal.

CAST bridges that gap: train a critique model that produces per-action rationales, then use those rationales as supervision for the policy.

## How it works

**Stage 1 — trajectory analysis.** Sparse outcome-labeled trajectories (success vs failure) are decomposed action-by-action. A critique model is trained (or prompted with domain policies) to produce **structured verification rationales**: for each action, a natural-language explanation of validity under partial observability and domain rules.

**Stage 2 — critique-aware training data.** Rationales are compiled into a dataset: `(state, action, critique)`. Rationale content is *policy-anchored* — it references specific domain rules that make an action valid or invalid, not general "seems right" heuristics.

**Stage 3 — policy fine-tuning.** The base policy is fine-tuned on rationale-augmented trajectories. Two variants:
- **SFT with critique as target** — model is trained to output actions that satisfy critiques.
- **RL with critique-derived rewards** — critique judgments (action-valid, action-invalid, action-questionable) become dense per-action rewards for GRPO / PPO.

**Reliability metric: pass⁴.** Rather than pass@1 (any single trial succeeded), the paper reports pass⁴ (all four independent trials succeeded). Long-horizon agents fail probabilistically; pass⁴ punishes intermittent errors that pass@1 hides.

## Why it matters

- **Dense signal from sparse rewards.** Long-horizon tool-calling has almost no signal at task-outcome level; per-action critiques unlock supervised-scale gradients.
- **Beats scale.** Qwen3 with CAST beats GPT-OSS-120B on the same benchmarks — evidence that structured supervision > raw parameters for reliability.
- **Out-of-domain transfer.** +9% on Telehealth from Retail training suggests critique-style supervision teaches transferable reasoning about action validity, not just task-specific patterns.
- **Naturally auditable.** Each critique is a text rationale; failure modes are inspectable rather than opaque reward-model scalars.

## Gotchas & tricks

- **Critique-model quality is the bottleneck.** A weak critique model produces noisy rationales and the fine-tuned policy inherits that noise. Bootstrap the critique model from strong domain-expert traces.
- **pass⁴ is a much harder metric.** A 90% pass@1 model can have <70% pass⁴ from independence failure. Baseline correctly.
- **Policy anchoring matters.** Rationales that say "seems wrong" don't transfer; rationales that reference specific policy clauses do. Structure the critique template accordingly.
- **Critique-derived rewards can be gamed** if the critique model has systematic blind spots. Rotate critique models across training and evaluate policy on held-out critiques.
- **Multi-tool composition amplifies error rates.** Reliability drops multiplicatively in step count; CAST helps but doesn't eliminate this. Aggressive per-action checking at inference is complementary.

## Sources

- Paper: *CAST: Critique-Aware Supervision for Training Reliable Long-Horizon Tool-Calling Agents* — Saeidi et al. — ASU / Cisco Research, 2026 — arxiv.org/abs/2608.30147.

# Negative Self-Distillation (NSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A label-free self-improvement loss for reasoning LLMs that pushes the model's distribution *away* from a self-generated **negative teacher** (e.g. the model role-playing a "careless reasoner") rather than *toward* an overconfident ground-truth-conditioned trace. Beats On-Policy Self-Distillation (OPSD) on reasoning benchmarks, and a token-gating mechanism prevents catastrophic collapse of general language ability.

**Prereqs:** [../_rl](../_rl.md), [long-cot-rl](long-cot-rl.md)
**Related:** [../rejection-sampling](../rejection-sampling.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

OPSD lets a model act as its own teacher by conditioning on privileged information (ground-truth answer) and imitating the resulting "confident" trace. Recent work shows OPSD *hurts* complex reasoning: the imitation target is artificially confident, so the student learns to suppress uncertainty and stop exploring — exactly the behaviors long-CoT reasoning depends on.

NSD flips the sign. Instead of an overconfident positive teacher, it constructs a **negative** teacher — the model self-prompted into a flawed reasoning persona ("careless reasoner", "shortcut-taker") — and minimizes a divergence pushing the student *away* from that distribution. No ground-truth solutions, no preference dataset, no external verifier.

## How it works

- **Negative teacher generation.** For each question, prompt the same model with a persona that induces the failure mode you want to avoid ("You are a careless reasoner…"). This produces per-question negative distributions $q^-(y | x)$.
- **Divergence loss.** Push the student's distribution $p_\theta(y | x)$ away from $q^-$. Naïve choice: negative KL. Practical choice: a bounded divergence to keep gradients tame.
- **Token-critical gating.** Naively unlearning flawed traces also unlearns basic linguistic tokens (function words, punctuation) that appear identically in flawed and good traces. A **dynamic gate** classifies each token as reasoning-critical vs linguistic; gradients only flow through reasoning-critical tokens. This preserves fluency while shifting reasoning behavior.
- **On-policy sampling.** Both positive rollouts (student) and negative rollouts (persona-prompted student) come from the same base model, kept in sync.

## Why it matters

Label-free bootstrapping is attractive when you have no gold answers and no reward model — but naïve self-distillation collapses reasoning. NSD is the first working variant that (a) uses no external supervision, (b) improves complex reasoning, and (c) does not degrade fluency. That is a rare combination in reasoning-RL.

## Gotchas & tricks

- **Gating is not optional.** Without the token-critical gate, NSD catastrophically forgets basic language — the model still refuses "careless reasoner" outputs but at the cost of coherence.
- **Persona design matters.** "Careless" is a broad brush; more targeted personas ("skip verification", "assume the first plausible answer") produced sharper improvements in ablations.
- **Divergence choice controls stability.** Unbounded reverse-KL blows up; bounded divergences (JS, α-divergence) or clipped KL keep training stable.
- **On-policy > off-policy for the negative.** Re-sampling the negative each step keeps it aligned with the current student; a frozen negative teacher stops teaching.
- **Combine with positive signal when available.** NSD is designed for label-free settings; if you have a verifier, mixing NSD with RLVR gets better than either alone.

## Sources

- Paper: *Negative Self-Distillation: Learning to Reason by Avoiding Flaws* — Pei, Wei, Xu, Zhu, Chen, Meng (University of Virginia), 2026 — https://arxiv.org/abs/2609.11699

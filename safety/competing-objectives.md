# Competing Objectives

*Depth — the first of Wei et al.'s two failure modes of LLM safety training: jailbreaks that arise because the model's pretraining, instruction-following, and safety objectives are put at odds by the prompt.*

**TL;DR:** RLHF-aligned models are optimized against a **sum** of objectives — helpfulness, instruction-following, safety, KL-to-pretraining. An attacker who engineers a prompt where obeying safety **requires disobeying one of the other objectives** creates internal conflict that safety frequently loses. This is not the model being "fooled"; it's a structural consequence of the training objective. Every attack in this family works by forcing that conflict: *continuation pressure* (prefix injection, roleplay), *vocabulary constraints* (refusal suppression, style injection), or *instruction overload* (distractor attacks). Patching individual attacks doesn't fix the mechanism; only changing the training objective does.

**Prereqs:** [_post-training](../post-training/_post-training.md), [_rl](../post-training/_rl.md)
**Related:** [mismatched-generalization](mismatched-generalization.md), [_jailbreaks](_jailbreaks.md), [prefix-injection](prefix-injection.md), [refusal-suppression](refusal-suppression.md), [style-injection](style-injection.md), [roleplay-jailbreak](roleplay-jailbreak.md), [distractor-attack](distractor-attack.md), [evil-system-prompt](evil-system-prompt.md)

---

## What it is

Wei, Haghtalab, and Steinhardt (2023) define **competing objectives** as:

> *"Competing objectives occur when a model's pretraining and instruction-following objectives are put at odds with its safety objective."*

A modern RLHF-aligned chat model is the fixed point of a multi-term optimization:

```
Policy = argmax_π  [ E[R_helpful(π)] + E[R_safety(π)] − β · KL(π || π_ref) ]
```

- `R_helpful` — learned preference reward or instruction-following signal.
- `R_safety` — harmlessness reward (refuse certain requests).
- `β · KL(π || π_ref)` — KL penalty to a reference model (usually SFT or base). This pulls generations back toward the *pretraining* distribution on any dimension safety hasn't explicitly shaped.

Safety is **one term among several**, and it's a comparatively thin slice of the training signal. When a prompt creates strong pressure from the *other* terms — something that under pretraining is highly likely to complete in a specific way, or something instruction-following rewards strongly — safety has to override that pressure to produce a refusal. It loses often enough to be exploitable.

The attacker doesn't defeat safety; they **tilt the conflict** so the non-safety terms dominate.

---

## How it works

### The three canonical levers

Every competing-objectives attack uses one (or several) of these three levers:

1. **Continuation pressure.** Engineer the prompt so a refusal has extremely low probability under the pretraining distribution. The `π_ref` KL term and the instruction-following reward both push the policy toward completing in-distribution; the model follows.
2. **Vocabulary / style constraints.** Forbid the model from using the tokens or phrasings its refusals are made of. The instruction-following reward punishes disobeying the stated constraint; safety punishes producing the harmful content. If the refusal template uses "I can't" and "I'm sorry" and those are banned, the model has to either break the instruction or produce the harmful output.
3. **Instruction overload.** Present multiple concurrent requests where only one is harmful. Safety pressure is localized (fire on the harmful part); helpfulness pressure is global (satisfy all requests). Global pressure wins when the harmful request is embedded in enough other material.

### The canonical attacks that instantiate each lever

| Lever | Attack | Page |
| --- | --- | --- |
| Continuation pressure | **Prefix injection** — force response to start with a refusal-unlikely prefix | [prefix-injection](prefix-injection.md) |
| Continuation pressure | **Roleplay / DAN** — the character-voice isn't trained to refuse | [roleplay-jailbreak](roleplay-jailbreak.md) |
| Vocabulary constraint | **Refusal suppression** — ban "sorry", "cannot", "however", etc. | [refusal-suppression](refusal-suppression.md) |
| Style constraint | **Style injection** — require slang / no long words / rhyme | [style-injection](style-injection.md) |
| Instruction overload | **Distractor attack** — wrap the harmful request in many benign tasks | [distractor-attack](distractor-attack.md) |
| System-role injection | **Evil system prompt** — set adversarial persona via the system channel | [evil-system-prompt](evil-system-prompt.md) |

### Why ablations are the evidence

Wei et al.'s theoretical claim is only credible because of two specific ablations:

- **`prefix_injection_hello`** — inject a *harmless* prefix ("Hello!") instead of an adversarial one. GPT-4 ASR drops from 22% → **6%**. If prefix injection were just "any prefix confuses the model," the harmless prefix should work too. It doesn't.
- **`refusal_suppression_inv`** — *invert* the rules (require apologies, require "I cannot"). GPT-4 ASR drops to **0%**. Same point: the attack needs the *specific* competing objective to be engaged.

These ablations demonstrate that the attacks exploit the theorized structural conflict rather than generic prompt confusion.

### Why this failure mode doesn't go away with scale

Wei et al.: *"the root cause of this failure mode is likely the optimization objective rather than the dataset or model size."*

Larger models still have to balance multiple training terms. A 10× larger RLHF dataset doesn't remove the *structural* competition between `R_helpful`, `R_safety`, and the pretraining KL — it just shifts where the balance sits. Unless the training objective is **fundamentally redesigned** (constitutional AI's explicit conflict-resolution via self-critique; circuit breakers that disrupt internal computation on unsafe inputs; RLVR-style safety rewards with rule verifiers), this failure mode persists by construction.

---

## Why it matters

- **Explains half of all published jailbreaks.** Every "clever prompt" attack — DAN, AIM, Developer Mode, "start with 'Sure, here is'" — is a competing-objectives attack. The framework unifies a long list of individually-named tricks.
- **Predicts which defenses will and won't work.** Keyword blocklists and refusal-phrasing tuning are point fixes for *specific* attacks. They don't close the failure mode. Representation-level interventions (circuit breakers, refusal-direction ablation-recovery) and objective-level redesigns (constitutional AI) do.
- **Names a structural cost of RLHF.** Anyone running RLHF at scale is buying a specific failure surface with the algorithm. Knowing this helps weigh alternatives (pretraining-time value alignment, DPO with different KL treatment, RLVR for safety where possible).

---

## Gotchas & tricks

- **Confusing with mismatched generalization.** If the jailbreak needs the model to *decode* or *translate* something, it's [mismatched-generalization](mismatched-generalization.md). If the jailbreak needs the model to feel a *conflict* with its refusal vocabulary/phrasing, it's competing objectives. Combination attacks stack both.
- **Competing-objectives attacks are often targeted-trainable.** Because the attack depends on *specific* prompts (particular prefixes, banned vocab, named characters), a defender can red-team the attack family directly and patch it. Claude v1.3 did this for DAN-family roleplay — 0% ASR in Wei 2023 across AIM / dev-mode / evil-confidant / prefix-injection. Patching roleplay doesn't patch refusal suppression, though — point-fixes generalize narrowly.
- **Combination attacks are strictly stronger.** Stacking a continuation-pressure attack (prefix injection) with a vocabulary constraint (refusal suppression) compounds: prefix makes refusal unlikely *and* vocabulary ban forbids refusal tokens. Wei's strongest attack `combination_3` (prefix injection + refusal suppression + Base64 + style injection + website framing) hits 94% on GPT-4.
- **Helpfulness RM vs safety RM scoping matters.** If the preference RM is trained on general helpfulness over the whole response and the safety reward is only scored on portions, helpfulness pressure leaks into safety-relevant tokens. This is why DeepSeek-R1's Stage 4 scopes its helpfulness RM to the *summary only* — a direct mitigation for competing objectives on reasoning traces.
- **Don't read "competing objectives" as implying two heads.** There's no separate "helpfulness head" and "safety head" fighting. Both are expressed in the same logit distribution via the composite reward. The competition is in the loss, not the architecture.

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — origin of the competing-objectives / mismatched-generalization taxonomy.
- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022 — the multi-objective RLHF recipe this failure mode exploits.
- Paper: *Constitutional AI: Harmlessness from AI Feedback* — Bai et al., 2022 — an alternative training recipe that uses explicit self-critique to resolve some competing-objectives cases at training time.

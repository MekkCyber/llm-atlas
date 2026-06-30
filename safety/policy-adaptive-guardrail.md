# Policy-Adaptive Guardrails (SingGuard)
*Depth — multimodal moderation models that take the safety policy as a runtime input.*

**TL;DR:** Most guardrails bake a fixed taxonomy into the weights and have to be retrained when product/region/deployment-stage rules change. Policy-adaptive guardrails treat the **active safety policy as a prompt-level argument**: given natural-language rules, the model checks content rule-by-rule and emits both a safety label and the triggered rule. SingGuard further runs along a **fast-to-slow reasoning spectrum** — direct judgment for easy cases, policy-grounded deliberation for hard ones — trained with a fast–slow decoupled RL objective.

**Prereqs:** [_attacks](../safety/_attacks.md), [safety README](../safety/README.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md), [_rl](../post-training/_rl.md)

---

## What it is

A multimodal guardrail family where the safety policy is *not* compiled into weights but provided per-call as natural-language rules. The model is trained to (a) parse and apply arbitrary rule sets, (b) emit a safety label plus the index of the rule that triggered the decision, and (c) deliberate proportionally to risk.

## How it works

**Runtime contract.** Input = (active policy rules, content under review). Output = (label ∈ {safe, unsafe, …}, triggered_rule_id, optional reasoning trace).

**Fast / hybrid / slow inference.** A single model exposes three regimes:

- *Fast*: direct label, no deliberation — used when content is clearly in/out of policy.
- *Hybrid*: short reasoning trace tied to candidate rules.
- *Slow*: full rule-by-rule policy-grounded deliberation — used when fast/hybrid confidence is low.

**Fast–slow decoupled RL.** Reward signals are computed independently for fast and slow tracks, so the model is rewarded both for correct snap judgments *and* for correct deliberated judgments. Coupling these (single reward) collapses one mode into the other in practice.

**Bench.** SingGuard-Bench: 56,340 examples spanning 80+ fine-grained risk types — built specifically to stress the policy-adaptive interface.

## Why it matters

- **Policy drift is the production reality.** Region, product surface, age-gating, and regulatory rules all shift mid-deployment. Recompiling a guardrail per change is unworkable.
- **Adaptive compute** for moderation parallels adaptive-compute reasoning for agents — most calls are easy and shouldn't pay for deliberation, but the hard tail must.
- **SOTA average F1** across 6 benchmark families (35 datasets), beating fixed-taxonomy peers and frontier-API moderation.

## Gotchas & tricks

- Quality of natural-language rules dominates evaluation: ambiguous rules silently degrade triggered-rule accuracy.
- Fast-slow decoupling is a *training* trick, not just an inference one — co-training a single track and inferring "deliberate when uncertain" underperforms.
- Adversaries can attack the policy parser (rule-injection) as well as the content; treat the policy channel as untrusted input when it comes from less-privileged tenants.

## Sources

- Paper: *SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning* — arXiv:2606.22873 — https://arxiv.org/abs/2606.22873

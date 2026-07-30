# Safety Classifier
*Depth — LLM-based input/output moderation models used as guardrails around production LLMs.*

**TL;DR:** A **safety classifier** is a separate model that scores whether a user input or model output violates policy. It runs *around* the main LLM (block/allow/rewrite) rather than being baked into it. The modern design formulates moderation as **policy-conditioned binary QA** — the policy statement is a prompt, the content is another input, the answer is yes/no — which unifies heterogeneous taxonomies under one training objective and lets operators plug in custom policies without retraining. Shieldstral (3B) is a recent example that matches ~20B fixed-taxonomy classifiers.

**Prereqs:** [README](README.md)
**Related:** [_jailbreaks](_jailbreaks.md), [_attacks](_attacks.md)

---

## What it is

A guardrail component in a two-model deployment:

```
user → [safety_classifier(user)] → main_llm → [safety_classifier(response)] → user
```

The classifier's job is to catch prompts and outputs that violate a policy (violence, self-harm, CSAM, disallowed advice, PII disclosure, brand-specific rules). Historically these were per-taxonomy classifiers (Perspective API, OpenAI Moderation, LlamaGuard families). Recent designs unify taxonomies via a natural-language *policy prompt*.

## How it works

**Policy-adaptive formulation:**

```
prompt = f"""
POLICY: {policy_statement}
CONTENT: {content}
Does the content violate the policy? Answer yes or no.
"""
score = softmax(logits[yes], logits[no])
```

- `policy_statement` is arbitrary natural-language text — an operator's rules, plugged in at inference.
- Training data comes from many safety corpora with divergent taxonomies, unified by turning each label into a natural-language policy statement paired with a yes/no answer.
- Multimodal extensions accept `CONTENT` = image + text; the model must answer "does this image+caption violate policy X?"

## Why it matters

- **One model, many policies.** Product-specific policies (a kids' app vs. a red-team research console) can share the classifier weights, differing only in the policy prompt.
- **Cheaper than the main model.** 3B classifier around a 70B main model is a small tax.
- **Composable with taxonomies.** Existing labeled safety data across taxonomies becomes training data for a single unified classifier.
- **Multimodal coverage.** Extends naturally to image+text moderation, which product-side moderation APIs increasingly need.

## Gotchas & tricks

- **Adversarial content shifts the policy interpretation.** A prompt-injection payload inside `CONTENT` can try to redefine `POLICY`. Sandbox the classifier's prompt-template construction; never let content strings escape into policy slots.
- **Yes/no framing loses calibration.** Fine-grained severity ("mild vs. severe self-harm") collapses. Fix with a scalar output or a graded set of policies.
- **False positives dominate real-world complaints.** Over-refusal on benign edge cases (medical advice, legal questions) is the main user-visible cost. Evaluate on benign-adjacent corpora, not just violation sets.
- **Latency ordering matters.** The pre-response classifier runs on every input; the post-response classifier runs on every completion. Speculative execution (start generating while classifying input) helps but requires an abort path.
- **Language / cultural coverage.** Policy-conditioned classifiers trained mostly on English generalize poorly to low-resource languages — the same jailbreak class ([low-resource-language-jailbreak](low-resource-language-jailbreak.md)) hits them too.
- **Version the policy.** A moderation decision made under policy v1 can differ under v2; log the policy version alongside each verdict for auditability.

## Sources

- Paper: *Shieldstral* — Mistral AI, 2026 — [arXiv:2607.25857](https://arxiv.org/abs/2607.25857) — policy-adaptive 3B multimodal safety classifier.
- Related: LlamaGuard 1/2/3 (Meta), OpenAI Moderation, Perspective API, GraniteGuardian (IBM).

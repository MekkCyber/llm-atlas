# HPSE: Hybrid-Policy Self-Editing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A knowledge-editing method that makes injected facts **composable** — usable in atomic questions and multi-hop reasoning, not just recallable verbatim. Frames editing as **self-distillation from a privileged in-context state**, and closes the pure-on-policy coverage gap with a **hybrid rollout** that splices missing facts into the student's own trajectory precisely where its coverage fails.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [dpo](dpo.md), [grpo](grpo.md)

---

## What it is

Unstructured knowledge editing (UKE) injects a **free-form passage** into a model that may state multiple facts at once (a paragraph, a Wikipedia snippet). Existing UKE editors succeed at literal recall — the edited model can regenerate the passage — but fail at **composability**: the model can neither answer atomic questions about individual facts in the passage, nor use them in multi-hop reasoning.

HPSE names composability as the missing property and traces it to a design flaw. Existing editors treat the passage as the sole learning source, so the model never learns to *use* the facts — it just memorizes their surface form. HPSE recasts editing as **proactive self-distillation** from a privileged version of the same model (with the passage in context) into the deployed model (without context).

## How it works

**Setup.** For each new passage $p$:

- **Teacher** = the same model with $p$ prepended to context (privileged access to the fact).
- **Student** = the model that will end up deployed (no context).
- The goal is to distill teacher behavior on questions about $p$ into the student.

**The pure-on-policy problem.** Standard on-policy distillation samples from the student and matches teacher logits. But because the injected knowledge is novel, the student's own rollouts **rarely visit** states where the fact would matter — the training signal never lands on the tokens that actually need updating.

**Hybrid rollout.** HPSE builds a mixed trajectory:

- Where the student's rollout would already engage with the fact (rare), stay **on-policy**.
- Where student coverage fails, **splice in** the missing fact from the teacher's rollout at the correct trajectory position — an off-policy patch aimed at the coverage gap.

The result is a training trajectory that (a) exercises the fact frequently enough to actually update behavior, (b) stays close enough to the student's own distribution that the distillation is stable.

**Theoretical claim.** The paper proves HPSE strictly dominates pure on-policy distillation in the regime where teacher rollouts contain much more fact-use than student rollouts — exactly the regime knowledge editing cares about.

## Why it matters

- **Names a real failure mode** — composability — that ships in every current UKE method and gets papered over by literal-recall metrics.
- **Plug-and-play uplift.** Improvements across **4 LLM backbones × 2 KE editors** — the mechanism is orthogonal to the underlying editor architecture (locate-and-edit, hypernetwork, etc.).
- **Reusable primitive.** Hybrid-rollout distillation appears whenever a student policy is undertrained on rare-but-important states — refusal training, tool-use, long-tail reasoning. Not just knowledge editing.

## Gotchas & tricks

- **Splice position matters.** Where you inject the missing fact into the student's rollout determines whether the update lands. Too early: student can't use it. Too late: the derivation has already gone wrong.
- **Requires two model instances.** The teacher (student + context) and student both need to be forward-passable simultaneously. Memory footprint doubles at edit time.
- **Passage boundary matters.** A short passage yields cleaner splice positions; multi-paragraph passages need per-fact segmentation to know what to splice where.
- **Not a substitute for retrieval.** For truly rare facts that will only be queried once, RAG remains cheaper. HPSE pays off when a fact will be used repeatedly and needs to compose with other model knowledge.

## Sources

- Paper: *Hybrid-Policy Self-Editing for Composable Unstructured Knowledge Editing* — Tianci Liu, Zihan Dong, Tianchun Li, Yi-Chung Chen, Qiming Cao, Xingchen Wang, Shiyang Wang, Zichen Miao, Linjun Zhang, Haoyu Wang, Jing Gao, 2026 — [arXiv:2608.11660](https://arxiv.org/abs/2608.11660).

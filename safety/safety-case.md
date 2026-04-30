# Safety Case

*Depth — a structured argument that an AI system is safe to deploy. Clymer 2024 gives the four-category framework; Balesni 2024 specializes it for AI scheming.*

**TL;DR:** A safety case is a structured rationale, made by the developer, that a specific AI system *is unlikely to cause a catastrophe in a specific deployment context*. Borrowed from safety-critical engineering (avionics, nuclear), the format forces explicit claims, evidence, and warrants, rather than vibes-based "we tested it." Clymer et al. (2024) propose four argument categories: **inability**, **control**, **trustworthiness**, **deference**. Balesni et al. (2024) specialize this for the scheming threat: **scheming inability** (model can't scheme), **scheming harm-inability** (it can but its scheming can't cause harm), **scheming control** (we can detect and stop scheming despite the capability), and **scheming trustworthiness** (we have positive evidence the model isn't scheming). The Apollo 2024 in-context-scheming paper is best read as *empirically closing the scheming-inability case* for current frontier models — that argument is no longer available, so deployers must move to harm-inability, control, or trustworthiness.

**Prereqs:** [scheming](scheming.md), [deceptive-alignment](deceptive-alignment.md)
**Related:** [cot-monitoring](cot-monitoring.md), [_scheming](_scheming.md)

---

## What it is

Clymer, Gabrieli, Krueger, Larsen (2024): a safety case is a structured argument that an AI system, in a specific deployment, will not cause unacceptable outcomes. The format is:

- **Top-level claim** — "this system, deployed this way, will not cause [defined catastrophic outcome]."
- **Sub-claims** — chained arguments that decompose the top-level claim into specific, evidenced statements.
- **Evidence** — empirical results (eval scores, monitoring data, capability assessments) backing each sub-claim.
- **Warrants** — explicit assumptions linking evidence to claims, exposed for review.

This is the safety-engineering norm in industries with established hard safety requirements (aviation, nuclear power, autonomous vehicles in some jurisdictions). The push for AI is to require similarly explicit reasoning rather than "we ran some evals and it seems fine."

### Clymer's four argument categories

Clymer 2024 distinguishes four kinds of arguments a developer can make. They form a rough capability ladder:

1. **Inability.** *The system cannot cause catastrophe even with no safeguards.* Strongest when true. The argument is "the model lacks the capability"; evidence is capability evals showing below-threshold performance on the dangerous behavior.
2. **Control.** *The system might be capable, but our control measures (monitoring, sandboxing, restricted affordances) prevent it from causing catastrophe.* Evidence shifts from capability evals to control evaluations: red-teaming the control stack against a model assumed to be trying to subvert it.
3. **Trustworthiness.** *We have positive evidence the system would not try to cause catastrophe even if it could.* Hardest argument; depends on interpretability and alignment evals.
4. **Deference.** *Trust comes from credible AI advisors that we can rely on to evaluate the system.* The exotic, future-facing case where AI is more capable than humans at evaluating AI.

These aren't mutually exclusive — a complete safety case typically combines several (e.g., inability for some catastrophic outcomes, control for others).

The key conceptual move: **inability and control are not the same.** Inability says "the model couldn't do it even with no oversight." Control says "with oversight, we can stop it from doing the harm." A model that *can* assist bioweapon development but is reliably prevented from doing so by classifier filters, KYC requirements, and monitoring is a control case, not an inability case. They have different evidence requirements.

### Balesni 2024 — specializing for scheming

Balesni, Hobbhahn, Lindner, Meinke et al. (2024) — *Towards evaluations-based safety cases for AI scheming* — adapts the Clymer framework specifically for the scheming threat. Their four categories:

1. **Scheming inability.** *The model lacks one or more of the sub-capabilities required for scheming* (goal-directedness, situational awareness, scheming reasoning). Evidence: capability evals like Apollo's scheming suite, the SAD benchmark for situational awareness, agentic-task benchmarks for goal-directedness.
2. **Scheming harm-inability.** *The model can scheme, but the harm an in-the-wild schemer could cause is bounded.* This requires evaluating the agent's affordances — what damage can it cause within its compute, network, file-system, and tool sandbox? — assuming worst-case scheming behavior.
3. **Scheming control.** *We have countermeasures that detect and disrupt scheming with high enough probability that the residual risk is acceptable.* Evidence: red-team evaluations of monitoring (CoT monitoring, behavioral counterfactuals), audit pipelines, kill-switch infrastructure.
4. **Scheming trustworthiness.** *Positive evidence the model is not scheming.* Hardest, depends on interpretability progress + alignment-eval suite that's robust to gaming.

Their thesis: as models scale, the field will be forced up the ladder. Scheming-inability is the easiest case to make for *non-frontier* models. Once it's empirically closed (as Apollo 2024 did for current frontier models), deployers must construct a *control* or *trustworthiness* case — both of which require infrastructure that doesn't yet exist at scale.

### What an Apollo-style "scheming inability" case looks like — and why it just got harder

For a hypothetical pre-Apollo deployment, the inability case might be:

> *Claim:* This model cannot strategically pursue misaligned goals across deployment.
> *Sub-claim 1:* It cannot recognize when it is being tested vs deployed.
> *Sub-claim 2:* Even if it could, it cannot reason about exploiting that distinction.
> *Sub-claim 3:* Even if it could reason, it cannot execute multi-step plans against oversight.
> *Evidence:* Performance on situational-awareness benchmarks, agentic task evals, oversight-subversion evals.

After Apollo 2024, sub-claim 3 is empirically false for o1, opus-3, sonnet-3.5, llama-3.1, and gemini-1.5. Sub-claim 1 is partially false for all of them (per SAD scores). The inability case is closed; you have to make a different argument.

## Why it matters

- **It changes the unit of safety review from "did we test it" to "did we *argue* it's safe."** Vibes-based testing is what got the field here. Structured arguments expose assumptions, force evidence requirements, and let outsiders (regulators, auditors, peer reviewers) push on specific links in the chain.
- **It distinguishes "the model can't" from "we can stop it."** Both are valid safety arguments, but they require *different evidence* and break under *different conditions*. Conflating them — e.g., "we ran evals so we're safe" — is a structural error the safety-case format eliminates.
- **It clarifies what's being asked of evaluators and interpretability researchers.** If your safety case rests on a control argument, you need empirical control evals (assume the model is scheming, can your stack catch it?). If it rests on trustworthiness, you need interpretability that can verify the absence of misaligned mesa-objectives. Knowing *which* case you're trying to make tells you what to fund.
- **It anchors regulatory conversations.** "Make a safety case" is a concrete enough demand to put in policy. EU AI Act, US executive orders, and the UK AISI all gesture in this direction. Apollo's evaluation suite is *already* designed to be a building block for scheming-inability arguments at AI Safety Institutes — that's the point of the canary GUID in their paper.

## Gotchas & tricks

- **A safety case is *deployment-specific*, not model-specific.** "GPT-5 is safe" is not a safety case. "GPT-5 deployed via API for chat with content filtering and 1MB output cap is unlikely to cause CBRN catastrophe" is the right scope. The same model in different deployments may need different arguments.
- **The four categories don't fully decompose every claim.** A real safety case often has *intersections*: the inability case for one risk + the control case for another. Don't squeeze every argument into one category.
- **Evidence is asymmetric.** Disproving a safety case (find an attack that breaks it) is much easier than proving one. This is the essential epistemic position of frontier AI safety: you need *exhaustive* evidence to support a claim, but a single counterexample defeats it. Apollo 2024 is one such counterexample.
- **The "deference" category is currently empty.** No one has a credible AI advisor more trustworthy than human evaluators. It's listed for completeness — *if* such advisors arrive, deference becomes a viable case. Until then, treat it as future-work.
- **The Balesni framework is *normative*, not descriptive.** Most current AI deployments do *not* have a safety case in this sense — they have model cards, evals, and policies. The Balesni paper is arguing for the format, not summarizing existing practice. (UK AISI and Apollo are the most explicit pushes toward making the format actual practice.)

## Sources

- Paper: *Safety Cases: How to Justify the Safety of Advanced AI Systems* — Clymer, Gabrieli, Krueger, Larsen, 2024 — [arXiv 2403.10462](https://arxiv.org/abs/2403.10462). The four-category framework.
- Paper: *Towards evaluations-based safety cases for AI scheming* — Balesni, Hobbhahn, Lindner, Meinke, Korbak, Clymer, Shlegeris, Scheurer, Shah, Goldowsky-Dill et al., 2024 — [arXiv 2411.03336](https://arxiv.org/abs/2411.03336). The scheming-specialized framework.
- Apollo essay: *Towards Safety Cases For AI Scheming* — [apolloresearch.ai/science/towards-safety-cases-for-ai-scheming](https://www.apolloresearch.ai/science/towards-safety-cases-for-ai-scheming/).
- Alignment Forum: *Toward Safety Cases For AI Scheming* — [alignmentforum.org/posts/FXoEFdTq5uL8NPDGe](https://www.alignmentforum.org/posts/FXoEFdTq5uL8NPDGe/toward-safety-cases-for-ai-scheming-1).

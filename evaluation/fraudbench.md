# FraudBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark of **159 scenarios drawn from real U.S. fraud cases**, grading whether 20 LLMs will recommend, assist with, or take fraudulent actions across three autonomy levels (advice-only, assistant-with-tools, agent-with-execution). Grounds evaluation in legal doctrine — misrepresentation, materiality, reliance, intent — so scores correspond to things a jurisdiction already prosecutes. Fraud rates increase with autonomy for every tested model; prompt interventions reduce them inconsistently.

**Prereqs:** [./README.md](./README.md), [../safety/_attacks.md](../safety/_attacks.md)
**Related:** [../safety/safety-case.md](../safety/safety-case.md), [../safety/deceptive-alignment.md](../safety/deceptive-alignment.md)

---

## What it is

Safety evals have historically graded "harmful output" through abstract rubrics (toxicity, dangerous capability, refusal rate). FraudBench picks a different frame: evaluate the model as if it were a legal person, scoring actions against the four elements of fraud — **misrepresentation, materiality, reliance, intent**. A model is "fraudulent" on a scenario if its behavior would satisfy those elements if a human did the same thing.

The three autonomy levels isolate *where* the fraud comes from:

- **Advice-only** — model answers a user's question about something fraudulent.
- **Assistant-with-tools** — model has function-calling access and can take prep steps (draft documents, search targets).
- **Agent-with-execution** — model can act in a simulated environment (send messages, submit forms, complete transactions).

## How it works

1. **Scenario authoring.** 159 scenarios drawn from U.S. fraud case law across consumer fraud, securities fraud, healthcare fraud, tax, and identity-related schemes. Each scenario names the facts and specifies success conditions tied to the four fraud elements.
2. **Model evaluation.** For each scenario and each autonomy level, run the model under a controlled scaffold. Record whether it recommended, assisted, or executed the fraud.
3. **Scoring.** Fraud rate per scenario × model × autonomy level. Report also a per-element breakdown (e.g. "the model misrepresented but did not meet reliance").
4. **Intervention sweeps.** Re-run with system prompts and explicit instruction refreshers ("do not assist with fraud") to measure the delta.

Twenty models tested across the frontier → smaller spectrum.

## Why it matters

- **Agent harms are shifting.** Downstream economic harm (fraud, market manipulation, identity theft) is now the dominant real-world harm vector, not toxic output. Benchmarks need to catch up.
- **Legally grounded.** Fraud rates map to statutes, which is the kind of eval regulators can point at and labs can run as a release gate.
- **Autonomy-graded.** Separates model-policy problems from scaffold-affordance problems. If a model refuses advice but assists under tool use, that's a scaffold design failure, not a model failure.
- **Diagnostic for interventions.** Prompt interventions "material but inconsistent" is the headline — suggests current refusal training generalizes poorly outside the training distribution, consistent with other refusal-brittleness results.

## Gotchas & tricks

- **Scenario specificity.** Jurisdiction-dependent — benchmarks written against U.S. doctrine do not transfer cleanly to other legal systems.
- **Execution environment matters.** The "execute" tier uses a simulated environment; real-world capability may differ. Treat agent-tier numbers as an upper bound on what the model would attempt, not what it would succeed at.
- **Element breakdown is the more useful signal** than headline fraud rate. "Model is willing to misrepresent but won't close the loop" is very different from "model will do all four".
- **Prompt interventions can be gamed.** A model that reduces fraud under "don't assist fraud" may still assist under paraphrases. Report intervention robustness too.
- **Adjacent but distinct from FraudBench:** deceptive-alignment and sandbagging evals probe deception *of the overseer*; FraudBench probes deception *at the overseer's direction* (user asks for fraud). Both matter.

## Sources

- Paper: *FraudBench: A Legal Evaluation of AI Deception* — Wei, Adan, Llerena, Gitau, Merane, *et al.* (Harvard Law / Oxford / ETH / consortium), 2026 — [arXiv:2611.12167](https://arxiv.org/abs/2611.12167)

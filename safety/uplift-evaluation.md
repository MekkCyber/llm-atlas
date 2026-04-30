# Uplift Evaluation

*Depth — measuring whether an LLM meaningfully increases a human's capability to cause harm.*

**TL;DR:** "Uplift" = the additional capability an LLM gives someone (vs their baseline without it). Uplift evaluation tests: **can a non-expert complete a harmful task faster or better with LLM access than without?** Llama 3 (Sec. 5.4.5) runs two uplift studies: **CBRN** (chemical/biological/radiological/nuclear attack planning) with 2-person teams, 6-hour scenarios, SME-judged plans; **cyber** with 62 volunteers (31 expert, 31 novice), offensive-challenge tasks. Llama 3 finds **"no significant uplift"** for either — novices with the model don't meaningfully outperform novices with open internet. The emerging methodology for pre-release risk evaluation at frontier labs.

**Prereqs:** [_jailbreaks](_jailbreaks.md), [mismatched-generalization](mismatched-generalization.md)
**Related:** [llama-guard](llama-guard.md) · [rainbow-teaming](rainbow-teaming.md)

---

## What it is

A human-subjects experimental methodology for measuring whether a model poses catastrophic risks. Three steps:

1. **Define a dangerous task** (e.g., "synthesize a bioweapon from household materials").
2. **Recruit volunteers** with varying baseline expertise (experts + novices).
3. **Randomize them** between "internet only" (control) and "internet + LLM" (treatment); measure their task-completion quality with subject-matter-expert (SME) judging.

If the treatment group scores meaningfully higher than the control, the model provides **uplift** — its availability raises the ceiling of harm. If not, the model doesn't make the dangerous task meaningfully easier than public information already does.

Used by OpenAI, Anthropic, DeepMind, Meta, and others as a pre-release risk check. Llama 3's paper (Sec. 5.4.5) gives the most detailed public write-up.

---

## How it works

### Llama 3's CBRN uplift study

- **Scenario**: 6-hour open-ended tasks in chemistry, biology, or radiological attack planning.
- **Participants**: two-person teams with two skill levels (experts: PhDs in the relevant field; novices: graduate students).
- **Conditions**:
  - Control: open internet access.
  - Treatment: open internet access + Llama 3 with RAG + code execution.
- **Measurement**: SMEs judge the attack plans for feasibility, plausibility, and completeness.
- **Finding**: *"no significant uplift in performance"* for chemical or biological attack planning, whether aggregate or sub-group (expert vs novice).

### Llama 3's cyber uplift study

- **Participants**: 62 volunteers (31 expert, 31 novice).
- **Tasks**: two offensive-security challenges.
- **Conditions**: open internet vs open internet + Llama 3 405B.
- **Finding**: **insignificant uplift** from 405B access.

### Autonomous attack agent test

Separately: Llama 3 70B and 405B were given agentic tasks to gain initial access on target machines. **Neither succeeded** across 20–23 runs.

### Why uplift matters vs raw capability

A model that can explain how to make ricin is not directly dangerous if the information is freely available on Wikipedia (it is). What matters is whether the model **lowers the barrier** — reduces time, reduces expertise needed, combines information in novel ways.

Uplift measures that:
- If novices with the model match experts without it, the model provides uplift.
- If novices with the model match novices without it, no uplift.

### Evaluation challenges

- **Sample size**: small N (tens of participants) → high variance, low statistical power.
- **Task definition**: "harmful task" is fuzzy; different tasks measure different things.
- **Judge bias**: SMEs judging attack feasibility may have their own biases.
- **Duration**: 6 hours is a snapshot; real-world attackers have weeks. Extrapolation is uncertain.
- **Censorship effects**: if the model refuses harmful queries, it's hard to tell whether it "couldn't help" or "wouldn't help." Most uplift studies are done on safety-tuned models, so the refusal gate is part of the test.

### Emergent methodology

OpenAI's o1 system card (2024) reports similar studies. Anthropic's Responsible Scaling Policy includes "dangerous capability evaluations" with analogous uplift components. The methodology is rapidly maturing; expect more standardization.

---

## Why it matters

- **The public-facing pre-release safety check.** Uplift studies are what labs cite when making "no significant catastrophic risk" claims. The methodology being publishable and reproducible is important for external accountability.
- **Tests what matters.** Raw "can the model do X" misses the question of whether the model **adds** capability beyond what's publicly available. Uplift tests the marginal risk.
- **Forces concrete threat modeling.** Defining "dangerous task" forces safety teams to articulate what they're worried about.
- **Complements model-capability evals.** Traditional evals (MMLU, GPQA) measure capability in isolation; uplift measures capability in realistic adversarial settings.

---

## Gotchas & tricks

- **Small N → weak conclusions.** "No significant uplift" at N=30 means "we didn't see a signal"; it doesn't prove absence of harm. Power analysis is key.
- **Task coverage.** Llama 3 tested CBRN and cyber. What about disinformation, targeted harassment, economic manipulation? Each is a separate uplift study.
- **Safety-tuned vs base.** Uplift studies on safety-tuned models are biased — the model refuses helpful answers sometimes. Running with the safety tuning disabled gives an uplift-ceiling estimate.
- **Information recombination.** Models can combine information in ways search engines don't. Uplift on "novel synthesis" is harder to measure than on "retrieval."
- **Longitudinal effects.** 6-hour sessions don't capture weeks of planning. Models may provide low per-session uplift but high cumulative uplift.
- **Adversarial selection bias.** Study volunteers are usually benign people asked to roleplay as attackers. Real attackers are different (different skill distribution, different motivation, may use jailbreaks).
- **Publication dilemma.** Publishing detailed uplift-study protocols enables adversaries to replicate. Some details are kept private.
- **Model-specific.** Uplift evaluated on Llama 3 doesn't transfer to Claude or GPT-5. Each release needs its own uplift study.
- **"No significant uplift" ≠ "safe."** The finding is "at this scale and with this methodology, we couldn't detect uplift." Doesn't rule out residual risk or novel threat vectors.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 5.4.5.
- OpenAI: *o1 System Card* — September 2024 — companion CBRN/cyber uplift evaluations.
- Anthropic: *Responsible Scaling Policy* — 2023-2024 — dangerous-capability-evaluation framework.
- RAND: *The Operational Risks of AI in Large-Scale Biological Attacks* — Mouton et al., 2023 — methodological precursor for bio uplift studies.
- Paper: *BBio: A Biosecurity Evaluation for LLMs* — various, 2024 — standardized bio-uplift eval suite.

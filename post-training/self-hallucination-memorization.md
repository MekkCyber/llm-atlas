# Self-Hallucination Memorization
*Depth — reasoning-centric post-training recipes cause LLMs to memorize false facts they generated in their own CoT.*

**TL;DR:** Outcome-based RL (e.g., RLVR) rewards the final answer and leaves everything the model wrote in its CoT to be shaped by the same gradient. Atmakuru & Goyal (2026) show that on knowledge tasks this causes **self-hallucination memorization**: false facts the model wrote during reasoning get reinforced by ~4.83% average recall gain (vs. 0.17% for unrewarded hallucinations), and the model becomes ~7.23% less likely to abstain from them later. The effect persists even when the CoT later retracts the false claim.

**Prereqs:** [rlvr](rlvr.md), [_rl](_rl.md)
**Related:** [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [../safety/cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

A failure mode of applying reasoning-centric post-training recipes (outcome rewards, injected cognitive behaviors like backtracking) to **knowledge tasks** — tasks where correctness depends on retrieving true facts, not deriving them. Documented across six open models (Qwen3 and OLMo/OLMoE families) on QuALITY and TriviaQA.

## How it works

Experimental setup:

1. Take a base model and a reasoning-focused post-training recipe (RLVR with outcome rewards, optionally injected backtracking).
2. Have the model answer knowledge questions with CoT; label each intermediate factual claim as true or false (automated + spot-checked).
3. Apply the recipe: only the final answer is rewarded. CoTs leading to correct final answers get positive signal; the intermediate hallucinations in those CoTs ride along.
4. After training, measure:
   - **Recall** of the specific hallucinated facts (does the model produce them again when prompted?).
   - **Abstention rate** on the same facts.

Findings:

- Rewarded hallucinations: +4.83% recall on average.
- Unrewarded hallucinations (same base, no gradient on their trace): +0.17% recall.
- Retracted hallucinations (the CoT explicitly rejects the false claim later) are memorized **about as strongly** as unretracted ones.
- Abstention on rewarded hallucinations drops by ~7.23%.

## Why it matters

A concrete critique of the "apply RLVR everywhere" trend. Outcome rewards cannot distinguish truth of intermediate claims, so any knowledge-domain reasoning RL risks promoting whatever the model made up on the way to a correct answer. The retraction result is the sharpest edge: **the model does not learn to doubt its own prior hallucinations just because the CoT later retracts them.** This has direct implications for:

- **Safety / factuality.** A reasoning-trained model becomes more confident in its own fabrications.
- **CoT monitoring.** Retractions in a monitored CoT may not reflect the policy actually having updated against the retracted claim.
- **Multi-stage pipelines.** Hallucinations produced in an earlier stage can be baked in by RL optimizing later stages.

## Gotchas & tricks

- **Mitigations tested are partial.** Reward-shaping that penalizes false intermediate claims helps, but requires a verifier for factuality — expensive on open knowledge.
- **Model scale doesn't fix it.** The effect appears in the 7B+ MoE family tested as well as smaller models.
- **Scope is knowledge tasks.** Math/code benchmarks have intrinsic verifiers that can catch CoT errors; the pure knowledge setting is the worst case.
- Open to doubt: the exact magnitudes depend on label noise in the hallucination tagger.

## Sources

- Paper: LLMs Memorize Their Own Hallucinations During Post-Training — Atmakuru & Goyal, 2026 — [arXiv:2611.26863](https://arxiv.org/abs/2611.26863)

# Program Judge (PAJAMA)
*Depth — distilling an LLM-as-judge into a committee of synthesized programs that score candidates deterministically, with LLM fallback for low-confidence cases.*

**TL;DR:** Replace the per-sample LLM-judge call with a *committee of programs* — small pieces of code synthesized to encode the judge's decision logic per rubric axis. Programs are deterministic, editable, and free at inference; the committee's aggregated verdict routes low-confidence cases back to an LLM. Match a **13B LLM judge** across five datasets and four model families; **RewardBench** reward model distilled from committee verdicts beats one trained on a proprietary LLM's labels at **two orders of magnitude lower API cost**.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

LLM-as-judge is the current default for automated evaluation and preference labeling, but it's expensive, latent-heavy, and opaque. A programmatic judge — a synthesized program that scores a candidate against a rubric axis — is transparent, deterministic, free to run, and editable. A *committee* of such programs, aggregated with confidence-based LLM fallback, closes the quality gap.

## How it works

Three stages:

1. **Program synthesis.** For each rubric axis, prompt a strong LLM to synthesize a small Python program that consumes `(instruction, response)` and returns a score. Sample multiple programs per axis to form a committee.
2. **Committee aggregation.** At evaluation time, run every program in the committee. Aggregate scores into a joint verdict (majority vote, weighted average, or a small learned aggregator). Attach confidence from committee agreement.
3. **LLM fallback.** When committee confidence is low, escalate the sample to a full LLM judge. Otherwise return the committee verdict directly. The escalation rate is a tunable knob — most samples never touch the LLM.

Because programs are text and deterministic, humans can *read* them, edit them, and audit their decisions — something an opaque LLM judge doesn't allow.

## Why it matters

- **Cost.** Committee runs are essentially free relative to LLM API calls; the LLM budget is spent only on escalations.
- **Latency.** Deterministic programs are milliseconds; LLM judges are hundreds of milliseconds to seconds.
- **Transparency.** You can see and edit the program.
- **Reward signal.** Program verdicts double as cheap preference labels — a reward model distilled from them beats one distilled from proprietary LLM labels on RewardBench at two OOM lower API cost.

## Gotchas & tricks

- The committee's failure modes are *shared* if programs are sampled from the same LLM under the same prompt — vary the prompt to get useful diversity.
- Escalation threshold interacts with cost/accuracy tradeoff; tune per deployment.
- Programs generalize poorly to task types they weren't synthesized for — plan for periodic re-synthesis as the eval distribution shifts.
- Programs are editable *and* auditable; treat this as a feature — human review of the committee is a real safety story LLM judges can't offer.

## Sources

- Paper: *Codifying the Judge: Scalable Evaluation via Program Distillation* — Huang, Qiu et al., 2026 — [arXiv:2607.22561](https://arxiv.org/abs/2607.22561)

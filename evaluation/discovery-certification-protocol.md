# Discovery Certification Protocol (DCP)

*Depth — a falsifiable protocol for auditing "discovery" claims by AI research agents: raw benchmark scores don't count, executable recovery + feedback tests do.*

**TL;DR:** A benchmark score doesn't distinguish *discovery* from *recovery* — an agent may have surfaced a known answer from its priors, public sources, or web content rather than actually discovering it. **DCP** replaces the leaderboard with a protocol of executable gates: **Gate 1** confirms useful improvement on sealed eval; **Gate 2** gives matched agents the registered starting info and the observed web content but withholds the target research history, and any recovery of the target voids the claim; **Gate 3** measures the average effect of truthful feedback vs. a neutral baseline. A deterministic, LLM-free verifier reproduces every decision from frozen evidence.

**Prereqs:** [../data/decontamination.md](../data/decontamination.md), [../safety/safety-case.md](../safety/safety-case.md)
**Related:** [README](README.md), [saescientist-bench](saescientist-bench.md)

---

## What it is

As "AI research agent" claims proliferate — an agent that finds a better SQL query, a novel catalyst, a faster kernel — the community lacks a shared language for *what a discovery even is*. DCP defines it operationally: an agent's output is a discovery only if adequately-informed matched agents *cannot* recover the same output from public sources and starting info, and the effect is robust under truthful feedback.

The protocol has three gates and a Core veto:

- **Gate 1 (usefulness).** Sealed evaluation confirms the claim improves a pre-registered metric by a pre-registered margin.
- **Gate 2 (non-recoverability).** Matched agents receive the registered starting information and observed Web content, but *not* the target research history. Every recovery witness (a valid method that reaches the target) triggers **DCP Core**, which requires: adequate controls, zero observed recoveries, and a finite-sample bound on recovery in one fresh registered episode.
- **Gate 3 (feedback effect, optional).** Measures the average effect of truthful feedback relative to a specified neutral policy from a shared checkpoint. **DCP Evidence** adds this effect after independent null calibration and a registered effect margin.

## How it works

The audit runs registered *episodes* — matched agent runs with fixed seeds, tools, starting information, and observation logs — enough to bound the recovery probability. Each episode ends with a machine-checkable decision: recovered / not-recovered / audit-incomplete. A deterministic verifier consumes the frozen evidence and reproduces the audit outcome, so protocol conformance can be re-checked without re-running any LLM.

Two controlled audits accompany the paper:

- **SQLite optimization** — zero recoveries in 96 episodes; upper bound 0.0468.
- **Virtual catalyst control** — zero recoveries in 96 episodes; upper bound 0.0468.
- Each paired study yielded 30 truthful-feedback recoveries and zero neutral-feedback recoveries under Gate 3.

## Why it matters

- **Falsifiability for AI research claims.** The analogue of a pre-registered study: what would falsify this? DCP names it.
- **Independent of any LLM judge.** The verifier is deterministic; disputes go to frozen evidence, not to another model.
- **Complements score-based benchmarks.** [Φ-Bench](phi-bench.md), [SWE-Bench Pro Verified](swe-bench-pro-verified.md), and [SAEScientist-Bench](saescientist-bench.md) measure *ability*; DCP measures whether the produced *artifact* is actually a discovery.

## Gotchas & tricks

- **Registration is load-bearing.** Post-hoc audits are meaningless — starting information, web-content snapshot, and evaluation must be registered before matched-agent runs begin.
- **"Adequately informed" is a judgment call.** Give matched agents too little and Gate 2 trivially passes; too much and no real discovery ever survives. The registration must include this level of care.
- **The Core bound is not a proof of novelty.** It bounds recoverability under the registered episode class, not over all possible strategies. Broader claims need broader episode classes.
- **Not a benchmark to beat.** DCP is a protocol; there is no leaderboard. Individual audits produce individual certifications.

## Sources

- Paper: *Scores Alone Do Not Prove Discovery: The Discovery Certification Protocol for Auditing AI Research Agents* — Ning, Zhong, Li, Zeng, 2026 — [arXiv:2609.09219](https://arxiv.org/abs/2609.09219).

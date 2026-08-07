# PIMiner — Agentic Prompt-Injection Red-Teaming

*Depth — a training-time strategy-library agent that generalizes to unseen target LLMs at test time with ~10 queries per sample.*

**TL;DR:** Existing prompt-injection red-teamers are RL-trained attacker models that generalize poorly to new targets — you re-train for every new defender. PIMiner (Wang, Yin, Geng, Jia, 2026, Penn State) is instead an agentic system that builds a **strategy library** during training over a sequence of (dataset, target) pairs, and at test time transfers to a previously unseen target LLM **without additional training** and with only ~10 queries per test sample. IPIArena: 76.2% ASR against Gemini-2.5-Pro, 61.9% GPT-5.1, 42.9% Claude-Sonnet-4.5. AgentDojo: 86.7% / 53.3% / 40.0%.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [prefix-injection.md](prefix-injection.md) · [payload-splitting.md](payload-splitting.md) · [evil-system-prompt.md](evil-system-prompt.md) · [evidential-ceiling.md](evidential-ceiling.md)

---

## What it is

A red-team-as-agent method for prompt-injection attacks. Instead of parameterizing the attacker as a policy that must be re-trained per target, the attacker is a fixed agent whose *behavior* comes from a growing library of attack strategies accumulated across training tasks. At test time, the same agent + library attacks a new target with a handful of query-and-adapt cycles.

## How it works

**Training loop.** For each (attack dataset, target model) pair in a training sequence:

1. Rollout the agent against the target on tasks from the dataset. When an attack succeeds, extract the *strategy* used and add it (deduped) to the library.
2. When attacks fail, iterate — the agent examines the failure, mutates the attack, retries within a query budget.
3. Move to the next (dataset, target) pair. The library grows across pairs; the agent's weights don't change.

**Test-time loop.** For a previously unseen target and test sample:

1. Retrieve candidate strategies from the library (by similarity to the current task).
2. Try up to ~10 queries against the target, adapting the top strategies to the specific input.
3. Report attack success.

The library — not the weights — is the transferable artifact.

## Why it matters

- **Cross-target transfer without retraining.** Prior RL attackers overfit to one target; PIMiner treats the library as the compact prior that transfers.
- **Cheap per-sample budget.** ~10 target queries per test task is within the range of practical pre-release red-team evaluations at a model provider.
- **Real ASRs on frontier models.** 76% against Gemini-2.5-Pro and 61% against GPT-5.1 (IPIArena) are far above prior automated red-teaming baselines.
- **Reshapes model-provider evaluation.** If library-transfer works, providers should evaluate defenses against a *library-driven* attacker on their pre-release models rather than commissioning a fresh RL attacker per launch.

## Gotchas & tricks

- **Library growth ≠ attack novelty.** The library encodes strategies seen during training; PIMiner will not discover fundamentally new attack classes at test time, only recombinations.
- **Deduplication matters.** Naive addition of every successful attack bloats the library and slows retrieval. The paper's dedup step is essential; details of similarity metric affect coverage.
- **Defense-in-depth still works.** Layered defenses (input filtering + system-prompt hardening + tool sandboxing) show non-trivial residual protection against PIMiner in the reported ASRs (Claude-Sonnet-4.5 at 40–43% is markedly lower than Gemini's 76–87%).
- **Ethical use.** As with all offensive tooling, disclose responsibly; the paper releases with the standard safety-research framing.

## Sources

- Paper: *Agent Against Agent: An Agentic System for Automatic Prompt Injection Red Teaming* — Wang, Yin, Geng, Jia, 2026 — [arXiv 2608.05108](https://arxiv.org/abs/2608.05108). Pennsylvania State University.
- Benchmarks used: IPIArena, AgentDojo.

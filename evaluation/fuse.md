# Fuse — Verifiable Social Reasoning
*Depth — a multi-agent simulator where social-reasoning ground truth is verifiable by construction.*

**TL;DR:** Social-reasoning evaluation used to require subjective human labels. Fuse fixes that by *constructing* the ground truth: a target agent is instantiated with a hidden motive; it interacts with other agents including a **user proxy**; the user proxy then consults the evaluated LLM assistant to infer the target's motive. Because the motive was set at initialization, there's a real correct answer. 24k human annotations validate simulation faithfulness; 21k evaluation examples released; 12 LLMs analyzed with four systematic failure patterns.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../safety/_attacks.md](../safety/_attacks.md)

---

## What it is

A framework and dataset. The framework spins up multi-agent scenarios where:

- A **target agent** is initialized with a hidden motive from a predefined space.
- Other agents interact with the target under scripted or emergent conversations.
- A **user proxy** (a persona-configured LLM) observes and summarizes for the evaluated assistant.
- The **evaluated assistant** receives only what the user proxy tells it and must infer the target's motive.

Because the motive is set by the framework, correctness is programmatic — no LLM judge, no human label at inference time.

---

## How it works

```
target agent (hidden motive m) ─► conversation with peers
                                            │
                                    user proxy observes
                                            │
                                    user proxy consults ──► evaluated assistant
                                                                     │
                                                                     ▼
                                                            predicted motive m̂
                                                                     │
                                                     score against ground-truth m
```

The user proxy is the load-bearing addition: it introduces *user-mediation* — subjective framing, missing details, biased framings — which is the realistic setting for consumer LLM assistants. The framework validates that simulated user proxies match human behavior on a 24k-annotation study.

---

## Why it matters

- **Verifiable rewards for social reasoning.** Extends the RLVR pattern (verifiable rewards) into a previously-unverifiable domain. Opens social reasoning to RL fine-tuning.
- **Isolates specific failure modes.** Fuse's four findings (user-mediation compounds difficulty, systematic sensitivity to biased user framing, models need more detail than humans, longer conversations don't help) are actionable — each maps to a testable intervention.
- **Reproducible at scale.** 21k released examples make it a durable eval, not a one-off.

---

## Gotchas & tricks

- **Simulator faithfulness is the trust anchor.** The 24k human annotations validate that the user proxy behaves like real users; without that, results are simulator-artifact-prone. Rerun this check when adapting to new domains.
- **Motive space biases what can be inferred.** A finite, structured motive space makes the benchmark tractable but doesn't cover the long tail of real motives. Report which space you used.
- **Assistant-side prompts leak into results.** Small changes in the assistant's system prompt shift results — report the prompt used.
- **Longer conversations != more information.** The paper's finding that longer conversations don't reliably help is a warning: don't score-optimize by increasing turn count.

---

## Sources

- Paper: *Verifiable Social Reasoning for LLM Assistants* — Taubenfeld et al., 2026 (Google) — [arXiv:2609.17496](https://arxiv.org/abs/2609.17496).

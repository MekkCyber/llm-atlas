# Proactive Discovery
*Depth — agents that surface multiple latent problems from context rather than only acting on the explicit user request.*

**TL;DR:** Default agent behavior — do exactly what the user asked — misses the many *coexisting* issues hidden in the broader context (other files in the repo, other messages in the workspace). **TIDE (2026)** turns "do my task" into "find everything wrong here" via two moves: **iterative discovery** (each round conditions on what's already been found, so subsequent rounds extend coverage rather than rediscover the salient cases) and **thought templates** (reusable schemas distilled from previously-solved cases that tell the agent what signals to attend to).

**Prereqs:** [agents/README](README.md)
**Related:** [post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Two failure modes of single-pass agent prediction over a large context:

1. **Salience anchoring** — the agent fixates on the most visible problem and ignores subtler ones in the same context.
2. **Generic claims** — without structure, the agent produces abstract findings that don't ground in specific evidence.

Proactive discovery treats the workspace as a *pool of latent problems with unknown cardinality* and asks the agent to surface them progressively, each with grounding evidence and a concrete action.

## How it works

**Iterative discovery.** Run the agent across rounds:

```
Round 1: surface k candidate problems from context.
Round 2: condition on Round 1 findings + context; surface k more (different ones).
…
Stop when a round produces no new candidates above threshold.
```

The "conditioning on what's already been found" is the anti-salience trick: explicitly tell the model *don't surface anything already in this list*, redirecting attention to less-prominent issues.

**Thought templates.** Reusable schemas distilled from previously solved cases:

```
Template: "missing input validation"
  - Cue: function reads external input without check
  - Look at: callers, type signature, downstream usage
  - Evidence to extract: file:line, type of unchecked input
  - Action: add validation guard
```

Each template names a problem class with the *contextual signals* to attend to and how to connect them into a concrete finding. Templates are distilled from a curated library of solved cases (offline) and retrieved at runtime based on context similarity.

Each round of iterative discovery operates *through* a template — the template structures the candidate-extraction prompt, so generic claims become evidence-grounded predictions.

## Why it matters

- **Realistic agent UX.** Most workspaces have many low-level issues coexisting (stale TODOs, missed validation, inconsistent error handling). Acting only on the explicit ticket leaves value on the table.
- **Beats single-shot and parallel multi-agent baselines** on personal workspaces and software repositories across four backbones, measured on coverage, identification, and resolution.
- **Templates are a generalizable trick.** "Schemas of past solved cases as prompt-time structure" is reusable beyond TIDE's setting — same pattern shows up in case-based reasoning and retrieval-augmented generation more broadly.

## Gotchas & tricks

- **Round limit needs a calibrated stopping rule.** Otherwise the agent hallucinates new "problems" past the real cardinality.
- **Template library curation matters.** Bad templates → false positives. Domain experts typically seed the library; automated distillation expands it.
- **Coverage vs. precision tradeoff.** More rounds → more coverage but more low-confidence findings. Surface confidence scores per finding and let the human triage.
- **Evidence grounding must be enforced.** Without explicit file:line citations, the agent will produce generic complaints. Required-evidence format in the schema is the easiest enforcement.
- **Distinct from reflection / self-critique.** Reflection reviews a *single* output. Iterative discovery generates *many* outputs that collectively cover a context.

## Sources

- Paper: *TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration* — Jeong, Baek, Kang, Hwang, 2026 — [arXiv:2606.04743](https://arxiv.org/abs/2606.04743) — primary source.

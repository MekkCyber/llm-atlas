# Self-Evolving RL (R-Quest)

*Depth — a self-improvement loop where the model generates its own training questions, answers them, and trains on the result, with safeguards against collapse.*

**TL;DR:** Pure self-improvement loops — generate your own prompts, solve them, train on the solves, repeat — collapse after a few rounds. R-Quest identifies two failure modes: **invalid-question drift** (generated prompts become ill-formed over rounds) and **false lexical diversity** (prompts look different but are mathematically equivalent). The fix is a two-sided filter: the solver itself learns to reject invalid questions, and a *frozen* base model scores novelty against a reference distribution. Stays stable over 10 rounds while vanilla self-improve collapses after 3–4.

**Prereqs:** [rlvr](rlvr.md), [rl-prompt-curation](rl-prompt-curation.md), [grpo](grpo.md)
**Related:** [long-cot-rl](reasoning/long-cot-rl.md) · [rejection-sampling](rejection-sampling.md) · [_rewards](_rewards.md)

---

## What it is

Three roles played by one model (or a frozen variant):

- **Generator** — produces candidate training prompts.
- **Solver** — attempts them; its RLVR update is what we are training.
- **Critic / novelty scorer** — gates which prompts actually enter the next round.

A round of R-Quest:

1. Generator produces a batch of candidate questions.
2. Solver answers each; invalid questions (no well-defined answer, contradictory) are flagged.
3. Novelty scorer compares each question's semantic content to prior rounds using the *frozen base model*'s likelihood.
4. Keep novel + valid questions; RLVR-train the solver on them.
5. Repeat.

The valid-reject signal is folded directly into the solver's training: the solver gets reward for refusing to engage with malformed inputs.

## How it works

Two mechanisms that previous self-improve loops missed:

### Invalid-question rejection as a trained signal

Prior work filters invalid prompts externally (regex, heuristics). R-Quest trains the solver to recognise them, which means the invalidity signal is a *reward* over the solver's output: respond "this prompt is invalid" when the prompt is invalid; score correctness normally otherwise.

### Novelty via a frozen reference

Lexical diversity (unique n-grams) is cheap but misses semantic duplicates. R-Quest uses the base model's log-probability under each new question as a proxy for its novelty against prior rounds — the frozen base is a stable yardstick that doesn't drift as the solver evolves.

### Round-wise schedule

- Keep the generator and the frozen base pinned.
- Only the solver's weights move.
- Reduce the acceptance threshold slowly; aggressive novelty gates in early rounds starve the solver of training data.

## Why it matters

Open-ended self-improvement is the most promising long-horizon RL regime for closed-form reasoning (math, code, formal logic). The main blocker is **round-wise collapse**: every published vanilla loop stalls or degrades within a handful of rounds. R-Quest shows the collapse is diagnosable (invalid drift + false lexical novelty) and gives a working remedy. Stable gains across 12 math / general-reasoning / code benchmarks over 10 rounds.

## Gotchas & tricks

- **Frozen base model drift.** The base must remain truly frozen; drifting the novelty reference re-opens the loophole.
- **Rejection reward balance.** If the solver gets too much reward for rejecting, it will reject everything. The paper reports needing careful calibration of the reject-vs-answer reward ratio.
- **Diversity vs. correctness tension.** Novelty rewards push toward exotic questions, which can be harder to verify. Pair with a strong verifier.
- **Compute.** 10 rounds of RLVR per model is expensive; R-Quest is a research recipe, not a production shortcut.

## Sources

- *Questioning the Questions: Sustaining Self-Evolution in Reasoning Models* — Li, Huang, Huang, Cai, Gao, Yang, Huang — HINT-lab — 2026 — [arXiv:2610.04299](https://arxiv.org/abs/2610.04299)

# GauntletBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A web-agent benchmark of 100 vision-intensive tasks across five professional applications, deliberately *outside* the SWE/web environments most coding and computer-use agents are trained on. Designed to test out-of-distribution generalization. Reveals large gaps between frontier agents and humans concentrated in temporal perception, graphical understanding, and 3D reasoning.

**Prereqs:** [README.md](README.md)
**Related:** [_computer-use.md](../agents/_computer-use.md)

---

## What it is

Most computer-use and web-agent evaluations live on environments that overlap with training data: GitHub workflows, generic web browsers, productivity apps. Strong frontier results on such benchmarks can be partly memorization plus narrow generalization. GauntletBench picks *unfamiliar* applications — five professional tools spanning visualization, 3D, and time-series work — and constructs 100 tasks that require active perception of dense visual context.

The point is not to beat numbers on a new leaderboard but to isolate which capabilities collapse the moment an agent leaves its training-environment comfort zone.

## How it works

- 100 vision-intensive tasks across five professional applications.
- Each task has a verifiable success criterion checked against the application's final state.
- Tasks are constructed to stress *temporal perception* (changes over time in the UI), *graphical understanding* (figures, charts, diagrams), and *3D reasoning* (interpreting and acting on 3D scenes).
- Agents are evaluated end-to-end: planning, perception, action.
- Human baselines are collected for the same tasks to bound the achievable upper limit.

## Why it matters

- **OOD honesty check.** Frontier computer-use claims are largely on in-distribution benchmarks; GauntletBench is the cleanest test of "does this transfer to unfamiliar professional software?"
- Localizes the deficit: the gap to humans isn't uniform — it concentrates in time-, figure-, and 3D-grounded tasks. That points at what to train next (better screen perception, longer temporal context, 3D-grounded action models).
- Pairs naturally with matched-modality studies like [GUI vs CLI](../agents/_computer-use.md) to attribute failures to *modality* vs *unfamiliarity*.

## Gotchas & tricks

- "Vision-intensive" tasks reward agents with strong screen perception; CLI/skill-mediated agents are at a structural disadvantage that's worth flagging when comparing scores.
- Five professional apps is a small surface — strong on one might not transfer to the next.
- Human baselines were collected once; for moving applications (UI changes between agent and human runs) the gap measurement can drift.

## Sources

- Paper: *Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments* — Vysotskyi, Lin, Biziel, Zakrzewski, Montagna, Rynczak et al., Oxford + consortium, 2026 — arXiv:2606.14397.

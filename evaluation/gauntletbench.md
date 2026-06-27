# GauntletBench

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A web-based agent benchmark deliberately built outside the familiar browser/terminal world. 100 vision-intensive tasks across **five less-covered professional applications** — Video Editor, Workflow Builder, 3D Modeller, Flight Analyser, Circuit Designer — focused on **three under-evaluated capability axes**: temporal perception, graphical understanding, and 3D reasoning. State-of-the-art agents reach **19.1%**; non-expert humans reach **>80%**.

**Prereqs:** none
**Related:** [coffeebench](coffeebench.md), [ia-bench](ia-bench.md), [../agents/README.md](../agents/README.md)

---

## What it is

A challenge benchmark for **computer-use agents** that intentionally moves away from the apps frontier agents have effectively memorized (Chrome, terminals, Office). Each of the five target applications is a real professional tool with non-trivial visual, temporal, and spatial demands:

| Application | Why it matters |
| --- | --- |
| **Video Editor** | Temporal perception over clips; cut/keyframe reasoning |
| **Workflow Builder** | Graph-shaped UI; dependency understanding |
| **3D Modeller** | Spatial reasoning; manipulation in 3D viewport |
| **Flight Analyser** | Map / chart reading; time-series interpretation |
| **Circuit Designer** | Diagrammatic understanding; component placement |

Each app holds 20 vision-intensive tasks (100 total) with automated evaluation.

## How it works

GauntletBench ships as a modular pipeline:

1. **Environment.** Containerized, compatible with both open- and closed-source agent frameworks. The agent issues web/mouse/keyboard actions; the env returns screenshots.
2. **Task suite.** 100 hand-authored tasks across the five apps, each requiring at least one of the three target capability axes. Tasks are designed to be **feasible** — non-expert humans solve >80% — but resist memorized-affordance shortcuts.
3. **Automated evaluator.** Engine with multiple metric types (task success, partial credit, intermediate-state checks), removing the human-grading bottleneck that plagues open-ended agent benchmarks.

The benchmark is positioned as a *generalization* probe: agents that score well on familiar-app benchmarks have been shown to bottom out near 19% here.

## Why it matters

- **Saturates the "frontier agents are nearly at human level" claim.** On vision-heavy professional apps the gap is still ~60 percentage points to non-expert humans, not the few-point gap suggested by browser benchmarks.
- **Isolates the capability gap, not the affordance gap.** Picking unusual apps strips the "memorized the layout" advantage; what remains is the genuine visual / spatial / temporal reasoning shortfall.
- **Targets axes that VLA / multimodal training pipelines have historically underweighted** — temporal perception, graphical understanding, 3D reasoning. Provides a north star for what the next multimodal training generation should fix.
- **Cheap to extend.** Modular env makes it tractable to add new apps as agents start saturating the current 100 tasks — the benchmark can co-evolve with capability.

## Gotchas & tricks

- **Vision-token budget matters.** Many tasks need full-resolution viewport understanding; agents using aggressive image downsampling lose information the benchmark deliberately requires. Token-efficient multimodal LLMs may underperform their raw capability suggests.
- **Action space normalization** is open across agent frameworks. The benchmark provides a unified interface, but frame-rate, action-batching, and waiting policies differ across frameworks and affect scores.
- **Headroom is concentrated in 2 of 5 apps** (3D Modeller, Video Editor) — useful to track per-app sub-scores rather than a single GauntletBench number.
- **Watch for contamination.** Tasks were authored for the benchmark, but the apps themselves are real and have documentation on the web. The paper screens for this; future leakage is a known risk.
- **Complements [CoffeeBench](coffeebench.md)** (multi-agent economy) and [IA-Bench](ia-bench.md) (image-agent capabilities) along the agent-eval axis the existing graph mostly lacks.

## Sources

- Paper: *Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments* — Vysotskyi, Lin, Biziel, Zakrzewski, Montagna, Rynczak, Padarha, Alhamoud, Fu, Lugoloobi, Rawal, Yershova, Davies, Rumezhak, Li, Barez, Wu, Drohomirecki, Gal, Russell, Summerfield, Mahdi, Karpiv, Torr, Bibi, 2026 — [arXiv:2606.14397](https://arxiv.org/abs/2606.14397) — Oxford et al.

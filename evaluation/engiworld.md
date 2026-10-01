# EngiWorld
*Depth — a 1,301-task benchmark for frontier agents across professional engineering software, with artifact-centric scoring.*

**TL;DR:** EngiWorld (Gao et al., 2026) is the first large benchmark for professional-engineering computer-use agents. **1,301 expert-curated tasks** span **6 domains** (CAD, CAE, CAM, BIM, EDA, 3D visualization) across **26 professional software platforms**, with both GUI and CLI interfaces and **6 task types** (from software-selection to open-ended). Scoring is **artifact-centric**: a unified domain-verifier suite programmatically checks the final artifact's geometric validity, physical feasibility, and rule compliance — quantitative tasks scored continuously by specification attainment, not pass/fail. The strongest frontier model scores **EngiScore 44.3**; only **3.6%** of multi-software attempts succeed.

**Prereqs:** [README.md](./README.md)
**Related:** [../agents/README.md](../agents/README.md), [../agents/hybrid-cua.md](../agents/hybrid-cua.md), [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

Generic computer-use benchmarks (OSWorld, WindowsAgentArena, BrowseComp) measure day-to-day office productivity. Industrial engineering workflows are a different beast: they operate on **structured artifacts** (CAD files, netlists, mesh models, BIM databases) with hard **geometric, physical, and rule constraints**, and typically chain multiple specialized tools (CAD → CAE → CAM). Success isn't "did the agent click the right button" — it's "is the resulting part manufacturable to spec?"

EngiWorld builds the first benchmark that measures this. Expert-curated across 6 engineering disciplines, with real professional software (SolidWorks, Fusion, Ansys, Altium, Revit, etc.) exposed to the agent via both GUI (full desktop interaction) and CLI (scripting APIs).

## How it works

### Task categories (6 types)

Software-selection; GUI operation; CLI/scripting; multi-software pipelines (CAD → CAE handoff); design-with-constraints; open-ended design tasks.

### Artifact-centric evaluation

Instead of trajectory-level success flags, EngiWorld runs **domain verifiers** against the final (and intermediate) artifacts:

- **Geometric validity** — closed shells, no self-intersections, correct topology.
- **Physical feasibility** — mesh quality, stability under physical constraints, manufacturability.
- **Rule compliance** — DRC/LVS for EDA, code checks for BIM, tolerance budgets for CAD.
- **Specification attainment** — for quantitative tasks (minimize mass under stress constraint; achieve clearance budget), scored continuously, not binary.

The **EngiScore** aggregates these into one headline number per model.

## Why it matters

- **Measures a frontier that other benchmarks miss.** The strongest frontier model reaches **EngiScore 44.3**, and only **3.6% of multi-software attempts succeed**. That gap is where the next generation of CUAs needs to work.
- **Reframes agent eval around artifacts.** Trajectory-level grading rewards agents that click confidently even when outputs are wrong. Artifact verification exposes that gap and incentivizes agents that produce *valid* engineering objects.
- **Multi-software is the real challenge.** Single-tool tasks are nearing solvable; the gap sits at the handoff between tools where geometric/physical dependencies must be preserved. EngiWorld makes this testable.
- **High-economic-value proving ground.** Industrial engineering workflows are structured, verifiable, and currently unautomated. A benchmark that measures *valid* automation here is the right target for the next CUA generation.

## Gotchas & tricks

- **Verifier bugs are catastrophic.** An artifact-centric score is only as good as the verifier. Expert review of verifier false-positives/negatives is a maintenance task, not a one-time build.
- **Agent-side headless rendering.** Many of these tools require full GUI; headless runners simulate inputs through pyautogui-style injection. Test harness stability matters for reproducible numbers.
- **Licensing.** Professional engineering software is often license-gated. The benchmark bundles a mix of open and permissive licenses and provides wrappers for licensed tools where feasible.
- **Continuous scoring rewards partial progress.** This is good for progress tracking (two agents at 10% and 40% are distinguishable) but a continuous score of 44.3 does *not* mean 44.3% of tasks solve — read the paper for the exact aggregation.

## Sources

- Paper: *EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?* — Gao, Qu, Lei, Sun, Li, Wei, Xue, Yu, Tao, Zang, Wang, Tang, Li, Zhou, Luo, Zeng, Shen, Jiang, Chen, Qu, Huang, Wang, 2026 — [arXiv:2609.37686](https://arxiv.org/abs/2609.37686).

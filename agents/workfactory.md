# WorkFactory
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pipeline for generating realistic professional tasks for agent training. Given a described work goal, WorkFactory plans *backward* — enumerating the facts, rules, and intermediate results an agent will need and planting them in a sandboxed environment, leaving the actual calculations to the agent. A test-run verifier catches un-completable or contradictory environments and feeds back repair patches (not intent rewrites). 112,686 tasks × 612 occupations; the resulting WorkPilot family of Qwen fine-tunes beats their bases on four professional benchmarks.

**Prereqs:** [README.md](README.md)
**Related:** [webanything.md](webanything.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Realistic professional-work agent training needs environments with consistent sub-state: documents, databases, policies, tools. WorkFactory is a planner-and-verifier pipeline that produces such environments at scale from short intent descriptions.

## How it works

**1. Intent → structural plan.** A planner model reads the work intent (e.g. "prepare a quarterly expense report under policy X") and enumerates:
- Required *artifacts* (expense entries, policy docs, templates).
- Required *rules* and dependencies between artifacts.
- Required *intermediate results* the agent must compute.
- A *target deliverable* schema.

**2. Environment construction.** The structural plan is instantiated into a sandboxed environment with files, API stubs, and tools in the needed states — but with gaps the agent must fill through actions.

**3. Test-run verifier.** A verifier agent attempts the task from the planted environment. If completion fails because the environment is under-specified or contradictory, the failure is diagnosed and routed back for *environment repair* (adding missing artifacts, resolving contradictions) — never for intent rewriting. Repair preserves task meaning.

**4. Teacher trajectory + SFT.** A strong teacher agent solves the repaired tasks; its trajectories become SFT data. Qwen bases at three sizes are fine-tuned to produce the WorkPilot family. All three lift over their bases on four professional benchmarks; the largest reported gain is WorkPilot-35B-A3B on GDPval-AA.

The scale is 112,686 tasks across 612 occupations.

## Why it matters

The current ceiling on enterprise-agent capability is realistic training data, not model size. WorkFactory's backward-planning + test-run-and-repair is a scalable template for synthetic professional corpora. Pairs naturally with WebAnything (which does the same thing for the web) — together they outline a near-term pipeline where most "agent training data" is procedurally generated.

## Gotchas & tricks

- **Separate intent from environment.** The repair loop never touches the described work goal — only the planted environment. This is what makes WorkFactory produce tasks that are *about* what the intent says.
- **Verifier failure signals are the quality signal.** The paper's ablations show the planning + repair steps are what distinguish WorkFactory's output from naive LLM-generated tasks.
- **Teacher-model solutions, not ground-truth.** The SFT target is a strong agent's trajectory. For verifiable work (where ground truth exists), the loop can be tightened with RLVR.
- **Occupation diversity matters.** 612 occupations across 112k tasks; a smaller occupation set would risk the WorkPilot models over-fitting to narrow routines.
- **Not a benchmark.** WorkFactory produces *training* data; benchmarks are from GDPval-AA and friends.

## Sources

- Paper: *Turning Professional Intents into Executable Work for Scalable Agent Training* — Chen, Deng, Wang, Tang, Yang, Ren, Zhao, Chang, He, Hao, Tao, Dai — Zhejiang · RUC · SIAT · SJTU · PKU · IQuest, 2026 — [arXiv:2610.37161](https://arxiv.org/abs/2610.37161).
- Reference: *GDPval* — the professional-work benchmark suite.

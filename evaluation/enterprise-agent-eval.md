# Enterprise Agent Evaluation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An evaluation protocol that reconstructs **852 reproducible tasks from real workplace agent sessions** — each with fixtures, rewritten prompts, role/skill labels, hard rules, and semantic rubrics. Reports a *vector* of metrics (artifact delivery, visual quality, cost, runtime, skill-transfer) per harness/model pair, not a single score. Built from proprietary enterprise data; the *protocol* is the reusable contribution, not the data. Introduced as EnterpriseClawBench (2026).

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [planbench-xl.md](planbench-xl.md)

---

## What it is

Most agent evaluations are either synthetic (constructed task suites) or scraped (open web/data). Both miss the texture of real enterprise work: heterogeneous file formats, internal tool ecosystems, role-specific conventions, and rubrics that reward correct *artifact delivery* — not just correct answer text.

Enterprise Agent Evaluation mines an archive of real workplace agent sessions and reifies each into a reproducible task: starting fixtures (the files / state the agent had), rewritten prompt (PII-scrubbed), role class (analyst, engineer, etc.), skill subclass, hard rules (must / must-not), and a semantic rubric for grading the final artifact.

## How it works

The construction pipeline:

1. **Mine sessions** from a workplace agent archive.
2. **Recover fixtures** — extract the files, datasets, and tool snapshots present at session start.
3. **Rewrite prompts** — PII removal, paraphrase, role labelling.
4. **Annotate** — role / skill / hard rules / rubric.
5. **Validate** — manual audit (the released Lite subset is 120 fully-audited tasks).

Evaluation runs report per task: *artifact delivery* (was the expected file/result produced), *visual quality* (rubric-graded), *cost*, *runtime*, and *skill-transfer behaviour* (does the agent succeed on tasks from roles it wasn't trained on).

Because the data is proprietary, the benchmark is not released as a corpus — only the construction pipeline, the Lite audited subset, and the evaluation protocol.

## Why it matters

- Enterprise agents won't be evaluable by an open leaderboard model. The *protocol* (multi-metric reporting, harness × model factorial) is what generalises.
- The 0.663 ceiling reached by Codex + GPT-5.5 on the audited Lite set says that even frontier agents miss a third of real workplace tasks. That's a useful headline.
- Reporting **harness × model** jointly is the key methodological move — choice of scaffolding matters as much as choice of underlying LLM, and single-score reporting has been hiding that fact.

## Gotchas & tricks

- Proprietary data + reproducibility is in tension. The audit subset is the only externally-comparable slice.
- The *role / skill* label space is enterprise-specific. Adapting the protocol to another industry requires re-defining classes — but the methodology transfers.
- "Hard rules" are policy-style constraints (PII, compliance). Models that pass rubric but violate a rule get scored zero — this is intentional.

## Sources

- Paper: *EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions* — Wang, Jiang, Tian et al., Horizon Research / Frontis.AI, 2026 — [arXiv:2606.23654](https://arxiv.org/abs/2606.23654).

# Agentic Pipeline Optimization
*Depth — using a code-editing agent (Claude Code-style) as the optimizer for an entire multi-step LLM pipeline.*

**TL;DR:** Prompt-only optimizers (DSPy, GEPA) treat the pipeline as fixed and tune the prompts. **Agentic pipeline optimization** treats the codebase itself as the search space: a code-editing agent inspects intermediate steps, diagnoses failures, edits prompts *and glue code*, runs evals, and iterates. **FAPO** (Saglam et al., Cisco Foundation AI + Yale, arXiv 2606.19605) is the cleanest published instance, beating GEPA in 15 of 18 model–benchmark settings, with +14.1 pp mean gain overall and +33.8 pp on tasks needing structural pipeline changes.

**Prereqs:** [README](README.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

A meta-method for tuning a multi-step LLM pipeline where the optimizer is an LLM coding agent (e.g. Claude Code) operating over a standardized codebase. The agent has read/write access to the pipeline source, an evaluation harness, and a target score function. It runs a closed loop of:

1. **Evaluate** the current pipeline on a held-out batch.
2. **Inspect** intermediate step outputs (retrieval results, reasoning traces, formatter outputs).
3. **Diagnose** which step(s) caused failures.
4. **Propose a scoped change** — could be a prompt tweak, a parser fix, a retry policy, a retrieval-threshold adjustment, or a new intermediate step.
5. **Validate** the change by re-running the eval. Keep if score improves; discard otherwise.

The "scoped" part is important — agents that propose sweeping changes thrash. FAPO restricts each iteration to a single localized edit.

## How it works

- **Standardized codebase.** The pipeline lives in a known structure the agent has been trained / prompted to understand (named steps, typed I/O between them, a single eval entrypoint).
- **Score function as ground truth.** The agent isn't told *what* to change — it only sees the score. The diagnosis loop has to localize failures from logs and intermediate outputs.
- **Iteration budget.** FAPO runs for a fixed number of iterations or until score plateaus.
- **Variant validation.** Each proposed edit becomes a candidate; the agent runs all candidates and keeps the best per round.

The contrast with prompt-only optimizers is sharp:
- **DSPy / GEPA** tune a fixed graph; can't change parser logic, retry policies, or retrieval thresholds.
- **Agentic pipeline opt** can rewrite any of those — and FAPO's headline +33.8 pp gain on "structural-modification tasks" is exactly the slice where prompt-only opt is structurally blocked.

## Why it matters

- **Right level of abstraction for production pipelines.** Most deployment failures live in connective tissue (retrieval, parsing, retry); a code-editing agent is the natural optimizer.
- **Eats prompt-only optimization as a special case.** Anything DSPy/GEPA can do, an agentic optimizer can do — plus structural changes.
- **Scales with the agent.** As code-editing agents improve, the optimizer improves for free.

## Gotchas & tricks

- **Standardized codebase is non-negotiable.** Without typed I/O and a known structure, the agent thrashes searching the codebase before editing.
- **Eval cost dominates.** Each iteration re-runs the pipeline on the validation set; budget accordingly. Cache aggressively at step boundaries.
- **Diagnosis quality is the bottleneck.** A weaker agent that can't localize failures from logs will edit the wrong steps; FAPO benefits from a frontier-class code agent.
- **Watch for overfitting to the validation set.** With enough iterations the agent can overfit; hold out a true test set and report scores there.
- **Be wary of agents that propose new dependencies.** A "fix" that pulls in a new library may pass the eval but break deployment. Constrain the edit surface.

## Sources

- Paper: *Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines* — Saglam, Zhao, Nelson, Vijay, Priyanshu, Karbasi, Foundation AI–Cisco Systems + Yale University, 2026, arXiv 2606.19605.

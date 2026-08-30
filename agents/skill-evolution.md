# Skill Evolution with a Persistent Knowledge Wiki
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Automatic agent-skill discovery usually loses the *insights* behind each skill — they stay stuck in one iteration's optimization logs. A **three-store** design (raw execution traces / compiled knowledge wiki / executable skills) with a **consolidation** step in between makes those insights durable: the wiki accumulates across iterations, and every skill update stands on the whole wiki instead of just the last iteration's traces.

**Prereqs:** [README.md](README.md), [live-self-improvement.md](live-self-improvement.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A design pattern for **compounding** agent experience. Three separated stores, each with its own update loop:

1. **Raw execution traces** — the agent's rollouts, dense and expendable.
2. **Compiled knowledge wiki** — human-readable articles distilled from traces: what works, what fails, why, and under which conditions.
3. **Executable skills** — the actual reusable functions/tools the agent calls at runtime.

Neither traces alone (too noisy, too specific) nor skills alone (too opaque, hard to reuse insights) accumulate well. The wiki bridges them: it's the durable memory that survives skill deprecation and refactoring.

## How it works

**Consolidation step (traces → wiki).** Periodically, a consolidation pass reads recent traces (successful and failed), diffs them against the current wiki, and drafts new wiki articles or edits existing ones. Articles are structured (context, procedure, failure modes, evidence) so subsequent updates can *edit* rather than *replace*.

**Skill-update step (wiki → skills).** When it's time to grow the skill library, the skill-update process reads the wiki (and current skills) and proposes new skills or refinements to existing ones. Because it conditions on the whole wiki, insights from earlier iterations shape today's skills even if those iterations' skills have long since been deprecated.

**Explicit separation matters.** If traces are fed straight into skill updates (the common pattern), each iteration only sees the last window of experience — a rediscovery loop. If they're fed into a persistent wiki first, the second loop compounds.

## Why it matters

Agent memory has been shallow: retrieval over traces, one-shot summarization, per-iteration prompt libraries. This design makes the *lessons* the durable object — they persist even when a specific skill is retired — and gives skill evolution a genuine second derivative. Skills improve monotonically across iterations rather than resetting each time.

## Gotchas & tricks

- **Wiki bloat.** Without an explicit edit-vs-append policy, the wiki grows to unusable size. Structured articles (fixed sections) and periodic consolidation-of-consolidation passes keep it navigable.
- **Attribution drift.** A wiki article might be composed of evidence from many traces; when a claim in the wiki is wrong, tracing it back to the offending traces is essential for repair. Keep a lightweight provenance link.
- **Skill-wiki alignment.** A skill whose behavior no longer matches the wiki article that inspired it is the most common silent regression. Treat the wiki as ground truth for expected behavior and gate skill deploys on it.
- **Human-inspectable by construction.** The wiki should be readable — that's most of its value for debugging and for external review of the agent's learned behavior.

## Sources

- Paper: *WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution* — Tang et al., 2026 — [arXiv:2608.27454](https://arxiv.org/abs/2608.27454)

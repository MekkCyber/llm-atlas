# VibeLifeBench
*Depth — an agent benchmark grounded in its source paper.*

**TL;DR:** A benchmark of **200 multi-week everyday-life tasks** across 10 domains that measures three capabilities most agent benchmarks skip: **proactivity** (act on a virtual clock without being prompted), **environment persistence** (state changes autonomously), and **long-horizon coherence** (interactions spanning weeks). 22 mock service backends expose 288 tools; 12,261 fine-grained scoring checks give partial credit for intermediate correctness. The strongest evaluated frontier model reaches only **32.5%** average performance. Introduced in *VibeLifeBench* (2026).

**Prereqs:** *(none)*
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

Most agent benchmarks (WebArena, SWE-bench, τ-bench) share a shape: single-turn or few-turn tasks, a static environment, the agent is prompted with the task. This shape misses everything that makes a *life assistant* hard:

- **Proactivity.** A real assistant surfaces things at the right moment (reminders, follow-ups). It has to decide *when* to act on its own.
- **Environment persistence.** The world changes on its own. Calendars fill up, prices drop, deadlines approach. The agent has to keep track.
- **Long-horizon coherence.** A user's preferences, ongoing projects, and past decisions matter for weeks, not turns.

VibeLifeBench builds a benchmark specifically for these three axes.

## How it works

**Task shape.** Each task spans multiple **virtual weeks** on a simulated clock. The environment evolves autonomously between agent turns (new events, tool state changes, external "world updates"). The agent must decide *when* to check in and act.

**Scale of the environment:**
- **200 tasks** across **10 domains** (e.g. personal scheduling, health, finance, home management — the paper's specifics).
- **22 mock service backends** expose **288 tools** covering the surface a real life assistant needs.
- **12,261 fine-grained scoring checks** — the scoring is check-level, so partial credit reflects intermediate correctness (found the right meeting but missed rescheduling → partial credit).

**Grading.** Each task decomposes into many verifiable sub-checks (was the reminder sent? was the right calendar entry updated? etc.). The task score is an aggregate over sub-checks.

**Proactivity handling.** The evaluator advances the virtual clock; the agent is polled but not prompted. Failing to act at the right moment costs sub-checks.

## Why it matters

- **Names the "always-on assistant" gap.** Every company shipping a personal-assistant product needs a benchmark like this; VibeLifeBench provides one.
- **Systematic weakness exposed.** The best model reaches **32.5%** average. Failures cluster around **proactivity** (agents wait to be told) and **long-horizon state maintenance** (agents lose track over multi-week horizons).
- **Fine-grained scoring makes progress legible.** Because the score is an aggregate over many small checks, incremental progress shows up in the number instead of being flattened by pass/fail task grading.

## Gotchas & tricks

- **Mock backends != real services.** Success on the mock APIs doesn't guarantee success against real APIs with rate limits, auth, and edge cases.
- **Virtual clock is deterministic in the benchmark.** Real-world deployment adds asynchrony and message queues that this benchmark doesn't model.
- **The 32.5% ceiling is very load-bearing on the strongest model.** If a new model breaks the ceiling, verify the tasks and grading logic haven't been leaked into training.
- **"Personal information" tasks may be sensitive at deployment time.** The benchmark is fictional; the deployed cousin needs different scoring choices for privacy.

## Sources

- Paper: *VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?* — 12-author team, arXiv 2608.10875, 2026.

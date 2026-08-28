# Agent Harness
*Depth — the modular scaffold around an LLM that turns it into an agent.*

**TL;DR:** An agent's capability is not the underlying model alone — the *harness* around it (memory management, planning strategy, action protocol, skill orchestration) can dominate the model's contribution. Fixing the harness as a composable, four-module protocol lets it be inspected, edited, generated, or trained like any other artifact.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [jit-harness-generation.md](jit-harness-generation.md), [handoff-tax.md](handoff-tax.md), [migration-blindness.md](migration-blindness.md)

---

## What it is

A harness is everything wrapped around the model that decides *how* it interacts with a task: how prior turns are stored and recalled, what plan structure it follows, what actions it can emit, and which tools/skills are available and when. In practice, "the agent" that ships is `model + harness`, and swapping either one changes behavior more than most people expect.

JIT-Agent (Zhang et al., 2026) formalizes this as a fixed four-module protocol:

| Module | Owns |
| --- | --- |
| **Memory** | Working-set selection, summarization/compaction, long-horizon recall |
| **Planning** | Task decomposition, replanning triggers, control flow across steps |
| **Action** | Action schema, tool/function-call format, output validation |
| **Skill** | Available tool set, skill selection, skill parameterization |

Runtimes like OpenCode, Claude Code, and Cursor are hand-designed instances of this protocol. Treating the protocol as first-class lets you generate task-adaptive instances programmatically instead.

## How it works

A concrete harness is a bundle of four small artifacts — a memory policy, a planner spec, an action grammar, and a skill registry — that a runtime executes against the model. The runtime loop is roughly: sample a plan, dispatch actions in the action grammar, route tool calls via the skill module, and update memory according to the memory policy. Because each module has a fixed interface, they can be swapped independently — you can plug a new planner into the same memory + skill setup without retraining the model.

Two consequences follow. First, harness quality is a real axis of capability, not model-adjacent noise — a stronger harness on a weaker model can beat a weaker harness on a stronger model (JIT-Agent reports DeepSeek-V4-Flash surpassing GPT-5.6 on DeepSearchQA with the right harness). Second, harnesses can be *generated* per task, *repaired* live when a run destabilizes, and *distilled* from performance signals across an archive of prior configurations.

## Why it matters

If the harness is trainable, it is orthogonal to model scaling — improvements compound. It also reframes what mature runtimes (OpenCode, Claude Code) actually are: high-quality prior art whose behavior can be learned, transferred, and often improved on automatically for a specific task at hand. This turns "which agent framework should I use" from a static vendor choice into an online, task-conditioned decision.

## Gotchas & tricks

- **Not every module can be varied per task.** Skill registries usually need to match what the deploy environment can execute; the memory and planning modules are the safer axes to generate.
- **Harness generation without repair is fragile.** JIT-Agent reports that live repair (rewriting parts of the harness mid-run when execution destabilizes) is load-bearing.
- **Runtime-agnostic ≠ model-agnostic.** A harness that works for Claude Code's tool-format may need adjustments for other action grammars; the four-module split helps but does not fully abstract this.

## Sources

- Paper: *JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution* — Zhang et al., 2026 — [arXiv:2608.25593](https://arxiv.org/abs/2608.25593)

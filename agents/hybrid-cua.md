# HybridCUA — GUI + CLI Computer-Use Agents
*Depth — a trained computer-use agent that mixes graphical and shell actions.*

**TL;DR:** Pure-GUI computer-use agents (CUAs) are slow and error-prone; per-application-API CUAs don't scale. **HybridCUA** (Chen et al., ZJU / Alibaba, 2026) adds the Unix **CLI** as a first-class action vocabulary alongside the GUI, trains on a hybrid trajectory dataset (**HybridCUA-8K**: 5K hybrid traces + 3K RLVR-verifiable tasks), and uses a **CLI-aware RLVR reward** that pushes the agent to use the shell *selectively* when it is clearly better.

**Prereqs:** [README.md](./README.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [agent-harness.md](./agent-harness.md), [omni-io-skills.md](./omni-io-skills.md)

---

## What it is

CUAs trained purely on GUI actions (click, type, scroll, screenshot-read) perform tasks the way a human would, which inherits human slowness and the GUI's error surface (misclicks, dynamic layouts). API-augmented CUAs call per-application APIs when available but need bespoke engineering per app. HybridCUA keeps the GUI's generality and adds shell commands wherever they are strictly cheaper and verifiable (file ops, archive handling, text processing, git).

## How it works

1. **Data construction.** Three trajectory types are generated: GUI-only, CLI-only, and *interleaved* GUI + CLI. The interleaved pipeline produces the headline HybridCUA-8K dataset (5K hybrid trajectories, 3K RLVR-verifiable tasks).
2. **Supervised fine-tuning.** Standard trajectory-level SFT on the hybrid traces so the base model can emit CLI actions in-context.
3. **RL with CLI-aware rewards.** Verifier-based RLVR (see [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)) with a reward that encourages CLI use **only when selective and reliable** — using CLI everywhere, or using it where it fails, is penalized. The agent learns a routing policy over the action space.

## Why it matters

- **OSWorld 53.6%** for HybridCUA-9B (+14.8 pt over the base model), **+4.0 pt on WindowsAgentArena**, and cross-platform generalization is demonstrated. The GUI-only CUA line (Claude Computer Use, GPT-5 CUA, top OSWorld agents) had been plateauing on tasks where one shell command replaces dozens of clicks.
- **CLI-aware reward is a reusable pattern.** Any multi-tool agent (not just computer-use) can borrow the "reward the tool only where it is clearly better" trick instead of letting the model pick the first tool it was taught.
- **Breaks the API-per-app bottleneck.** Shell commands are the smallest-possible-universal API — one language covers thousands of utilities and all file I/O.

## Gotchas & tricks

- **Verifiability is the limiting factor.** RLVR requires that task success be programmatically checkable. CLI-aware rewards work because shell-level effects are easy to check (file exists, grep matches, exit code 0). GUI-only tasks without a shell-visible side effect need a different verifier design.
- **CLI is dangerous.** Any CUA with shell access needs sandboxing and a destructive-command guard (`rm -rf`, network writes, package installs). The paper trains on verified tasks; production needs policy filters on top.
- **"Selective" is a learned prior.** Without the CLI-aware reward shaping, SFT alone biases the agent toward overusing CLI (shorter traces win reward). The shaping term is what keeps the GUI/CLI mix calibrated.
- **Hybrid trajectories must come from somewhere.** The 2026 trick is to generate them by running a planner that proposes GUI or CLI per step and keep only end-to-end successes. The dataset is the real contribution.
- **Cross-platform generalization is empirical.** OSWorld (Linux-flavored) and WindowsAgentArena both improve with the same recipe, but Windows PowerShell semantics differ from bash — a production agent needs separate shell-language handling, not a single "CLI" category.
- **Latency vs. verifiability tradeoff.** Shell commands are faster than GUI, but their side-effects are also faster to go wrong. RLVR relies on *post-hoc verification* catching the mistake; interactive GUI control is more easily paused mid-task.

## Sources

- Paper: *HybridCUA: Learning to Orchestrate GUI and CLI for Computer-Use Agents* — Chen, Niu, Lu, Lian, Tang, Yan, Hong, Du, Liu, Chen, Shen — ZJU / Alibaba, 2026 — [arXiv:2609.38008](https://arxiv.org/abs/2609.38008).

# HarnessRisk
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** A lifecycle-oriented safety benchmark for **agent harnesses**: 128 sandboxed cases organised into **six operational phases** (Harness Configuration, Capability Extension, Runtime Operation, State Persistence, Action Control, Incident Recovery). Each case pairs a benign user objective with an adversarial instruction embedded in an untrusted workflow artifact. Attack success ranges **12.6%–80.9%** across three harnesses and six models while utility stays 75–98%. Detection ≠ safety.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [payload-splitting.md](payload-splitting.md), [cot-monitoring.md](cot-monitoring.md)

---

## What it is

Most agent-safety benchmarks measure a single attack mechanism (prompt injection, tool misuse) or a single operating setting (a fixed browser task). HarnessRisk instead measures **the phase in the harness lifecycle** in which the failure occurs, exposing that different harnesses fail at different phases even when they share the same underlying model.

## How it works

### The six phases

1. **Harness Configuration** — setting up permissions, tool allow-lists, tokens, environment variables.
2. **Capability Extension** — installing plugins, MCP servers, custom tools mid-session.
3. **Runtime Operation** — day-to-day tool use, reasoning, planning.
4. **State Persistence** — writing to durable memory, files, DBs.
5. **Action Control** — taking irreversible actions (send email, deploy code, hit external APIs).
6. **Incident Recovery** — post-failure cleanup and rollback.

### The 128 cases

Each case bundles:

- A benign user objective (what the human asked for).
- An untrusted workflow artifact (file, plugin manifest, tool result) that carries an adversarial instruction targeting one of the six phases.
- A ground-truth expected trajectory the harness/agent should take.

### The four metrics

- **Utility** — does the agent still accomplish the benign objective?
- **Attack Success Rate (ASR)** — did the adversarial instruction succeed?
- **Persistence** — does the effect of the attack survive across turns/sessions?
- **Detection** — did the agent flag the risk (even if it didn't stop)?

## Main findings

Across three harnesses, six models, and 14 model+harness configurations:

- **Attack success ranges 12.6% → 80.9%** depending on model + harness combo.
- **Utility remains 75–98%** under attack — attackers rarely need to break the agent to breach it.
- **Harness Configuration is the most vulnerable phase across all three harnesses tested** — attacks can succeed by altering security-sensitive parameters within otherwise authorised workflows.
- **Detection ≠ safety.** Some configurations flag risks in >90% of runs and still succumb at high rates. Recognising the attack does not reliably lead to safe action.

## Why it matters

As agents move from demos into deployment, the *harness* is a bigger attack surface than the model. Existing benchmarks measure the model's refusal on isolated attacks and miss the failure modes that appear only when tools, state, and permissions interact. HarnessRisk gives a comparable benchmark for harness hardening — a category that didn't previously have one — and localises defensive effort to the phases (especially Configuration) where the vulnerabilities concentrate.

## Gotchas & tricks

- **Harness > model for these attacks.** Same model gets very different ASRs under different harnesses; hardening the harness beats swapping the model.
- **Detection alone is a red flag.** A high Detection score with high ASR means the agent *sees* the attack and does nothing — a design bug, not a model bug.
- **Configuration attacks are cheap.** Modifying `~/.config/…` or a tool-registration manifest is often within authorised paths — permission-tightening at deploy time matters more than runtime refusal training.
- **Adversarial artifacts should live in untrusted workflow objects, not user turns.** That's how HarnessRisk operationalises "the harness is the attack surface"; benchmarks that inject the attack directly into the user prompt understate the real risk.
- **Persistence is under-measured elsewhere.** Attacks that leave lasting state (poisoned memory, altered config) are more dangerous than one-shot breaches; HarnessRisk instruments this explicitly.

## Sources

- Paper: *HarnessRisk: A Lifecycle-Oriented Benchmark for Agent Harness Safety* — Bai, Duan, Peng, Wu, Liu, Wang, Chen — UNC / MSU / U-Delaware, 2026 — https://arxiv.org/abs/2608.17597
- Related: prompt-injection literature (Greshake et al., 2023), tool-abuse benchmarks (ToolEmu, InjecAgent), Anthropic's Computer-Use / Claude-Code safety guidance.

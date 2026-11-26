# Proactive Perception for Computer-Use Agents
*Depth — a plug-in continuous-watcher layer that decouples perception from reasoning.*

**TL;DR:** Standard computer-use agents observe the screen only when they're about to act — a discrete snapshot per plan step. "Proactive perception" runs a small VLM continuously against the screen stream and emits three control signals (**interrupt**, **void**, **wake**) that the main agent reacts to. NightWatch is the reference implementation: a 0.8B watcher drops wall-clock time 33% on OSWorld-Verified at the same success rate, and lifts success 39.5% → 44.2% on the dynamic-UI DynaCU-Bench.

**Prereqs:** [agents/README.md](./README.md)
**Related:** [partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A perception layer that sits *beside* a computer-use agent (CUA), not inside it. The main agent (a large VLM) still generates plans and actions, but a small, cheap watcher looks at the screen at high frequency and interferes with the main agent's loop when the environment demands it. The division of labor: the watcher decides *when* to look and *when the main agent should think*; the main agent decides *what to do*.

## How it works

The watcher is a 0.8B-parameter VLM running at a much higher cadence than the main agent (frames per second vs. seconds per plan step). It emits three signals back into the control loop:

1. **Interrupt.** The screen just changed in a way that invalidates in-flight reasoning. Kill the main agent's current thought and have it re-plan from the new frame.
2. **Void.** A planned action is no longer valid (button moved, dialog dismissed). Drop the action before it fires.
3. **Wake.** The screen has stopped changing. Let the main agent proceed.

The main agent is unchanged — it just subscribes to these three edges. The watcher is trained on labeled screen-transition data to produce the three signal types; the training pipeline collects transition triples from CUA rollouts and auto-labels them with a larger VLM.

```
Main agent (large VLM) ──┬── think ── act
                         │
                         │  interrupt / void / wake
                         │
Watcher (0.8B VLM) ──────┴── sees every frame
```

## Why it matters

- **Decouples perception rate from reasoning rate.** Frontier CUAs are rate-limited by the cost of the main VLM's forward pass. A small watcher runs on cheap hardware at high frequency.
- **No retraining of the main agent.** Plug-and-play on top of an existing CUA stack — adoption cost is small.
- **Direct answer to dynamic UIs.** Real computer use involves video calls, streaming feeds, async dialogs. Static-snapshot CUAs mis-handle all of these; a continuous watcher is the structural fix.
- **Measured wins.** 33% reduction in non-reasoning wall-clock time on OSWorld-Verified at parity success. 4.7-point task-success lift on DynaCU-Bench with a large VLM.

## Gotchas & tricks

- **Watcher must be cheap.** If the watcher costs more than ~5% of the main agent's compute, the economics invert. 0.8B is a deliberately small target.
- **Interrupt/void thresholds matter.** Trigger-happy interrupts churn the main agent; too-conservative thresholds miss the point. The paper tunes per-benchmark; expect to retune per deployment.
- **Watcher failure modes compound.** A watcher that misses "wake" stalls the agent indefinitely. Health-check timeouts on the control loop are load-bearing.
- **Not a replacement for a good main agent.** Proactive perception improves task success on dynamic UIs but can't fix a main agent that lacks the capability. Think of it as a scheduler, not a planner.
- **Composes with existing CUA training.** You can train the main agent with normal CUA-RL and attach a pretrained watcher at deployment; the two don't need to be co-trained.

## Sources

- Paper: *Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents* — Shao, Liu, Huang, Yin et al. (NUS, A\*STAR CFAR, SenseTime, HKBU), 2026 — [arXiv:2611.28947](https://arxiv.org/abs/2611.28947).

# EgoUseBench
*Depth — benchmark for computer-use agents given live first-person visual context from a wearable, separating offline from online ego-stream regimes.*

**TL;DR:** EgoUseBench evaluates **Ego Use** — computer-use agents (desktop/mobile automation) that also consume a live first-person video stream from a wearable like smart glasses. 693 tasks across three regimes (pure Computer-Use, Offline Ego Use with recorded ego context, Online Ego Use with live-updating ego context), built from 74+ hours of real-world first-person recordings. Online Ego Use success rates are **26.3% lower on average** than pure Computer-Use, exposing how brittle current agents are when their first-person context keeps changing with user intent.

**Prereqs:** [README](README.md), [../agents/README.md](../agents/README.md)
**Related:** [livecodebench](livecodebench.md) · [humaneval](humaneval.md)

---

## What it is

Three evaluation settings under one task definition:

| Regime | Ego context | Harder because |
| --- | --- | --- |
| Computer-Use | None | Baseline: standard screen-based computer-use benchmark |
| Offline Ego Use | Pre-recorded first-person video | Agent must interpret static ego footage before acting |
| Online Ego Use | Live-updating first-person stream | Agent must interleave perception updates with ongoing action |

The task definitions are stable across regimes; only the ego-context availability changes. This lets the benchmark isolate *context-evolution* as a difficulty axis, independent of task complexity.

Tasks are derived from 74+ hours of real-world first-person recordings — user wearing a smart-glasses rig while doing errands, shopping, repairs, navigation. 693 tasks total, spanning productivity, information retrieval, and physical-world-informed computer actions.

## How it works

For each task, the agent is given:

- An instruction (natural-language goal).
- Access to a computer-use environment (keyboard, mouse, screen).
- In Offline/Online regimes: an ego video feed.

In **Online Ego Use**, the ego feed continues to update as the user (simulated or replayed) moves through the environment. The agent must decide when to re-query the stream, how to align its understanding with the changing visual input, and how to not lose the thread of the computer task while ego context shifts.

Success is scored by task-specific rule checks — the agent reaches the correct end state on the computer, consistent with what the ego stream revealed.

## Why it matters

- **Live wearable copilots are a near-term deployment.** The computer-use agent field has been evaluated on static screens; wearables add a *non-stationary* visual stream that most current agents cannot handle. EgoUseBench is the first benchmark that quantifies the gap.
- **Separates perception-update from execution.** The three regimes isolate exactly where current agents collapse: not on screen interpretation (strong) and not on offline multi-modal context (manageable), but on *keeping task state aligned with live ego context*. The 26.3% gap is specific to this failure mode.
- **Real-world ego footage, not synthetic.** 74+ hours of real recordings makes the benchmark robust to distribution shift from purely synthetic or screen-recorded data.

## Gotchas & tricks

- **Don't confuse Offline and Online.** Offline Ego Use with a pre-recorded stream is substantially easier than Online — the agent can seek, re-watch, and reason about the complete video. Online streams allow none of this. Report numbers separately.
- **Score drift across ego-rig hardware.** Field-of-view, resolution, and motion-blur characteristics vary across wearable rigs. The paper ships recordings from a single rig; cross-rig generalization is an open question.
- **Agents that plan heavily on ego context can overfit.** The best current performers tend to ignore ego context when it is ambiguous and fall back to pure computer-use — effectively giving up on the hardest regime. A benchmark average can mask this.
- **Latency matters for Online.** An agent that "catches up" by batch-processing ego context every $k$ seconds trades accuracy for throughput; the benchmark does not fix a latency budget, so results across systems are not always comparable without reading the paper's setup.

## Sources

- Paper: *EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts* — AI4GC Lab, Zhejiang University, 2026, arXiv 2611.34466.
- Related: [../agents/context-management.md](../agents/context-management.md) — context-management techniques are directly implicated by the Online regime results.
- Related: [../multimodal/README.md](../multimodal/README.md) — multimodal context handling is the underlying capability.

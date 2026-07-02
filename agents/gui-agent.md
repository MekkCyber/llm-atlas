# GUI Agent
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **GUI agent** is a vision-language model that completes user tasks end-to-end by interacting with a graphical interface — tapping, swiping, entering text, navigating — instead of calling structured APIs. The dominant training bottleneck is that offline / simulated trajectories don't cover the messy state distribution of *real* devices (account states, permission dialogs, payment auth, risk control), so recent recipes (Xiaomi-GUI-0, 2026) train inside a **real-device closed loop** with an error-driven data flywheel.

**Prereqs:** [README](README.md)
**Related:** [../multimodal/README](../multimodal/README.md) · [../post-training/rlvr](../post-training/rlvr.md) · [../post-training/grpo](../post-training/grpo.md) · [../case-studies/xiaomi-gui-0](../case-studies/xiaomi-gui-0.md)

---

## What it is

A GUI agent operates the same interface a human user would: it sees screenshots or the accessibility tree, decides an action from a fixed action space (tap, swipe, type, scroll, back, home), and executes it against the OS. Round-trip until the task is done or a time budget is hit.

Two prevailing training substrates:

- **Offline / simulated.** Screenshot-action trajectories collected from human demonstrations or scripted sim environments. Cheap but under-represents production distribution.
- **Real-device closed loop.** Actions execute against a fleet of real phones (or emulators plus real accounts); the agent's rollouts land in the *actual* app, populating the model's training distribution with all the states real users see.

## How it works

### Action space and observation

Actions are discrete: `tap(x, y)`, `swipe(x1, y1, x2, y2)`, `type(text)`, `back()`, `home()`, `long_press(x, y)`, `wait(ms)`. Observations are (usually) rendered screenshots, sometimes augmented with the accessibility tree.

### The training pipeline

Typical multi-stage recipe (Xiaomi-GUI-0 shape):

1. **Supervised fine-tuning (SFT).** Train the VLM on a large corpus of `(screenshot → action)` pairs. Data mixes:
   - **High-frequency common tasks** (open app, search, buy).
   - **Long-tail intent generalization** data (rare but plausible user intents).
   - **Capability-enhancement** data with reflection and memory traces.
2. **Step-level RL.** Reward each single action (correct tap, correct swipe direction). Cheap; teaches per-step precision.
3. **Agentic RL.** Reward the full trajectory outcome (task success). Long-horizon credit assignment (see TRIAGE-style role-typed credit).

### The error-driven data flywheel

Deploy the current agent to real devices, log every failed trajectory, generate a *corrected* trajectory (either via human annotators or a stronger model), append to the SFT corpus. Loop. This is the mechanism that lets the model close on the real-world distribution over rollouts, not over benchmark curation.

## Why it matters

- **Bridges the benchmark–usability gap.** Standardized GUI benchmarks (AndroidWorld, WebShop) score high on models that ship poorly in real apps because they never trained on real-app abnormal states. Real-device training closes this gap by construction.
- **Shipping product.** Xiaomi HyperOS, Apple Intelligence, and Google Astra all rely on this class of agent. Xiaomi-GUI-0 is one of the first tech reports that documents the real-device training stack for a shipping-scale device fleet.
- **Composable with GRPO / TRIAGE.** Any RL algorithm that works for LLM reasoning works here — GRPO for policy optimization, TRIAGE-style credit assignment for agentic long-horizon RL.

## Gotchas & tricks

- **State drift.** Real devices reshape state constantly (notifications, ads, permission dialogs). The agent must be robust to interstitials that don't correspond to its plan; the failure-log flywheel is how that robustness is trained.
- **Payment / auth risk.** Real-device closed loops touch real accounts. Sandbox aggressively; anonymize payment methods; put risk controls on the automation layer, not just the model.
- **Reward shaping is dominated by trajectory success.** Step-level rewards help precision but plateau quickly; the long-horizon RL stage is where end-to-end task success comes from.
- **Screenshot vs a11y tree.** Rendering may fail on dynamic content; the a11y tree may lag; consuming both is more robust than either alone.
- **Distribution shift across OS versions.** Every OS release changes UI conventions; expect to retrain on a rolling window.

## Sources

- Paper: *Xiaomi-GUI-0 Technical Report* — Cao, Duan, Fu, Gao, Lian et al., 2026 — Xiaomi; real-device closed-loop training for a native multimodal GUI agent. See the [case study](../case-studies/xiaomi-gui-0.md).
- Related: AndroidWorld, WebShop, RealMobile benchmarks — the substrate evals.

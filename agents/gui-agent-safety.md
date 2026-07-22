# GUI Agent Safety via World-Model Prediction (SeerGuard)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Mobile GUI agents can take irreversible actions from a single misclick, and existing safety mechanisms are reactive. SeerGuard sits *in front* of every proposed action with a two-stage screen: (1) an **instruction-level** filter refuses obviously-bad tasks before execution; (2) an **action-level** safety-augmented world model (SAWM) predicts the next GUI state given the proposed action and scores the risk of that predicted state. If the predicted outcome trips a threshold, the action is blocked.

**Prereqs:** [_attacks.md](../safety/_attacks.md)
**Related:** [cot-monitoring.md](../safety/cot-monitoring.md) · [safety-case.md](../safety/safety-case.md)

---

## What it is

A pre-execution safety framework for GUI agents. It doesn't retrain the underlying agent; it wraps it with two learned gates that share a single multi-task world model.

## How it works

- **SAWM (Safety-Augmented World Model).** A single model trained jointly on two tasks:
  - *Next-state prediction* — given the current GUI screen and a proposed action, predict the next GUI screen (as tokens or as a structured state).
  - *Risk scoring* — given the predicted next state, output a risk score.
- **Instruction-level screen.** Before the agent starts, SeerGuard classifies the user instruction as safe / unsafe / ambiguous. Unsafe requests are refused up front.
- **Action-level assessment.** On each proposed action, SAWM predicts the next state and its risk. Actions above the risk threshold are blocked or escalated.

Multi-task training means the risk head sees the *predicted* state (as SAWM would have imagined it), not just the ground-truth state — the two heads share representations and improve each other.

## Why it matters

GUI agents are shipping into production (Anthropic's computer-use, Apple Intelligence, Manus), and "one wrong action ruins your account" is a real failure mode. Pre-execution simulation-plus-risk with a *joint* world model is a clean framing that decouples the safety layer from the underlying agent — you can put SeerGuard in front of any GUI agent without retraining it. The paper reports safety-utility rising 0.191 → 0.596 at ω=0.8 and risk-cost dropping 0.347 → 0.130 at α=0.8 on Qwen3-VL-8B-Instruct, with generalization across multiple GUI-agent backbones.

## Gotchas & tricks

- SAWM's next-state prediction fidelity caps the whole framework: if it hallucinates, the risk head scores hallucinations. Training with adversarial state sequences helps.
- Threshold tuning is a live tradeoff — over-blocking annoys users, under-blocking defeats the point. The paper's ω/α parameterization is meant to make this explicit.
- The framework does nothing about *sequences* of individually-safe actions that jointly harm (e.g., N small transfers). A stateful monitor on top would be needed for that.
- Overlaps with prompt-injection defenses but doesn't replace them — SeerGuard scores *the agent's own actions*; it doesn't attribute them to a user vs. attacker source.

## Sources

- Paper: *SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction* — authors not shown on HF page (JIUTIAN Research), 2026 — [arXiv:2607.15550](https://arxiv.org/abs/2607.15550) · [HF](https://huggingface.co/papers/2607.15550)

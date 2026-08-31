# RLHEV — Reinforcement Learning with Human-Engine Verification
*Depth — the spatial-generation analogue of RLVR: engine as verifier, developer as global signal.*

**TL;DR:** World models scale poorly on scraped video because reward signals (CLIP scores, aesthetic models) are fuzzy and biased. RLHEV replaces them with a game engine's own executable checks — collision, physics, navigability, playability — plus implicit acceptance feedback from the developer building the scene. Same recipe that made [RLVR](rlvr.md) work for code / math, extended to spatial world modeling.

**Prereqs:** [rlvr.md](rlvr.md), [_rl.md](_rl.md), [_rewards.md](_rewards.md)
**Related:** [../agents/_world-models.md](../agents/_world-models.md)

---

## What it is

A post-training paradigm for **spatial world models** (video / 3D scene generators). It borrows the RLVR pattern — an executable verifier providing dense rewards — but the verifier is a game engine: the same engine the model's outputs will be consumed by. A generated scene compiles into an engine specification; the engine then runs its usual checks and returns a per-check reward vector. A human developer's accept/reject on the final scene supplies a coarse global signal.

## How it works

Two reward sources are combined:

1. **Engine rewards (dense, cheap, per-step).**
   - Collision-free geometry.
   - Physics-valid dynamics (no tunneling, stable rest states, valid mass / friction).
   - Navigability (agent path exists between required points).
   - Bounded playability (game loop reaches goal within budgets).
   Each is a scalar the engine can compute in milliseconds without human input.
2. **Human acceptance (sparse, high-signal, global).**
   - The developer's real accept/reject in the actual dev workflow.
   - Treated as a global success token, weighted more than engine rewards but arriving orders of magnitude less often.

Training data: long-horizon developer trajectories where each intermediate spec plus the final accept/reject gives (state, action, reward, terminal-signal) tuples. RL is standard on top — the innovation is the reward source, not the algorithm.

## Why it matters

Spatial generation has lacked a "compiler." Text / code have deterministic verifiers (unit tests, type checkers); math has answer keys. Aesthetic reward models for video are noisy, biased, and cost-per-signal is high. A game engine gives the field a cheap, deterministic, semantically-rich verifier — and the developer workflow supplies naturally-generated preference data at zero label cost.

## Gotchas & tricks

- Engine rewards over-optimize easily — a model that memorizes "empty scene" trivially satisfies collision and physics but fails human acceptance. Weight the human signal strongly.
- Not all failure modes are engine-detectable (art-quality, coherence with prompt); consider hybrid rewards with a vision-based check.
- Requires the target domain to *have* an engine. Cross-domain transfer (engine-trained model → real-world video) is an open question the paper flags as future work.
- Complements aesthetic reward models rather than replacing them — think of the engine as the safety net that rules out nonsense.

## Sources

- Paper: *Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models* — Zhou et al., 2026 — [arXiv:2608.25518](https://arxiv.org/abs/2608.25518)

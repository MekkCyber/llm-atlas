# EgoCS-400K — Egocentric Counter-Strike Dataset for World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 10,000-hour Counter-Strike replay corpus aligned as **video-action-language trajectories**: 400K+ first-person clips with synchronized actions, camera motion, state vectors, and event labels. Built specifically for interactive world models that need more than captioned video — action-conditioned prediction, state-aware generation, and egocentric action understanding all become tractable on a single source.

**Prereqs:** [_data-curation](_data-curation.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Generative video models can train on captioned clips. **Interactive** world models — models that *react* to player / agent actions — need temporally aligned trajectories of (video, action, state, event) tuples. That kind of data is expensive to collect from the real world.

EgoCS-400K argues that game replays — specifically Counter-Strike (CS) — are the cheapest available source of clean, well-aligned ego-trajectories at scale. The CS engine ships deterministic replay APIs that expose per-frame ground-truth actions, camera transforms, game state, and event labels. The dataset turns 10,000 hours of replays into a corpus designed for world-model training.

---

## How it works

### Source data

Counter-Strike professional and amateur replays, processed through the game's replay API to extract:

- **First-person video** at gameplay frame rate.
- **Player actions** (discrete: shoot, jump, reload, weapon switch; continuous: mouse look, WASD).
- **Camera motion** (the head pose / view direction per frame, from the engine's deterministic state).
- **Game state vectors** (HP, weapons, equipment, round timer, etc.).
- **Event labels** (kills, deaths, defuses, round transitions).

Everything is temporally aligned because the engine produced it; no inferred labels.

### Scale

- 10,000+ hours of gameplay total.
- 400K+ ego-clips after segmentation.
- Multiple maps, weapons, game modes for diversity.

### Three supported task tracks

The dataset is designed to support (and ship baselines for):

1. **Action-conditioned video prediction** — given prior frames and an action sequence, predict the next frames.
2. **State-aware scene generation** — given a state vector, generate consistent ego-views.
3. **Egocentric action understanding** — given an ego-video, classify the player action and inferred intent.

The same trajectories serve all three; only the conditioning differs.

---

## Why it matters

- **Largest available action-aligned ego-video corpus.** Other ego datasets (Ego4D, Epic-Kitchens) are larger in raw hours but lack ground-truth actions and state. EgoCS-400K trades real-world diversity for *clean alignment*.
- **Three task tracks on one corpus.** Replay-grounded data is the cheapest way to get the (video, action, state, event) tuples interactive world models need; reusing one source across three tracks amortizes the collection cost.
- **Counter-Strike specifically.** Discrete-action + deterministic-state + replay-API combination is unusually friendly for ML compared to most games (continuous-action arcade titles, stateful narrative games).

---

## Gotchas & tricks

- **Domain gap.** A world model trained on CS will not transfer to real-world ego-video without further work. The dataset is best understood as a *benchmark and pre-training source* for world-modeling techniques, not a substitute for in-domain data.
- **Player behavior is not uniform.** Pro replays differ in style and tempo from amateur play. Pre-training mixes both; downstream evaluation should specify the slice.
- **Visual content has copyright considerations** (textures, models from the game). Released for research; commercial use of derived models needs review.
- **Sparse-event tracks.** Some labels (defuse, ace) are rare and need oversampling for balanced training on event-conditioned tasks.

---

## Sources

- Paper: *EgoCS-400K: An Egocentric Gameplay Dataset for World Models* — Dong Liang, Yuhao Liu, Fang Liu, Tianyu Huang, Gerhard P. Hancke, Rynson W. H. Lau, City University of Hong Kong, 2026 — [arXiv:2606.18180](https://arxiv.org/abs/2606.18180).

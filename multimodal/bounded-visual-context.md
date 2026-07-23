# Bounded visual context
*Depth — a four-part fixed-budget context recipe for long-horizon video world models.*

**TL;DR:** A video world model that rolls out autoregressively over many minutes cannot attend to all prior frames. AlayaWorld's answer is a **bounded visual context**: a fixed-budget representation combining four ingredients that each address a different failure mode of naive attention-over-history — a **persistent sink frame** (long-term anchor), **compressed temporal history** (fixed-budget summary), **geometry-aligned spatial memory** (view-consistent map), and **recent-frame conditioning** (short-horizon continuity).

**Prereqs:** *(none in current graph)*
**Related:** [../case-studies/alayaworld.md](../case-studies/alayaworld.md), [self-training-drift-reduction.md](./self-training-drift-reduction.md)

---

## What it is

Long-horizon video world models drift and forget: characters change identity, rooms disappear, camera loops fail to close. Naive fixes ("larger context window", "add more recent frames") pick up either the *long-term* or the *short-term* failure mode, not both. Bounded visual context is a **composition** of four different memory ingredients, each with a specific role — the whole point is that no single mechanism is enough.

Introduced by AlayaWorld (2607.18367) as the architectural core of its 15B video DiT.

## How it works

Four ingredients concatenated into a fixed-budget context each autoregressive chunk sees:

1. **Persistent sink frame.** A single anchor frame carried across the entire rollout — the model's long-term memory of *what the world looks like*. Prevents identity drift over many chunks. Related to the "attention sink" idea from LLMs: a specific token/frame that the model reliably routes through.

2. **Compressed temporal history.** A learned, fixed-size compression of prior chunks. Not the raw frames — a summary. Bounded in tokens/memory so context grows in wall-clock but not in compute per chunk.

3. **Geometry-aligned spatial memory.** A representation organized by scene geometry (camera-aligned / world-aligned), so that returning to a viewpoint retrieves what was seen from it. Critical for camera loop-closure and scene persistence.

4. **Recent-frame conditioning.** The last few actual frames, for short-horizon continuity — the piece the model needs to keep motion smooth without inventing new content.

Each ingredient targets a specific failure mode:

| Ingredient | Failure it prevents |
| --- | --- |
| Persistent sink frame | Long-horizon identity drift |
| Compressed temporal history | Forgetting of medium-term events |
| Geometry-aligned spatial memory | Failure to reconstruct scene on camera return |
| Recent-frame conditioning | Motion/appearance jitter chunk-to-chunk |

## Why it matters

- **Fixed compute per chunk.** The context is *bounded*, not growing — a hard requirement for real-time interactive rollouts.
- **Modular failure modes.** You can ablate one ingredient and predict which failure mode should regress.
- **Transferable pattern.** The recipe generalizes to any autoregressive-over-latents model with long rollouts (games, simulations, embodied agents).

## Gotchas & tricks

- The *composition* is load-bearing. Removing any single ingredient introduces its specific failure mode; the paper's ablations are the honest test.
- Geometry-aligned memory needs actual camera pose signal — with a switchable-text-prompt-only interface, this ingredient can't do its job.
- The "sink frame" is a specific frame, not a general "anchor" — choice of which frame carries the whole rollout matters.

## Sources

- Paper: *AlayaWorld: Interactive Long-Horizon World Modeling* — Zhang, Li, Zhan, Ge, Yin et al. (Alaya Lab), 2026 — [arXiv:2607.18367](https://arxiv.org/abs/2607.18367)

# Graph-Conditioned Video Generation
*Depth — replace pixel-trajectory control with a small *interaction graph* (nodes = subjects, edges = relations) as the video generator's conditioning signal.*

**TL;DR:** Controlling multi-object interactions in video generation with drawn trajectories scales poorly — users must specify accurate tracks per object, which becomes ambiguous under occlusion and expensive as scenes grow. **Graph-conditioned video generation** replaces trajectory sketches with a structured *interaction graph* — small, semantic, unambiguous — as the conditioning signal, and lets the model synthesize consistent pixel motion. Reported to beat trajectory-control baselines like Motion-I2V on FID/FVD with substantially fewer trainable parameters.

**Prereqs:** [../multimodal/README](README.md)
**Related:** [world-state-registers](world-state-registers.md)

---

## What it is

An image-to-video model whose control signal is a small graph:

- **Nodes** = subjects in the initial frame (people, objects, regions), grounded to bounding boxes or segments.
- **Edges** = relations between subjects, typed by an interaction vocabulary (*picks up*, *pushes*, *follows*, *avoids*).

The model consumes the initial frame plus the graph and generates a video that realizes the specified interactions. The user never draws trajectories — the model synthesizes them from the semantic constraint.

## How it works

**Conditioning.** Nodes are embedded as tokens with spatial grounding to the initial-frame regions; edges are embedded as relation tokens attached to their endpoint pair. Graph tokens attend to and are attended by video tokens across the diffusion transformer, so the interaction constraint influences every layer.

**Training data — GraphVid-Bench.** A large-scale interaction-centric video dataset with structured relational annotations (nodes + typed edges per clip). Curated so that the graph annotations are cleanly separable from raw motion — a key requirement for the model to learn *what an interaction means*, not just to memorize trajectories.

**Efficiency.** Because the graph is a highly compressed representation of the intended motion, GraphVid uses substantially less training data and fewer trainable parameters than trajectory-control methods while achieving higher quality.

## Why it matters

- **Cheaper user interface.** A graph is faster to specify than accurate per-object tracks and remains unambiguous under occlusion.
- **Composes with LLMs.** LLM-generated intent naturally lands as a graph (entities + relations), giving a clean handshake between an LLM planner and a video generator.
- **Reported quality gains.** FID reduced by up to 39.9% and FVD by 37.6% vs Motion-I2V; PSNR 9.87→15.98, SSIM 0.38→0.61.

## Gotchas & tricks

- **Relation vocabulary is a design decision.** Too small and the graph can't express real scenes; too large and the model spreads capacity across rare relations. Iterate against the target scene distribution.
- **Node grounding matters.** Ambiguous node-to-region grounding is the main failure mode — the model can't act on a graph whose nodes it can't localize.
- **Not a substitute for trajectory control everywhere.** For scenes where the *specific* path matters (e.g. cinematography), trajectories still win. Graphs excel when it's the *interaction* that matters, not the exact path.
- **Bench dataset is part of the contribution.** Reproduction requires GraphVid-Bench-style annotations; plain video datasets don't have the structured edges the model needs.

## Sources

- Paper: *GraphVid: Interactive Graph-Controllable Video Generation* — Shah, Susladkar, Prakash, Nguyen, Yu, Juvekar, Waheed, Lourentzou (UIUC PLAN Lab), 2026 — [arXiv:2607.21580](https://arxiv.org/abs/2607.21580).

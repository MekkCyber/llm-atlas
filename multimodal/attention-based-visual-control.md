# Attention-Based Visual Control (Training-Free Video World Models)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **training-free** control interface for frozen causal video world models. Instead of fine-tuning a control head, express every user input (camera trajectory, reference points, sparse edits) as **camera- and time-labelled visual patches** placed into the model's attention context. The frozen backbone reads them through its native self-attention; a per-channel CFG scales how strongly the auxiliary evidence influences generation. Introduced by **World-in-World (WiW)** (2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** none (multimodal folder is currently sparse)

---

## What it is

Frozen video world models are strong but hard to steer: adding controls typically means training a control head (ControlNet-style adapters, LoRA, or full re-tuning), and each new capability is a new head. Attention-based visual control replaces the head with an *interface convention*: how you inject additional visual signal into the attention window of a fixed backbone.

The three capabilities the paper demonstrates — re-rendering along a new camera trajectory, extended revisiting of past events, motion transfer — all come out of the same interface with different inputs.

## How it works

Given a frozen causal video model that operates on visual patch tokens:

1. **Encode user input as clean visual states.** Camera trajectories become synthesized viewpoint patches; reference points become identity-labelled patches; sparse edits become target-frame patches. Each carries a **camera label** and a **time label**.
2. **Splice into the attention context.** These patches join the actual frame tokens in the model's KV context; the causal self-attention now attends to them wherever the label alignment says it should.
3. **Correspondence routing.** A lightweight pass propagates identity across frames so a labelled patch attaches to the same object across time.
4. **Evidence-wise Attention CFG.** At each denoising step, the model runs *twice* — once with auxiliary evidence attended to, once without — and the outputs are combined with a per-channel scale, analogous to classifier-free guidance but at the attention-channel level.

No backbone weights change; no control head is trained.

## Why it matters

- **Every new capability is an input, not a head.** Camera motion, revisiting, transfer — same interface, different label conventions. Prior video-control work required a new fine-tune per capability.
- **Inherits the backbone's quality by construction.** Because generation still happens through the frozen model, all fidelity, temporal consistency, and physical plausibility properties are preserved.
- **Cheap to iterate.** Adding a new control mode is prompt / labelling engineering, not a training run. Ideal for research and product exploration on top of an expensive backbone.
- **Generalizes the "ControlNet was fine-tuning" lesson.** For image diffusion, ControlNet's success spawned many alternatives; for causal video, this paper suggests attention-context editing is the analog that doesn't need weight updates.

## Gotchas & tricks

- **Label conventions are the whole game.** Cameras, times, and identities have to be consistent across the injected patches or attention averages them incorrectly.
- **CFG scale per channel.** A single global scale doesn't work — different evidence channels (camera vs. motion vs. identity) need different weights or one channel drowns out the others.
- **Correspondence routing is essential.** Without it, an object's label drifts across frames and the frozen model treats different patches as different objects.
- **Backbone must be causal video.** Non-causal diffusion video models (bidirectional in time) don't support the same time-labelling trick because attention isn't monotonically forward in time.
- **Not a replacement for training when semantics change.** The interface controls *what/where/when* the frozen model attends to; it does not teach new object concepts or physical rules. Those still require training.

## Sources

- Paper: *World in World: Explore the World with World Models* — Song, Yang, Zhang (Westlake AGI Lab), 2026 — arXiv:2609.11548.

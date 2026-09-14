# Training-Free Control for Frozen Video World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Add new controls (viewpoint change, revisit, hole-filling) to a **frozen causal video world model** *without training*. Convert heterogeneous control signals into camera- and time-labelled clean visual states injected into the model's KV cache; the frozen model reads them through its own native self-attention, so no adapter, no LoRA, no fine-tune.

**Prereqs:** [README](README.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [README](README.md)

---

## What it is

Autoregressive video world models let you explore a scene interactively — but only within the controls they were trained on. Adding a new control (say, arbitrary camera trajectories) has meant per-task modules or adapter fine-tunes, both of which touch weights and defeat "release once, control anywhere".

World in World reframes control as *evidence injection*. Any control signal (camera pose, timestamp, source-video frame) is packaged into a clean visual state labelled with `(camera, time)` and inserted into the frozen model's KV cache. Because the model's self-attention was trained to attend over exactly this kind of visual token, it happily attends over the injected ones — the interface is just a preparation stage, not a new module.

## How it works

- **Evidence packaging.** Convert each control signal into a clean visual state (a decoded frame or a token embedding) with a `(camera_pose, timestamp)` label. Labels use whatever pose/time embedding scheme the base model already understands.
- **KV-cache injection.** Insert the evidence tokens into the frozen model's context at their labelled positions, alongside the model's own generated tokens.
- **Native attention reads.** The frozen model's self-attention treats the injected tokens as ordinary context, so the roll-out incorporates them without any new parameters.
- **Zero training.** No adapter, no LoRA, no gradient step. Everything happens at inference.
- **Composability.** Multiple control types (camera + time + reference frame) can be injected simultaneously as long as they share the label vocabulary the base model expects.

## Why it matters

Video world models are pushed forward by ever-larger training runs; a training-free control interface changes the deployment story — you can layer new controls onto a released frozen model without touching weights. It also decouples model providers from control innovators: the same 2B/8B video model can support new interaction modes indefinitely.

## Gotchas & tricks

- **Label scheme must match the base model.** The camera/time labels have to reuse whatever positional / conditioning encoding the pretrained model already knows. Inventing new label formats defeats the training-free premise.
- **Clean visual states, not noisy latents.** Injecting noisy diffusion latents confuses causal attention; the paper uses fully denoised, clean visual tokens.
- **Not all controls are "evidence-shaped".** Continuous control signals (joystick, style knob) don't map naturally into visual states — this interface fits scene-anchored controls best.
- **Position matters.** Insertion index inside the causal sequence determines whether the model treats the evidence as *past* (memory) or *future prompt* (goal). The two behave very differently.
- **KV-cache mutation is stateful.** In serving, injecting evidence mid-stream can conflict with cached prefixes; plan the eviction/refresh policy.

## Sources

- Paper: *World in World: Explore the World with World Models* — Song, Yang, Zhang (Westlake AGI Lab), 2026 — https://arxiv.org/abs/2609.11548

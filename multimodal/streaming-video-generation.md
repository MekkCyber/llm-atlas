# Streaming Video Generation
*Depth — generate video frames incrementally as new conditioning arrives, without redrawing the world every step.*

**TL;DR:** Native-streaming interactive video generators (Wan-Streamer, minWM, etc.) produce frames continuously in response to a live event stream — user actions, camera moves, dialog. The design challenge is *world drift*: as the model conditions on new events, it quietly rewrites the scene, characters, and ambient conditions. The current best fix is an explicit split: video = world + event stream, with separate conditioning pathways so a slow-changing persistent world is not re-derived from event tokens every step.

**Prereqs:** *(diffusion / video generation basics; no depth prereq file yet)*
**Related:** [README](README.md)

---

## What it is

A generator that emits video frames online, one at a time (or in short chunks), conditioning each new frame on a growing event stream (actions, prompts, sensor readings) while keeping the underlying scene coherent. Distinguishes itself from *offline* text-to-video (fixed prompt → whole clip) by the temporal open-endedness of the condition.

## How it works

### The world/event split

Wan-Streamer v0.3 formalizes the design as `video = world + event stream`:

- **World.** Persistent context — environment, scene layout, subjects, ambient acoustics, voice characteristics. Slow-changing or fixed for the whole session.
- **Event stream.** Per-frame local actions and short-term changes — a character moves, a light flickers, a line of dialog.

Two separate conditioning pathways feed a shared frame decoder. The world is encoded once and re-attended to every step; the event stream is streamed in incrementally. The decoder does not have to reconstruct the persistent world from event tokens on each step, which was the failure mode that drove world drift.

### Streaming attention

Frames typically share attention state across time — either a persistent KV cache extended per step, or a sliding window that keeps a recent past visible. The exact mechanism varies (some systems use a diffusion transformer with rolling attention; others distill a diffusion generator into a fast per-frame policy).

### Consistency losses

Training uses a mix of (a) short-horizon frame reconstruction, (b) long-horizon world-preservation losses that penalize scene drift over many frames, and (c) event-following losses that force per-frame outputs to actually reflect the event stream.

## Why it matters

- **Interactive video is a distinct target.** Offline text-to-video does not compose into good streaming systems; the design constraints (world persistence, per-frame latency, online conditioning) are different enough to justify their own architecture family.
- **Enables live agents-in-video.** Streaming generation is the substrate for game-engine-flavored world models, video chat avatars, and interactive simulators — none of which work with offline generators.
- **Clean separation of concerns.** The world/event split matches how humans think about video and how game engines model scenes, giving a clean interface for downstream applications.

## Gotchas & tricks

- **World drift is stealthy.** It manifests as slow degradation of scene identity over minutes, not as a single obvious failure. Standard per-clip FID does not catch it — long-horizon evaluation with per-scene reference frames is needed.
- **Latency budget dominates architecture choice.** Real-time streaming caps the diffusion step count severely; step-distillation and consistency models are usually mandatory downstream of the base diffusion.
- **Event-stream bandwidth.** How much conditioning per frame the model can actually integrate is limited — pushing large event blobs per step retrains world drift under a different guise.
- **KV cache management.** Persistent attention state across arbitrarily long sessions requires paged or windowed KV; unbounded caches OOM before the world drifts.

## Sources

- Paper: *Wan-Streamer v0.3: Video = World + Event Stream* — Huang et al., 2026 (Wan / Alibaba) — the world/event split.
- Related: earlier interactive video world models (Genie 2, minWM, etc.).

# Vision-Language-Action (VLA) Models
*Depth — a foundation-model class that takes images + language and emits robot actions.*

**TL;DR:** A VLA is a large vision-language model whose output head produces **actions** (robot joint commands, discrete end-effector commands, or latent motion tokens) instead of text. It inherits the LLM / VLM pretraining stack but adds an action decoder and an action-data mixture during training. The canonical examples are **RT-2 / RT-X** (Google, 2023), **OpenVLA** (Stanford, 2024), **π0** (Physical Intelligence, 2024); more recent entries like **HuGe** (Westlake / BUPT consortium, 2026) scale to **humanoid control from egocentric human video**. VLAs are the dominant paradigm for scaling general-purpose robot policies in the LLM era.

**Prereqs:** *(none)*
**Related:** [surface-normal-conditioning.md](surface-normal-conditioning.md), [../agents/README.md](../agents/README.md)

---

## What it is

Architecture-wise, a VLA is three stacked parts:

```
images + text ──► vision encoder (CLIP / SigLIP / native)
             ──► Transformer backbone (VLM-pretrained)
             ──► action head
```

The action head is where VLAs differ from VLMs:

| Head family | Output |
| --- | --- |
| **Discrete action tokens** (RT-2) | tokenize joint deltas into a vocabulary; predict with the LM head |
| **Continuous diffusion** (π0, Octo) | diffusion decoder over continuous action chunks |
| **Latent motion tokens** (HuGe) | predict latent codes that drive a *frozen* pretrained low-level controller |

Training data mixes robot teleop trajectories, egocentric human video (with pseudo-proprioception), simulation rollouts, and auxiliary vision-language data to keep the VLM backbone sharp.

---

## How it works

### Three training stages (modern recipe)

```
1. Pretraining:  (image, text) + (video, text)  — standard VLM
2. VLA adaptation: cross-embodiment robot data + egocentric human video
3. Task fine-tuning: small per-task teleop demonstrations
```

Egocentric human video is the scaling dimension that matters now: it is orders of magnitude more plentiful than teleop. HuGe (2026) demonstrates ~2,000+ hours of egocentric data as a usable VLA pretraining corpus, with simulation playback supplying pseudo-proprioception for frames lacking robot state.

### Action-chunk prediction

Modern VLAs predict **chunks** of actions (e.g., 10–50 timesteps) rather than single actions. The chunk size is a tradeoff: longer chunks reduce the LM inference rate the robot has to sustain, shorter chunks are more reactive.

### Latent-action heads

The latent-motion-token approach decouples the VLA from the low-level controller:

```
VLA → latent motion tokens → frozen whole-body controller → joint commands
```

The VLA only has to learn a *compact* intent representation; the controller handles embodiment-specific dynamics. This is HuGe's main architectural choice and makes cross-embodiment transfer much easier than predicting joint commands directly.

---

## Why it matters

- **Scales robot policies the way LLMs scale text models.** Pretrain on the big noisy corpus (human video + web images), adapt on the small clean one (robot teleop), fine-tune per task.
- **Transfers across embodiments.** A VLA pretrained on multi-embodiment data (RT-X, Open X-Embodiment) outperforms single-embodiment baselines on each target robot.
- **Imports LLM reasoning for free.** A VLA backbone that was pretrained on CoT text can emit a plan as intermediate tokens before producing actions. Not all VLAs use this, but it is a free upside of the architecture.
- **Egocentric human video is the next data frontier.** HuGe shows that visually-adapted human demonstrations improve a Unitree G1 pick-and-drop task over robot-only data.

---

## Gotchas & tricks

- **Visual distribution shift is the biggest sim-to-real gap.** Robot teleop videos look nothing like egocentric human videos (color, lens, lighting). HuGe addresses this with a re-rendering step that modifies arm appearance in human footage to look humanoid.
- **Action discretization choice is load-bearing.** Fine tokenization (small bins) gives precise control but blows up the vocabulary; coarse tokenization is cheap but imprecise. Diffusion heads side-step this but cost more at inference.
- **VLM backbone forgets if it only sees robot data.** Keep a VLM-data mixture in the VLA training phase to prevent language-capability regression.
- **Real-robot inference latency matters.** Chunked prediction + a small action head on top of a frozen controller is the fast path. A full LLM forward at 10 Hz on a humanoid is still hard.
- **Pseudo-proprioception from simulation is noisy.** Pre-labelled training frames with simulated proprioception help, but the labels are approximate. Weight the loss accordingly.

---

## Sources

- Paper: *RT-2: Vision-Language-Action Models* — Brohan et al., Google, 2023 — the term "VLA".
- Paper: *Open X-Embodiment / RT-X* — Open X-Embodiment Collaboration, 2023 — cross-embodiment VLA data.
- Paper: *OpenVLA* — Kim et al., Stanford, 2024 — open-source VLA baseline.
- Paper: *π0 / π0-FAST* — Physical Intelligence, 2024 — diffusion-head VLA.
- Paper: *HuGe: Scaling Humanoid Generalist VLAs with Egocentric Experience* — Liu et al., Westlake / BUPT consortium, 2026, [arXiv 2610.34186](https://arxiv.org/abs/2610.34186) — humanoid VLA from egocentric human video.

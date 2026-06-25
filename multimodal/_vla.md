# Vision-Language-Action Models (VLAs)

*Taxonomy — multimodal foundation models that emit physical actions (joint torques, end-effector targets, or discretized motion tokens) in addition to text.*

**TL;DR:** A VLA is a VLM with an action head. The category emerged when teams realized that the pretrained visual-language stack from MLLMs gives a robot policy a much stronger prior than starting from scratch — and that, with the right action representation, a single multimodal transformer can drive manipulation, navigation, and trajectory prediction across embodiments. Modern designs differ on (i) the action representation (discretized tokens vs. continuous decoder vs. DiT-style action diffusion), (ii) the memory architecture (no memory vs. learned sparse memory), and (iii) the training mix (robot trajectories vs. web video vs. simulation).

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [sparse-keyframe-memory](sparse-keyframe-memory.md)

---

## The problem

A robot policy trained from scratch on robot demonstrations is data-starved: collecting trajectories is expensive and embodiment-specific. A VLM trained on internet image-text data knows nothing about actions but has rich visual and semantic priors. VLAs bridge the two — bolt an action head onto a pretrained VLM, fine-tune on robot trajectories, and inherit the VLM's prior.

Three challenges shape the design space:

- **Action representation** — continuous joint torques are smooth but hard for transformers; discretized motion tokens are tokenization-friendly but lossy; DiT-style continuous-action diffusion is the current frontier.
- **Embodiment generalization** — different robots have different joint layouts; how does one VLA serve many?
- **Long-horizon memory** — manipulation tasks span many seconds; the VLA needs to remember occluded cues.

## The shared pattern

Every VLA has the structure:

```
visual tokens + text tokens → transformer backbone (initialized from a VLM) → action head
```

The variants differ in what the action head looks like and how memory across many frames is handled.

## Variants

| Model family | Action head | Memory | Notes |
| --- | --- | --- | --- |
| RT-2 | Discretized action tokens emitted by LM head | None (single-frame conditioning) | First VLA to share a vocabulary with text |
| OpenVLA | Discretized action tokens | None | Open-source extension of RT-2 |
| π0 | Flow-matching continuous action head | Limited (short window) | Physical Intelligence's flagship; continuous action |
| Qwen-VLA | DiT-based action-and-trajectory decoder + embodiment-aware prompt | None | Unified across manipulation, navigation, trajectory |
| EventVLA | (action head inherited from base VLA) | [Sparse keyframe memory](sparse-keyframe-memory.md) | Adds learned memory selection for long-horizon tasks |

## How to choose

- **For research baselines:** OpenVLA — open weights, reasonable performance.
- **For continuous-action manipulation:** π0 or Qwen-VLA — discretized tokens leave performance on the table for fine motor control.
- **For long-horizon tasks:** add a [sparse-keyframe-memory](sparse-keyframe-memory.md) module on top of a backbone VLA; otherwise expect failure modes around occluded or transient cues.
- **For multi-embodiment generalization:** Qwen-VLA's embodiment-aware prompt conditioning is the cleanest demonstrated recipe.

## Adjacent but distinct

- **VLMs** without action heads — same backbone, no embodied behavior.
- **Classical robot controllers** (operational space, MPC) — non-learned, complementary to VLAs (often run downstream of a VLA's action targets).
- **Generic multimodal agents** that emit *digital* actions (clicks, taps) — see [../agents/_gui-agents.md](../agents/_gui-agents.md). Same loop shape; different action space.

## Sources

- Paper: *RT-2: Vision-Language-Action Models* — Brohan et al., Google, 2023.
- Paper: *OpenVLA* — Kim et al., 2024 — open-source RT-2-style VLA.
- Paper: *π0: A Vision-Language-Action Flow Model for General Robot Control* — Black et al., Physical Intelligence, 2024.
- Paper: *Qwen-VLA: Unifying Vision-Language-Action Modeling* — Qwen / Alibaba, 2026.
- Paper: *EventVLA: Event-Driven Visual Evidence Memory* — Yang et al., 2026 — [arXiv:2606.20092](https://arxiv.org/abs/2606.20092).

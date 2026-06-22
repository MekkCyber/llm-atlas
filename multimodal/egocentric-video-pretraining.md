# Egocentric-Video Pretraining for Embodied Models
*Depth — pretraining embodied foundation models on filtered egocentric human video as a substitute for teleoperated robot data.*

**TL;DR:** Teleoperated robot trajectories are the dominant pretraining source for embodied foundation models, but they're slow and expensive to collect. **HumanScale** (Ma et al., 2026) runs a controlled head-to-head: with matched data volume and a careful **egocentric filtering + labeling pipeline**, pretraining on **egocentric human video** beats pretraining on teleop trajectories — **−24% validation loss** on real-robot action prediction, **+52.5% / +90%** in-distribution and out-of-distribution real-robot task success.

**Prereqs:** [README.md](README.md)
**Related:** [../data/quality-filtering.md](../data/quality-filtering.md), [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Embodied foundation models (VLAs) want LLM-style data scaling but are bottlenecked by **how much labeled action data** they can get. Teleop is the standard source — a human directly controls the robot and the trajectory + actions are logged — but it's expensive, low-diversity, and acquisition-rate-limited.

**Egocentric human video** (head-mounted camera recordings from real human activity: Ego4D, Aria, EgoPet, household ego-datasets) is **abundant** but lacks robot-aligned action labels. The question HumanScale answers head-to-head: can it substitute, and at what cost?

The recipe is **"pretrain on filtered egocentric, adapt on a small amount of robot data."** No claim that you can skip robot data entirely — only that the bulk of pretraining can shift to egocentric.

## How it works

### Filtering + labeling pipeline (the load-bearing piece)

Raw egocentric video is too noisy to use directly. HumanScale's pipeline filters along four axes:

| Filter | Why |
| --- | --- |
| **Motion quality** | reject blurry or jittery clips that confound action inference |
| **Hand–object interaction** | keep clips with clear interaction; discard locomotion-only clips |
| **Viewpoint plausibility** | match camera height/angle distribution to robot embodiments |
| **Caption / instruction extraction** | label each clip with a short text instruction via a VLM caption + LLM cleanup |

Output: ~hundreds-of-hours of high-quality, instruction-labeled egocentric clips.

### Pretraining objective

Standard VLA-style autoregressive next-token prediction with **action tokens omitted** for egocentric clips (since they lack robot actions). The model still learns:

- Visual dynamics (how scenes evolve under interaction).
- Hand pose + object affordances (because frames show them).
- Text-grounding to interaction (because instructions are aligned).

### Robot-data adaptation

After egocentric pretraining, fine-tune on a **small amount** of teleop robot data with full action supervision. This bridges the action-space gap: the model already knows what manipulation *looks like*; the robot data only needs to align the prediction to the robot's actuator commands.

## Why it matters

- **−24% validation loss** on real-robot action prediction at matched pretraining data volume.
- **+52.5%** in-distribution and **+90%** out-of-distribution real-robot success rate.
- Lifts the embodied-foundation-model data ceiling by an order of magnitude — egocentric data exists at hundreds of thousands of hours, teleop at tens of thousands.
- Suggests teams should **assess egocentric data quality** before paying the cost of robot data collection.

## Gotchas & tricks

- **Filtering is the load-bearing step.** Raw Ego4D-style data without quality filtering underperforms teleop; with the pipeline it overperforms. The "egocentric > teleop" claim is not unconditional.
- **OOD wins come from diversity.** Egocentric data spans many environments / objects / lighting conditions; teleop is collected in a few lab setups. The OOD margin is the most reliable signal.
- **Caption quality matters.** Bad captions → bad text-grounding. The cleanup step (VLM + LLM) is non-optional.
- **Camera height / angle gap.** Head-mounted cameras and robot wrist/torso cameras differ; the viewpoint filter softens this but doesn't close the gap. Robot fine-tuning still does real work.
- **No claim of egocentric-only training.** The recipe is hybrid. Pure egocentric pretraining without any robot fine-tuning fails to converge on action prediction.

## Sources

- Paper: *HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining* — Ma, Bi, Deng, Zhai, Zhang, Huang, Liang, Gong, Tu, Tang, Li, Chen, Wang, Wang, Kang, Huang, Dou, Dong, Xie, Matusik, Chua, Zhou, 2026, arXiv 2606.20521.
- Background datasets: Ego4D, Project Aria, EgoExo4D — public egocentric corpora the pipeline filters from.

# ImageWAM
*Depth — using image-editing KV cache (not video generation) as the world-action context for VLA policies.*

**TL;DR:** World Action Models usually pair a policy with a *video* world model that decodes future frames. ImageWAM swaps that for a pretrained **image-editing** model whose **KV cache** is consumed directly as a compact world-action context — no future frame is rendered at inference. Achieves comparable or better task performance than video-based WAMs while cutting FLOPs to **1/6** and latency to **1/4**. Authors at SJTU + Tencent Robotics X + Tsinghua, arXiv 2606.19531.

**Prereqs:** [README](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

WAMs (World Action Models) close the loop between visual world modeling and robot control: predict the world's future, then derive the action. ImageWAM observes that the *prior* an image-editing model provides — "given current frame and a task instruction, produce the target frame" — is a better fit than the prior video generation provides ("predict an entire future trajectory of frames"). Image editing only needs to model the target-frame transformation, focuses capacity on action-relevant changes, and ties task instructions to localized edits.

Crucially, ImageWAM does **not** decode the target frame at inference: it consumes the image-editing model's internal **KV cache** from the denoising pass as the conditioning input for a flow-matching action expert.

## How it works

At inference time:

1. **Current frame + task instruction** → fed into the pretrained image-editing model.
2. **Denoising pass runs** but stops short of decoding the final frame. The KV cache produced during this pass is the model's compressed "what the world should look like next given this task" representation.
3. **Flow-matching action expert** consumes the KV cache as its conditioning context and outputs a continuous action trajectory.

The training recipe is correspondingly simple: take an off-the-shelf image-editing backbone, attach the action expert, and supervise on robot trajectory data. No video model to pretrain, no future-frame rollouts to manage.

## Why it matters

- **Right substrate, right prior.** Image-editing models already encode "what task instructions look like as targeted visual changes" — directly useful for action prediction. Video generation spends capacity on temporal/appearance details the policy doesn't need.
- **Massive efficiency win.** 6× fewer FLOPs and 4× lower latency vs video-based WAMs without quality loss.
- **No long-horizon error accumulation.** Video-based WAMs imagine a full future and can compound errors that mislead actions; ImageWAM's one-shot target-frame prior bypasses the rollout entirely.
- **KV-cache-as-context** is reusable beyond robotics — anywhere a pretrained generative model's internal state is a richer signal than its output.

## Gotchas & tricks

- **Image-editing backbone choice matters.** The editing model needs to handle instruction grounding well — generic inpainting backbones underperform editing models trained on instruction-image-target triplets.
- **KV-cache shape coupling.** The action expert's input dimension is tied to the editing backbone's KV layout; swapping backbones requires retraining the expert.
- **Cannot model long-horizon multi-step plans natively.** ImageWAM models a single target frame; complex multi-stage tasks need an explicit planner or hierarchical extension.
- **Attention analysis is a useful sanity check.** The paper shows the editing caches focus on task-relevant change regions; if your attention maps don't, the editing prior isn't doing what it should.

## Sources

- Paper: *ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?* — authors not fully listed on HF page, Shanghai Jiao Tong University + Eastern Institute of Technology + Tencent Robotics X + Tsinghua + Zhongguancun Academy, 2026, arXiv 2606.19531.

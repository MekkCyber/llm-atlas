# Video MLLMs

*Taxonomy — multimodal large language models that consume video as input.*

**TL;DR:** Video understanding in MLLMs is best decomposed into three orthogonal functions — *watching* (perception), *remembering* (memory state), and *reasoning* (trace + output) — each of which has its own set of techniques and tradeoffs. The dominant single-best stack changes per task type (short-clip / long-video / streaming / ego-centric), but the functional split lets methods be compared on what they actually upgrade rather than which benchmark they topped.

**Related taxonomies:** [_moe](../architectures/_moe.md)
**Depth files covered here:** (none yet — populated as depth files land)

---

## The problem

Video sits at the awkward intersection of long-context (thousands of frames), multimodal alignment (frames + audio + text), and limited compute. Naively feeding frames into an image MLLM either drops most temporal information (sparse sampling) or melts the GPU (dense sampling). Worse, evaluation has historically been balkanised — "video QA", "long-form QA", "streaming video", "instructional video" — making it hard to tell which architectural choice solves which axis.

## The shared pattern

All video-MLLM systems can be typed by four roles:

```
input video ─► [perception] ─► [memory] ─► [reasoning] ─► prediction
```

- **Perception** turns frames (and audio) into tokens. Cheap variants: sparse sampling + image encoder. Expensive: dense audio-visual encoding with cross-modal alignment.
- **Memory** holds past perception state. Offline = all-tokens-in-context; streaming = recurrent / sliding-window state; hybrid = compressed long-term + verbatim short-term.
- **Reasoning** is the LLM's traversal of memory to produce an answer. Text-only reasoning vs. "thinking with video" (interleaved generation + perception).
- **Prediction** is the final output — answer, caption, action, etc.

Comparing methods along the role they upgrade (not which benchmark they topped) is more informative.

## Variants

| Function upgrade | Example technique class | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Perception — fine-grained | dense frame sampling + multi-resolution encoders | compute, memory | tasks requiring small visual details |
| Perception — audio-visual | shared audio-visual tokenisers | training data scarcity | instructional, lecture, dialogue |
| Memory — offline | long-context Transformer | quadratic cost | full-video QA on short videos |
| Memory — streaming | recurrent state / sliding window | drift, lossiness | online surveillance, agent-style |
| Memory — hybrid | compressed key-frames + recent verbatim | engineering complexity | long videos with bursty relevance |
| Reasoning — text-only | standard CoT over a single perception pass | weak when evidence is sparse | short-clip QA |
| Reasoning — with video | interleaved generation + targeted re-look | latency, system complexity | long-horizon evidence gathering |

## How to choose

- **Short clips, single answer:** dense perception + text-only reasoning is the modern default.
- **Long videos, sparse evidence:** hybrid memory (key-frame compression + verbatim recent) plus interleaved reasoning.
- **Streaming/online:** recurrent memory or sliding-window attention; accept some drift.
- **Audio-heavy (lectures, dialogue):** invest in audio-visual perception; otherwise audio is wasted.

## Adjacent but distinct

- **Image MLLMs** — same backbone family but no memory axis worth typing.
- **Video generation** — flow-matching / diffusion DiT models; different problem (synthesis, not understanding).

## Sources

- Survey: *Watch, Remember, Reason: Human-View Video Understanding with MLLMs* — Meng, Tan, Xu, Gao et al. — 2026 — [arXiv:2606.07433](https://arxiv.org/abs/2606.07433)
- Tracker: github.com/marinero4972/Awesome-HumanView-VideoUnderstanding

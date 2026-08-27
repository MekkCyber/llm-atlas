# LAION-BVD
*Depth — one specific dataset / curation pipeline, grounded in its source paper.*

**TL;DR:** **LAION-BVD** is a 10-million-hour open video dataset for multimodal pretraining, harvested from CommonCrawl at web scale: 1.3B platform-specific video URLs → 80M downloaded videos → clips extracted via content-aware scene detection → synthetic video *and* audio captions generated per clip. Released fully open. Comparable in scale-vs-openness to what LAION-5B was for the image-text era, an order of magnitude bigger than prior open video datasets. Released by Hochlehnert, Nezhurina, Cherti, et al. (LAION / Tübingen), 2026.

**Prereqs:** [_data-curation.md](_data-curation.md)
**Related:** [quality-filtering.md](quality-filtering.md), [deduplication.md](deduplication.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Open video pretraining has been bottlenecked by dataset scale. Prior open corpora max out around 1M–10M hours only when combining many sources with restrictive licensing, and platform-specific APIs (YouTube-Data, Twitter) come with rate limits, TOS constraints, and unpredictable availability. LAION-BVD reroutes around the platforms: harvest video URLs from **CommonCrawl** — the same open source that made LAION-5B possible for images — and download the underlying videos.

Result: a 10-million-hour corpus (80M videos, ~7-minute average) with matched synthetic video and audio captions.

## How it works

### The pipeline

1. **CommonCrawl URL harvesting** — scan CommonCrawl WARC files for links to video files (mp4, webm) and platform-embedded video URLs (YouTube, Vimeo). Deduplicate at URL level. Yields **1.3B candidate URLs**.
2. **Download filter** — attempt download for a large subset (respecting the site's `robots.txt` and rate constraints). Successful downloads: **80M videos**, ~10M total hours.
3. **Content-aware scene detection** — split each video into clips at shot changes rather than fixed windows. Clips are semantically coherent units.
4. **Synthetic bi-modal captioning** — generate a video caption (via a strong VLM) *and* an audio caption (via an audio-LM) per clip. Both modalities' captions are stored.
5. **Scene-frame extraction** — additionally extract the middle frame of each scene as a still image with its caption, giving a large image-text auxiliary corpus with a distinct visual distribution from web-scraped images.

### Why the audio caption stream matters

Audio-language models have suffered from a data drought: most video datasets discard the audio stream or use it only for lipsync. LAION-BVD's audio captions are trainable supervision at web scale — the paper reports strong audio-text benchmark performance from models trained on this stream.

## Why it matters

- **Scale.** ~10× the open video-corpus scale prior to release. Consistent scaling laws with model and data size on standard video-text and audio-text benchmarks.
- **Audio-language pretraining.** One of few open resources large enough to pretrain audio-language models at scale. Independent of video-only interest.
- **Scene-frame image corpus.** Middle-of-scene frames have a distinctly different distribution from standard scraped web images (LAION-5B, DataComp) — text overlays, motion blur, cinematic composition. Models trained on it achieve strong image-text retrieval, suggesting it's complementary rather than redundant.
- **Reproducibility.** CommonCrawl base means any researcher can recompute a similar dataset from scratch; not gated on platform APIs.

## Gotchas & tricks

- **License heterogeneity.** CommonCrawl surfaces URLs regardless of license. Downstream users must respect the underlying content's license, which is often unclear at the individual-video level. The paper documents this and recommends per-project filtering.
- **Synthetic captions are noisy.** VLM- and audio-LM-generated captions are useful supervision but not ground truth. Downstream models trained on them inherit caption-model biases; consider caption-diversity augmentation.
- **Scene-detection tuning.** The threshold that decides "scene change" trades off clip count vs. semantic coherence. LAION-BVD's chosen threshold works for general video; specialized domains (sports, security cameras) may want different settings.
- **Deduplication is URL-only, not content-only.** Two URLs may serve the same underlying video (mirrors, re-uploads). Downstream users may want an additional content-hash dedup pass. See [deduplication.md](deduplication.md) for canonical patterns.
- **Audio caption quality varies by modality.** Speech-heavy audio captions can degenerate into transcript-lite; music-heavy audio captions are richer. Segment by audio type before training if you care.

## Sources

- Paper: *LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training* — Hochlehnert, Nezhurina, Cherti, Radonjic, Wiedemer, Schuhmann, Beaumont, Brendel, Schölkopf, Koepke, Jitsev, Bethge, 2026. [arXiv:2608.24845](https://arxiv.org/abs/2608.24845).
- Related: *LAION-5B* — the image-text era precursor; same CommonCrawl-based methodology, image modality.

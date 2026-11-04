# Omnimodal Embeddings

*Taxonomy — multi-modality embedding models, factored into shared capabilities and shared method families.*

**TL;DR:** Multimodal embedding research has been carved up by modality (text, image, audio, video), but the field is converging on a shared design question: produce a single embedding space that a user can query across modalities. The space has three capabilities to aim for — **coverage**, **comparability**, **compositionality** — and three method families that target them — **transformation**, **alignment**, **joint embedding**. Modern omnimodal embedders mix multiple families; pure contrastive joint-embedding dominates for comparability and is making the most progress on compositionality.

**Related taxonomies:** none yet.
**Depth files covered here:** none yet — this is a new taxonomy; depth files land as individual methods get documented.

---

## The problem

A single application often needs to retrieve across modalities: find the image matching a text description, the video matching a question, the audio snippet matching a mood. Per-modality embedders (BERT for text, CLIP's vision tower for images, CLAP for audio) each live in incompatible spaces. Stitching them together at query time with modality-specific routers is brittle and loses cross-modal reasoning.

**Three things go wrong** when you try to unify naively:
- **Coverage gap** — the embedder has never seen one modality (e.g. audio), so vectors in that modality are meaningless or out-of-distribution.
- **Comparability gap** — embeddings exist for every modality but their inner products across modalities are not calibrated. Cosine similarity between a text and image vector doesn't mean the same thing as between two text vectors.
- **Compositionality gap** — the embedder can retrieve "red car" and "beach" individually but not "red car on a beach" as a compound retrieval. Vector arithmetic across modalities doesn't hold.

## The shared pattern

Every omnimodal embedder is a combination of a **capability it targets** and a **method family** that produces it.

```
Capability  ─────────  Method family it is usually produced by
──────────             ─────────────────────────────────────
Coverage    ─────────  Transformation (project each modality
                       into a shared space via per-modality
                       encoders feeding a common head)
Comparability ─────────  Alignment (contrastive or matching
                       losses that calibrate cross-modal
                       similarities)
Compositionality  ────  Joint embedding (shared tower trained
                       on cross-modal composed queries; late
                       fusion)
```

A modern omnimodal stack combines all three.

## Variants

| Family | Key idea | Typical approach | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Transformation | Each modality has its own encoder; project outputs into a shared space | per-modality encoders + shared projection | gains coverage; cross-modal comparability not guaranteed | expanding to a new modality |
| Alignment | Train with cross-modal contrastive / matching losses | CLIP-style contrastive on paired data | cross-modal similarities calibrated; heavy data demand | retrieval across modalities with abundant pairs |
| Joint embedding | One shared backbone takes multimodal input and emits one vector | late-fusion transformer over interleaved tokens | best compositionality; most expensive | multimodal reasoning, agent memory |

## How to choose

**Just want a cross-modal retriever on paired text+image data?** Alignment (CLIP-style contrastive) is the strongest baseline and solves comparability directly.

**Adding a new modality to an existing stack?** Transformation (keep the shared head frozen; train a new encoder to project into it) is the lightest-touch path. You buy coverage without re-aligning the existing modalities.

**Need to retrieve by compositional query?** Joint-embedding (one shared backbone over interleaved multimodal tokens) is the only family that reliably composes. More expensive but the gap on compositional queries is large.

**Practically:** the 2026 default is a joint-embedding backbone with contrastive-alignment auxiliaries, and transformation adapters to add modalities incrementally. The three families are composable, not alternatives.

## Adjacent but distinct

- **Multimodal LLMs** (VLM / audio-LMs) — generate conditioned on multimodal inputs. Embeddings are a byproduct; retrieval is not the goal.
- **Modality-specific SOTAs** (CLIP for text-image, CLAP for text-audio, ImageBind for binding modalities to vision) — each is a special case of this taxonomy's alignment family.
- **Retrieval-augmented generation (RAG)** — uses embeddings but doesn't train them. Belongs in agents / retrieval, not here.

## Sources

- Paper: *From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings* — Zhang et al., Ant Group / SJTU / ANU / Tongji / NUS, 2026 — [arXiv:2611.00172](https://arxiv.org/abs/2611.00172).
- Paper: *Learning Transferable Visual Models From Natural Language Supervision (CLIP)* — Radford et al., 2021 — canonical alignment baseline.
- Paper: *ImageBind: One Embedding Space To Bind Them All* — Girdhar et al., 2023 — transformation + alignment to a common vision anchor.

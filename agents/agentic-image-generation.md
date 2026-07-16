# Agentic Image Generation
*Depth — search-augmented text-to-image where a reasoner decides when to fetch external knowledge before the generator draws.*

**TL;DR:** Text-to-image models confidently hallucinate anything they don't know about — recent products, obscure people, current events. Agentic image generation wraps the generator with a reasoning + tool-calling loop: a **reasoner** analyzes the prompt, decides whether the generator's parametric knowledge is enough, calls a search tool if not, and passes retrieved context to the generator. Introduced by SearchGen (2026) as a co-trained pipeline; the naïve "just prepend search results" baseline actively hurts.

**Prereqs:** *(none)*
**Related:** [../evaluation/searchgen-bench](../evaluation/searchgen-bench.md)

---

## What it is

An architecture pattern for text-to-image generation that mirrors retrieval-augmented text LLMs. Three components:

1. **Reasoner** — an MLLM that reads the prompt, identifies knowledge-intensive spans (entities, dates, factual claims), and emits either a search query or a "no search needed" signal.
2. **Search tool** — a retrieval index (web, image database, structured KB) returning textual + visual context.
3. **Generator** — a diffusion (or similar) T2I model that consumes prompt + retrieved context and produces the final image.

The point of the split: the generator's parameters absorb **stable knowledge** (things unlikely to change), while **contextual knowledge** (things that change with the world) lives in the retrieval index and gets injected per prompt.

## How it works

The SearchGen paper co-trains the reasoner and generator so they specialize:

- The reasoner is trained to gate search calls — it learns *when* the generator alone will produce a factually correct image and *when* it won't.
- The generator is trained to consume retrieved context robustly — it must ignore noisy retrieval and use the useful bits.

Naïve baselines fail in specific ways:

- **Retrieval-always** injects noise into every prompt, hurting overall quality.
- **Retrieval-never** is where frontier T2I sits today — 21–28/100 on knowledge-intensive prompts on the paper's benchmark.
- **Retrieval + fixed generator (no co-training)** helps a little, but the generator over-trusts retrieved images and reproduces artifacts.

Co-training solves both: gated retrieval + a generator that treats retrieved context as a soft prior.

## Why it matters

- **Fixes a hard failure mode.** Text-LLMs got RAG. Image models still confidently render 2026 flagship phones as generic slabs because "no knowledge cutoff" isn't a thing for T2I. Agentic gen is the direct analog fix.
- **Splits training compute along the right axis.** Facts change faster than aesthetics. Putting facts in the index and aesthetics in the weights matches how you actually want to update the system — swap the index, keep the generator.
- **Template for other modalities.** Same pattern generalizes to text-to-video, text-to-3D, text-to-audio.

## Gotchas & tricks

- **Retrieval quality dominates.** A noisy search tool destroys the co-training signal — the reasoner learns to *never* search because the noise is worse than parametric hallucination.
- **The gate is the hard part.** Reasoners over-trigger (call search on prompts the generator already handles) or under-trigger (skip search on entities the generator has never seen). Training-signal design for the gate is where the paper spends most of its effort.
- **Visual retrieval vs. textual retrieval.** Injecting a retrieved *image* changes generation dynamics more than injecting a text caption. The right modality of retrieved context depends on the failure the reasoner is trying to fix.
- **Evaluation needs a knowledge-intensive benchmark.** Standard T2I metrics (FID, CLIPScore) don't measure factual correctness. See [searchgen-bench](../evaluation/searchgen-bench.md).

## Sources

- Paper: *Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation* — Wang et al., HKUST / Qwen / Waterloo, 2026 — arXiv 2607.05382.
- Benchmark: SearchGen-Bench (20,839 knowledge-intensive prompts, 12 failure categories).

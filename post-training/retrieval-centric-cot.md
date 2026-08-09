# Retrieval-Centric Chain-of-Thought (RC-CoT)

*Depth — CoT conditioned on the initial retrieval result (and its errors), not on the query alone; trained with retrieval-oriented RL.*

**TL;DR:** Standard "CoT for retrieval" rewrites the query using a CoT that considers only the query itself. Retrieval-Centric CoT (RC-CoT, introduced by UniME-R1, 2026) instead conditions the CoT on the *top-k retrieval result* — what the retriever surfaced, and where it likely misunderstood — then either reranks (if the target is in the top-k) or emits a refined query for a full-corpus re-retrieval. The CoT-writing "adviser" is trained with SFT + retrieval-oriented RL so the CoT is scored on whether it actually improves retrieval hit rate.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [_rewards.md](_rewards.md)

---

## What it is

An embedder + adviser pair for multimodal / general retrieval:

- **Embedder.** A dual-mode retriever that can score candidates either from raw query embeddings or from CoT-augmented query embeddings.
- **Adviser.** A generative LLM that reads the top-k candidates the embedder surfaced and produces one of two outputs:
  1. **Rerank verdict** — the target is in the top-k; adviser reranks and returns.
  2. **RC-CoT** — the target is *not* in the top-k; adviser writes a chain-of-thought that identifies the discriminative cue the embedder missed, then the embedder re-runs full-corpus retrieval augmented with the CoT.

The two branches are chosen inside training and inference by a lightweight target-presence check on the top-k.

---

## How it works

1. **First-pass retrieval.** Embedder scores the corpus; top-k candidates go to the adviser.
2. **Presence check.** Determine whether the true target is in the top-k (during training) or use a confidence proxy (during inference).
3. **Branch a — rerank.** Adviser ranks the top-k candidates directly against the query and returns the argmax.
4. **Branch b — RC-CoT.** Adviser writes a CoT that names the specific confused feature (e.g., "the query asks for the object's *color*, top-k differs on *shape*"); the embedder re-embeds the query concatenated with the CoT and re-runs corpus-wide retrieval.
5. **Training.**
   - Hard-negative mining: seed the adviser with realistic near-miss retrieval failures.
   - Joint SFT on direct retrieval and RC-CoT-augmented retrieval.
   - Retrieval-oriented RL: reward = retrieval hit-rate improvement over the direct retrieval baseline; algorithm is a GRPO-style policy update on the adviser.

## Why it matters

- **CoT-for-retrieval finally has an aligned training signal.** Prior query-rewriting CoTs are trained on likelihood or SFT; RL against actual retrieval metrics closes the loop between "the CoT reads well" and "the CoT retrieves better."
- **Conditioning on failure is a general move.** The same shift — "condition on what the earlier component got wrong, not on the raw input" — applies beyond retrieval (tool-use, code generation, agent planning).
- **Multimodal-safe.** Because the adviser is an LVLM, the mechanism works for image-plus-text queries with no additional machinery.

## Gotchas & tricks

- **Presence proxy at inference.** During training the target is known; at inference you need a confidence signal for "is the target likely in top-k?" — thresholding embedder score gaps is the usual choice.
- **Hard-negative mining is load-bearing.** Without it, RC-CoT rarely fires (the target is nearly always in top-k in easy data) and the adviser overfits to reranking.
- **Full-corpus re-retrieval is expensive.** RC-CoT should fire only when the presence proxy is low-confidence; otherwise the reranking branch is much cheaper.
- **Reward hacking on retrieval metric.** Retrieval hit-rate can be gamed by generic CoTs that spam discriminative keywords. Keep a KL to a reference adviser to control this.

## Sources

- Paper: *Learning from Failures: Retrieval-Centric CoT via Hard Negatives for Unified Multimodal Retrieval* (UniME-R1) — Sun et al., DeepGlint, 2026 — [arXiv:2608.06060](https://arxiv.org/abs/2608.06060). Evaluated on MMEB-V2 and a broad set of multimodal retrieval benchmarks.

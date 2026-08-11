# Offline Top-K Distillation
*Depth — cache the teacher's top-K logits once and train the student against the cache.*

**TL;DR:** Standard "online" knowledge distillation holds a large teacher and a smaller student in memory simultaneously — the teacher runs a forward pass on every student training step. Offline Top-K distillation precomputes the teacher's top-K logits per training example once and stores them; the student trains against the cache with no teacher in memory. Reported: matches online KD's loss with ~29% faster iterations and ~41% higher throughput on a single H200.

**Prereqs:** [README.md](README.md), [../_post-training.md](../_post-training.md)
**Related:** [chunked-kl-loss.md](chunked-kl-loss.md), [_distillation.md](_distillation.md)

---

## What it is

In classical KD, the training-time loss is the KL divergence between the student's output distribution and the teacher's output distribution over the vocabulary. To evaluate it, the teacher must produce logits for every training example — an expensive forward pass on a model much larger than the student.

Offline Top-K observes: the KL over the full vocabulary is dominated by a small number of top-mass tokens. Cache only those top-K tokens' logits per example, once, then train the student against the cache. Teacher runs at cache-build time; not at training time.

## How it works

**Cache build:**
1. Run the teacher over the training corpus once.
2. At every position, keep the top-K teacher logits and their token IDs (K on the order of 32–128).
3. Store the cache to disk. Size = $|\text{corpus}| \times K \times (\text{logit} + \text{id})$.

**Student training:**
```
for batch in dataloader:
    student_logits = student(batch.inputs)     # forward
    teacher_topK = batch.cached_teacher_topK   # loaded from cache
    loss = KL(student_logits, teacher_topK)    # over K tokens only
    loss.backward()
```

The KL is computed only over the top-K positions; the remaining vocabulary tail is approximated (either zero-mass or a uniform tail). Because top-K captures the bulk of the teacher's mass, this approximation is tight — the paper shows near-identical final training loss vs online KD.

## Why it matters

- **~29% faster per iteration; ~41% higher throughput** on a single H200 GPU vs online KD.
- **Teacher is out of memory during training.** Frees ~teacher-sized VRAM budget for larger batches / longer context.
- **Cache is a one-time cost.** Amortizes across every distillation run over the same data — critical when you run hundreds of ablations.
- **Composes with [chunked-kl-loss](chunked-kl-loss.md).** Chunked KL over only top-K positions is even cheaper.

## Gotchas & tricks

- **K matters.** Too small (K=8) and the top-K tail is too approximate; too large (K=1024) and disk / IO becomes the bottleneck. K ~ 32–128 is the sweet spot.
- **Tail approximation choice.** Zero-mass tail is simplest; uniform-mass tail is slightly better; explicit "temperature-smoothed tail" is fanciest. Difference is small for reasonable K.
- **Cache invalidation.** Any change to the teacher, tokenizer, or corpus voids the cache. Version it.
- **IO can bottleneck at scale.** Cache reads become the critical path if the cache doesn't fit in SSD / RAM. Shard, compress (16-bit logits), or overlap with compute.
- **Doesn't help online curriculum.** If the student's training data is dynamically selected (curriculum, rejection sampling), you can't precompute cache for it.

## Sources

- Paper: *Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss* — Ryskulov, García-Ferrero, Montero et al., Multiverse Computing, 2026 — arXiv:2608.03796.
- Code: https://github.com/CompactifAI/Full-Chunked-KL-Loss.

# Speech-LLM Audio-Encoder Compression
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compress the audio encoder inside a speech-LLM by **progressively pruning encoder layers** and repairing the damage via behavioral probes, cross-scale distillation, and a final LoRA adaptation on the LLM — instead of retraining the encoder from scratch. Yields ~20% fewer params at competitive error rates on Chinese/English speech benchmarks.

**Prereqs:** [README](README.md), [../post-training/fine-tuning/README](../post-training/fine-tuning/README.md)
**Related:** [README](README.md), [../inference/README](../inference/README.md)

---

## What it is

Speech LLMs stack an audio encoder (Whisper-style or Conformer) in front of a text LLM. The encoder is often the fattest tower at inference — depth is what pushes decode latency. Dropping encoder layers is tempting, but naively pruning blocks perturbs embeddings enough to break the downstream LLM's expectations.

X-AuT compresses that encoder in a **progressive, probe-guided** loop: pick which layers to drop using behavioral evidence, then realign the pruned encoder's representations to the original before letting the LLM re-adapt.

## How it works

1. **Behavioral probes.** Remove each encoder layer independently and measure task-level accuracy delta on a small probe suite. Rank layers by "least damaging when removed".
2. **Progressive pruning.** Drop the top-ranked layers in stages (e.g. 18 → 16 → 14), retraining briefly between stages so subsequent probes see the compressed state.
3. **Cross-scale distillation.** Align the pruned encoder's hidden states to those of the full-depth teacher: at each retained layer, minimize a distance between student features and the teacher's corresponding features (possibly at a different depth — hence "cross-scale").
4. **Representation alignment.** A learned projection maps student to teacher space so the LLM downstream sees an interface that looks like the original encoder.
5. **LoRA adaptation on the LLM.** Final LoRA pass on the LLM absorbs any residual mismatch — the LLM has to accept slightly different feature statistics from a shallower encoder.

## Why it matters

Speech-LLM decode latency is dominated by encoder depth for many workloads. A layer-drop + repair recipe means teams can trim encoders **per deployment** without touching the massive pretraining run — the same trick applied to text-encoder trimming or vision-encoder pruning.

## Gotchas & tricks

- **Behavioral probes ≫ magnitude / gradient scores.** Removing a layer with tiny weights can still break the model; task-level accuracy delta is the honest signal.
- **Progressive, not one-shot.** Pruning 4 layers at once collapses accuracy; the paper's staged 18→16→14 preserves it.
- **Cross-scale is the point.** Aligning layer $i$ of student to layer $i$ of teacher fails when depths differ; letting the alignment cross depths (student layer $i$ ↔ teacher layer $j$) is what makes the pruned encoder recoverable.
- **LoRA on the LLM, not the encoder.** Trying to add LoRA to the pruned encoder over-fits the alignment loss; keep the encoder frozen after distillation and let the LLM adapt.
- **Language coverage matters.** The 10-benchmark Chinese-English suite catches failure modes that single-language probes miss.

## Sources

- Paper: *X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation* — Zhang, Zou, Chen, et al. (XPENG AI), 2026 — https://arxiv.org/abs/2609.11412

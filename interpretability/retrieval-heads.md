# Retrieval Heads in Omni-Modal LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Retrieval heads — the sparse set of attention heads responsible for long-context "needle-in-haystack" behavior — were previously identified in text-only LLMs. This paper extends the characterization to omni-modal MLLMs via paired audio-visual probes (same answer, two modalities). Across 18 checkpoints the retrieval heads stay sparse (<2% of heads), are largely *shared* across modalities, and are inherited from the backbone when a model is extended. Ablation collapses retrieval; a ranking-loss specialization of just these heads lifts Qwen3-VL-8B's page Recall@1 on MMDocIR by +6.03.

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A "retrieval head" is an attention head whose attention pattern copies information from a specific position in the long context to the generation position — the mechanistic substrate of needle-in-haystack behavior. In text LLMs (Wu et al. 2024) these were found to be sparse and consistent across tasks. This paper asks the same question of omni-modal MLLMs that handle text, image, audio, and video jointly.

## How it works

**Probe design.** The authors build paired audio-visual inputs where the answer is embedded either as spoken audio or as on-screen text, holding the question constant. The task is to retrieve the answer. Any head that reliably attends from the generation position to the answer's token position — in both audio and visual conditions — is a candidate retrieval head.

**Identification.** For each head $h$, measure its attention mass on the answer token position across a batch of probes. Rank heads; take the top-$k$ as the retrieval-head set. Causally verify by zero-ablating the ranked set and checking collapse of retrieval accuracy.

**Findings.** (a) Retrieval heads are <2% of total heads. (b) The sets identified from audio probes and from visual probes overlap heavily — they are modality-shared. (c) Across 18 omni-modal checkpoints the retrieval-head set is almost unchanged from the text-only or VLM backbone the model was extended from, i.e. inherited, not relearned. (d) Zero-ablating this small shared subset also degrades document QA, spatial perception, and visual math — these capabilities route through the same heads.

**Specialization.** Train *only* the retrieval heads with a ranking-contrastive objective on a page-retrieval task (positives: correct page; negatives: distractor pages). Qwen3-VL-8B page Recall@1 on MMDocIR goes up +6.03. Document-retrieval gains across three model families.

## Why it matters

Clean mechanistic result for omni-modal models: capability transfer across modalities routes through the *same* heads. That makes retrieval heads both a diagnostic (what the model inherited) and a cheap intervention point. Training 2% of heads to shift retrieval behavior is roughly LoRA-cheap but mechanistically targeted. Updates the retrieval-heads literature for the multimodal era.

## Gotchas & tricks

- **Rank by attention mass, not by activation magnitude.** The signal is "where attention goes," not "how strong the output is."
- **Zero-ablate, don't mean-ablate.** The paper shows mean ablation underestimates how specialized these heads are.
- **The shared subset is the right intervention target.** Modality-specific retrieval heads exist but are smaller and less causally important.
- **Specialization uses a ranking loss on the retrieval subtask only.** General fine-tuning on retrieval data doesn't match the gain.
- **Not the same as "attention heads that attend to BOS."** Those are induction/suppression heads; retrieval heads attend to task-relevant positions in context.

## Sources

- Paper: *A Broader Look into MLLM's Retrieval Heads* — Li, Li, Lin, Yu et al. — IIE-CAS · Tencent · UCAS, 2026 — [arXiv:2610.32411](https://arxiv.org/abs/2610.32411).
- Reference: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — the text-only predecessor.
- Reference: *MMDocIR* — the document-retrieval benchmark they lift.

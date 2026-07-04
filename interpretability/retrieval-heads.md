# Retrieval Heads
*Depth — the attention heads that carry answer content from context to output, and how to find the ones that do it *without* copying.*

**TL;DR:** In a long-context prompt, some attention heads specialize in *pulling the answer out of the context* — they read the needle span and write it into the residual stream at the output position. Prior detectors reward heads whose *attended token* matches the *generated token*, which only catches literal copying. **LOCOS** (Logit-Contribution Scoring) instead scores each head by the projection of its OV-circuit output onto the answer-token unembedding direction, so it also finds heads that *paraphrase* the retrieved span. Ablating the top-50 LOCOS heads on Qwen3-8B drops NoLiMa ROUGE-L from 0.401 to 0.000 while parametric recall and arithmetic stay at baseline.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../fundamentals/dca.md](../fundamentals/dca.md)

---

## What it is

A **retrieval head** is an attention head whose function is to take content from a specific context span and route it through its OV-circuit into the residual stream at the answer position. In long-context QA — needle-in-a-haystack, multi-hop QA, BABILong, MuSiQue — a small number of heads carry most of this work; ablating them collapses accuracy while leaving unrelated capabilities (arithmetic, parametric recall) intact.

Two flavors:
- **Literal retrieval heads** — attend to a token in the needle span, write that same token. Detectable by matching *attended-token = generated-token*.
- **Non-literal retrieval heads** — attend to a needle span but write a *semantically related* token (paraphrase, translation, deduction). Missed by attend-token matching entirely, because the answer token is *not* in the attended position.

## How it works

**LOCOS (Logit-Contribution Scoring)** replaces the literal-match criterion with a *write-aware* score. For a head $h$ at answer position, its OV-circuit output is $o_h \in \mathbb{R}^{d_{\text{model}}}$; the answer token has unembedding direction $u_a$. LOCOS scores:

$$
s_h = \langle o_h,\, u_a \rangle_{\text{needle}} - \langle o_h,\, u_a \rangle_{\text{off-needle}}
$$

The score contrasts the head's logit contribution when the needle is present vs. shifted off-position, computed in a single forward pass. Heads whose OV-output *pushes the logits toward the answer token specifically because the needle is in the context* rank high — regardless of what they attend to.

Ablate the top-k heads (mean-ablation) and measure task collapse; the drop measures how much the model actually depended on those heads.

## Why it matters

- **Interprets synthesis, not just pointing.** Attend-token detectors implicitly assume the model *copies* the answer. Real long-context QA involves paraphrase and inference; LOCOS surfaces the heads doing that work.
- **Ablations are much sharper.** On Qwen3-8B, ablating 50 LOCOS heads drives NoLiMa ROUGE-L from 0.401 to **0.000** — the best attention-based baseline still retains 0.292. MuSiQue drops 0.55 → 0.08 and BABILong 0.62 → 0.20, while a random-heads control stays within 0.05.
- **Retrieval-specific, not general.** The same ablation leaves parametric recall and arithmetic untouched. The heads are doing *retrieval*, not general reasoning — a clean specificity result.

## Gotchas & tricks

- **The needle-vs-off-needle contrast is essential.** Using absolute logit contribution alone confounds retrieval heads with heads that always push toward the answer token (e.g. label priors). The contrast isolates the *context-dependent* component.
- **Mean-ablation, not zero-ablation.** Zero-ablation is out-of-distribution and can collapse general capabilities; mean-ablation preserves the head's average behavior and isolates the *variable* signal.
- **Transfers across families.** Demonstrated on Qwen3, Gemma-3, and OLMo-3.1 — but the *identity* of top heads is model-specific. Re-run the detector per model.
- **Doesn't tell you *how* the paraphrase happens.** LOCOS identifies which heads write toward the answer; understanding the OV circuit itself (which input features it transforms, how) is a separate mech-interp step.

## Sources

- Paper: *Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads* — Gema, Alex, Minervini (Edinburgh / Heriot-Watt), 2026 — [arXiv:2607.01002](https://arxiv.org/abs/2607.01002).
- Predecessor: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — the attention-based literal-copy detector LOCOS supersedes.
- Benchmark: *NoLiMa* — non-literal retrieval evaluation used to validate LOCOS.

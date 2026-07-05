# LOCOS — Logit-Contribution Scoring

*Depth — score attention heads by what they transmit into the logits, not by where they attend.*

**TL;DR:** Existing retrieval-head diagnostics look at *where* each head attends. LOCOS looks at *what* each head transmits by projecting the head's output-value contribution onto the unembedding direction of the correct answer token. Heads with a large gap between their needle-position and non-needle-position projections are the ones performing **semantic transformation** — the non-literal retrieval that paraphrase and multi-hop tasks depend on. A single forward pass surfaces the ranking.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [_retrieval-heads.md](_retrieval-heads.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A per-head diagnostic that isolates *functional* retrieval heads — heads whose output value circuit actually pushes the model toward the right answer — from heads that merely attend to the needle. It uses information the model already produces during a single forward pass on a retrieval task, so scoring the whole head set is cheap.

## How it works

### The scoring quantity

For each head $h$ at position $t$, decompose its output value contribution and project it onto the correct-answer unembedding direction $u_a$:

$$
\text{contrib}_{h}(t) = \langle O_h \cdot v_h(t),\; u_a \rangle
$$

where $v_h(t)$ is the head's value at position $t$, $O_h$ is the head's output projection into the residual stream, and $u_a$ is the unembedding row for the target answer token.

### Needle vs non-needle contrast

Split positions into **needle positions** (where the answer information actually lives in the context) and **non-needle positions**. LOCOS scores head $h$ by the gap between its typical contribution at needle vs non-needle positions. Large gap → the head is selectively transmitting the retrieval-relevant signal into the logits.

### Ablation as verification

To confirm functional relevance, mask (zero) the top-ranked LOCOS heads and re-run the retrieval task. If they were doing the work, task accuracy drops sharply; unrelated capabilities remain intact. Compare against masking heads ranked high by attention-pattern retrieval — LOCOS-ranked ablations should hurt more on *non-literal* retrieval specifically.

## Why it matters

- **Separates "attends to" from "transmits from."** Attention weights say what a head is looking at; they don't say what it does with what it sees. For copy tasks the two agree; for non-literal retrieval they don't.
- **Cheap.** One forward pass per input. No path patching, no ablation sweep for scoring (only for verification).
- **Reveals a functional class.** Non-literal retrieval heads are a real, distinct category — LOCOS's targeted-ablation curves demonstrate that hurting them doesn't hurt other tasks.

## Gotchas & tricks

- **Needle labeling is the input.** LOCOS depends on knowing which positions carry the answer information. Automating this on messy real tasks is a work item; on synthetic needle-in-haystack it's given by construction.
- **Model-family generalizations may be per-family.** The paper tests three model families; expect the *identity* of top LOCOS heads to differ across architectures but the *methodology* to transfer.
- **Not a substitute for path patching.** LOCOS gives a functional ranking; path patching gives a causal graph. Use both when the goal is a full circuit story.
- **Answer-token ambiguity hurts.** Multi-token answers require aggregating projections across tokens; the paper reports mean over the answer span, but this is a knob.

## Sources

- Paper: *Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads* — Aryo Pradipta Gema, Beatrice Alex, Pasquale Minervini (Edinburgh / Heriot-Watt / Miniml.AI), 2026 — [arXiv:2607.01002](https://arxiv.org/abs/2607.01002).
- Related: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — the attention-pattern retrieval-head baseline LOCOS contrasts against.

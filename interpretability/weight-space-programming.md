# Weight-Space Programming
*Depth — when an LLM writes a smaller model's weights by hand, it chooses a readable table, not a transformer.*

**TL;DR:** Give a frontier LLM ten labeled examples, a sandbox, and the choice of either filling in transformer weights or designing its own architecture. All three tested frontier LLMs converge on **sparse tables** — each row a labelled fact, queried by a fixed procedure. The LLM-designed table beats the LLM-written transformer by a wide margin with ~1800× fewer parameters, and the weights are human-readable.

**Prereqs:** [../architectures/transformer-block](../architectures/transformer-block.md)
**Related:** (interpretability/ is still bare — this seeds it)

---

## What it is

A study of **what architecture LLMs pick when they have to write weights by hand** (no gradient descent). The authors fix the task (multiple-choice QA with 10 labeled examples and a sandbox for validation) and vary only whether the LLM is forced to write weights into a transformer, or free to design its own structure.

This is not about gradient-descent training — it's about which representational structure is *writable by a language model from examples*.

## How it works

The sandbox:

1. The LLM sees 10 labeled MC examples and a validation set stub.
2. It must emit a program that implements an inference function.
3. **Fixed-architecture condition**: the program must instantiate a transformer of fixed shape; the LLM writes its weight tensors as literals.
4. **Free condition**: the program may define any architecture; the LLM writes it top-to-bottom.
5. Accuracy on held-out MC questions scores the program.

What emerges in the **free** condition across three different frontier LLMs:

```
# Structure every LLM converges on
table = [
  ("fact about Paris", "France", "capital"),
  ("fact about Berlin", "Germany", "capital"),
  …
]

def answer(question):
  row = find_row(question, table)   # simple keyword / embedding lookup
  return row.target_field
```

Each row is **human-readable English** plus a target field. Inference is `find_row → project`. Removing a row changes about one other answer (locality). Fixing a wrong answer edits exactly one row.

Scaling the table up by distillation gives an **associative-memory model up to 17× faster on GPU than a same-sized transformer**.

## Why it matters

- **Transformer universality has a precondition.** Transformers are the default because they train well with gradient descent, not because they are the right structure for a task a-priori. In a non-gradient setting, LLMs pick a different structure — one with the three properties they can actually reason about: *named weights*, *locality*, *editable*.
- **Interpretability is structural, not post-hoc.** The LLM-designed model's weights are literally plain-text facts. Compare to post-hoc interpretability on transformers, which spends enormous effort recovering what LLM-written tables give you for free.
- **Neurosymbolic framing.** Table-of-facts + fixed query procedure is a classic symbolic AI shape; this paper is empirical evidence that frontier LLMs *prefer* that shape when they are the optimizer.

Headline numbers: LLM-designed table models hit up to **54.4% test accuracy** on multiple-choice tasks; LLM-written transformers do far worse at similar parameter counts.

## Gotchas & tricks

- **Pretrained LLMs are still more accurate.** The claim is about which structure to *write weights into*, not which structure wins when trained by gradient descent. Pretrained Transformer > LLM-designed table > LLM-written Transformer, at roughly matched scale.
- **The free-condition result depends on the LLM's vocabulary.** All three tested frontier LLMs converged on tables, but the paper notes this is partially attributable to tables being a common Python idiom in their training data.
- **Not a training-free recipe.** You still need an LLM in the loop at inference (or for the design step). The 17× speedup is for the *resulting* table-based model, which runs standalone.
- **Scales poorly to generative tasks.** The result is cleanest on MC; free-form generation with the same structure is much harder and the paper acknowledges it.

## Sources

- Paper: *Weight-Space Programming Doesn't Need Transformers* — Hyunjong Ok, Jaeho Lee (POSTECH), 2026 — [arXiv:2611.22055](https://arxiv.org/abs/2611.22055)

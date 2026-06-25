# DREAM: Dense Retrieval via Autoregressive Modeling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a dense retriever using a frozen LLM's next-token prediction loss as the only supervision — no contrastive positives or negatives required. Retriever similarity scores are injected as weights into selected attention heads of the LLM, so the LM loss's gradient flows back through attention to the retriever. Beats contrastive baselines on BEIR and RTEB across embedding backbones from 0.5B to 3B.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [_data-curation.md](_data-curation.md)

---

## What it is

Standard dense retrievers train with InfoNCE or similar contrastive losses, which require labeled (query, positive document, hard negatives) triples. Mining good negatives is the bottleneck for high-quality retrievers. DREAM sidesteps the entire contrastive setup with a much weaker assumption: *if a document is relevant to a query, conditioning an LLM on the document should make the target output easier to predict.*

## How it works

The challenge is plumbing — the LM loss lives inside a frozen language model, the retriever is a separate encoder. DREAM bridges them through attention:

1. For a (query, target-output) training example, the retriever encodes the query and each candidate document, producing similarity scores.
2. The selected candidate documents are packed into the LLM's prompt.
3. In selected attention heads of the (frozen) LLM, the retriever's similarity scores are injected as weights determining how much each document's tokens contribute when predicting the target output.
4. The LLM autoregressively generates / scores the target with the standard next-token loss.
5. The loss back-propagates through the attention mechanism to the retriever; the LLM itself stays frozen.

The retriever learns to assign high similarity to documents whose attention contribution helps the LLM predict the target — exactly the documents that are actually relevant to the query.

## Why it matters

- Removes the labeled-triples bottleneck. Any (query, target) pair where the target depends on the query is usable training data — abundant in any QA, summarization, or instruction corpus.
- Pairs naturally with LLM training pipelines: the same data the LM trains on can train the retriever, with no extra labeling.
- Demonstrates a more general idea — using a frozen LM as a *judge* whose loss provides gradients for an external module — that should transfer to other components (rerankers, tool selectors, agent planners).

## Gotchas & tricks

- "Selected attention heads" is a hyperparameter; not all heads are equally informative. The paper specifies which heads work best; ablating differs by backbone scale.
- Requires the candidate documents to fit in the LLM's context. Long-document retrieval needs chunking or extra summarization.
- Tested on 0.5B–3B embedding models; scaling to larger embedders is open.
- The frozen LLM's quality bounds the supervision signal: a weak LM gives noisy gradients.

## Sources

- Paper: *DREAM: Dense Retrieval Embeddings via Autoregressive Modeling* — Tang & Yang, HKUST, 2026 — [arXiv:2606.24667](https://arxiv.org/abs/2606.24667).

# Search-Native Pretraining (QueryLM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Interleave **web-search queries and locally retrieved results** directly into pretraining text so that next-token prediction *includes* learning when and how to search. 360M–3.3B scale on Wikipedia + DCLM-Edu. A 1.4B QueryLM matches a FLOP-matched 3.3B baseline on knowledge-intensive tasks; the retrieval-following behavior survives standard SFT.

**Prereqs:** [mid-training](mid-training.md)
**Related:** [mtp](mtp.md) · [wsd-schedule](wsd-schedule.md)

---

## What it is

Retrieval augmentation is usually bolted on at inference (RAG) or taught at post-training. Both approaches have to overcome the base model's "answer-from-weights" prior.

Search-Native Pretraining rewrites the pretraining corpus so that **documents that would benefit from retrieval already contain inline retrieval traces**:

```
…As of 2025 the population of <query>population of Toulouse 2025</query>
<result>Toulouse: 498,000 (INSEE, 2024 estimate)</result>
Toulouse was about 498,000…
```

The model sees the query/result pattern as part of ordinary token prediction. There is no separate retrieval loss, no RL, no mid-training stage — just a corpus with retrieval woven in.

---

## How it works

**Corpus construction.** Take a pretraining corpus (Wikipedia, DCLM-Edu). For each document, identify spans that reference factual knowledge likely to live in retrievable form. Automatically generate a search query that would surface that fact, run it against a local web index (or static snapshot), and splice the top result into the document before the fact appears.

**Schema.** Special tokens delimit `<query>` and `<result>` blocks. During training, the model learns to:
1. Predict when a query would be useful (when to emit `<query>`).
2. Phrase the query.
3. Continue the document assuming the result has been returned.

**Scale.** The authors train model sizes from 360M to 3.3B parameters, each on the same mixture. The 1.4B model is the sweet spot for the comparisons they report.

**SFT survival.** Standard instruction-tuning SFT does not erase the retrieval behavior. Downstream agents can call retrieval with the same special-token format they saw in pretraining.

---

## Why it matters

Shifts the retrieval/knowledge interface from a post-hoc patch to a **native modality** of the pretraining distribution. Payoff: a 1.4B model with native search matches a 3.3B model trained on the same tokens without it. The retrieval behavior being token-prediction-native also means the model is less likely to ignore or hallucinate around retrieved content.

Complements the broader "modify the pretraining objective to inject behaviors" line of work: unlike [MTP](mtp.md) (which modifies the objective itself), this modifies the data distribution — a cheaper and more composable intervention.

---

## Gotchas & tricks

- **Index coverage matters.** A static snapshot used in pretraining can go stale; the paper uses DCLM-Edu-timed retrievals to avoid train-test leakage.
- **Query quality dictates downstream quality.** Garbage queries produce a model fluent in emitting bad queries.
- **Needs retriever at inference too.** The model has the *behavior* baked in but still needs a retriever to execute queries at inference. The authors release both.

---

## Sources

- Paper: *Search From the Start: Search-Native Pretraining for Language Models* — Ponkshe, Je, Taga, Raffel, Jaggi, 2026 — [arXiv:2611.26809](https://arxiv.org/abs/2611.26809)

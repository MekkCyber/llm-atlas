# Knowledge-Probe Hallucination Suppression

*Depth — Llama 3's factuality pipeline: teach the model to refuse on questions it confidently gets wrong.*

**TL;DR:** Frame hallucination as an alignment problem: *"post-training should align the model to know what it knows, not add knowledge"* (Llama 3 Sec. 4.3.6). Sample a factual question derived from a pretraining snippet; ask the model K times; if the model is **consistently informative but wrong**, **generate a refusal** ("I don't know" style) and teach it back. The model learns to detect its own unreliable knowledge areas and refuse rather than hallucinate. Simple, cheap, automated via Llama 3 itself as the judge.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [reward-modeling](reward-modeling.md) · [capability-experts](capability-experts.md)

---

## What it is

A specific post-training pipeline for reducing hallucination. Llama 3's own framing (Sec. 4.3.6):

> *"Our factuality training follows the principle that post-training should align the model to 'know what it knows' rather than add knowledge."*

Attributed to prior literature on factuality in LMs (Gekhman et al. 2024, Mielke et al. 2020). The insight: a model's parameters encode approximate facts; when asked a question in its knowledge domain, it answers correctly; at the edges of its knowledge, it **confidently hallucinates** instead of saying "I don't know." The fix isn't to feed it more facts (that's pretraining) — it's to teach it to recognize its own unreliability.

---

## How it works

### The pipeline (Sec. 4.3.6)

```
1. Extract snippet from pretraining data (a paragraph of factual content).
2. Generate a factual question from the snippet (Llama 3 as question-generator).
3. Sample K responses from Llama 3 for the question.
4. Score each response's CORRECTNESS: Llama 3 as judge, with the original snippet as reference.
5. Score each response's INFORMATIVENESS: Llama 3 as judge (is the response substantive vs evasive?).
6. IF the responses are consistently INFORMATIVE BUT INCORRECT:
       The model has confidently wrong knowledge about this question.
       Generate a REFUSAL ("I'm not sure about that.") as the target.
       ADD (question, refusal) to SFT data.
   ELSE IF consistently CORRECT:
       Model knows this. No action needed.
   ELSE IF consistently NOT INFORMATIVE (evasive, doesn't commit):
       Model already declines. No action needed.
```

The key case: **consistently informative but incorrect** → confidently wrong → needs a refusal to be taught.

### Why this works

- If the model **occasionally** gets it right, the inconsistency suggests it has some knowledge; teaching a refusal would discard that. Skip.
- If the model **consistently refuses** already, nothing to do.
- If the model **consistently hallucinates** with high informativeness, the failure is "confident wrongness." Teaching a refusal on exactly these prompts surgically targets the failure mode.

The scale matters: automated with Llama-3-as-judge, generate thousands of (question, refusal) pairs, mix into SFT.

### Why not just always teach refusals

Over-refusal is a known failure mode. If you teach the model to refuse on everything it might be wrong about, it refuses on easy questions too (excessive hedging, "I can't be certain..."). The knowledge-probe filter selectively targets confidently wrong responses — preserving confident correct behavior elsewhere.

### Llama 3 adds a small labeled dataset

In addition to the automated pipeline, Llama 3 includes **a small labeled factual dataset for sensitive topics** (medical, legal, political). These have handcrafted refusal / calibration responses.

---

## Why it matters

- **Targeted hallucination reduction.** Doesn't require knowing which facts the model knows (impossible to enumerate); the knowledge-probe pipeline auto-detects them.
- **Automated.** Llama 3 itself generates questions, judges correctness, and generates refusals. No human labeling.
- **Preserves confident correct behavior.** Unlike uniform refusal tuning, only the detected-weak areas get refusals.
- **Composable with other post-training.** Plugs into standard SFT. The refusal-tuned SFT data mixes with everything else.

---

## Gotchas & tricks

- **Judge-model correctness signal is itself imperfect.** If Llama 3 is the judge on a question Llama 3 doesn't know, the judge is wrong. Pretraining snippet as reference helps; still not perfect.
- **K must be large enough to detect consistency.** K = 3 gives too noisy a consistency signal; K = 8 or 10 is typical.
- **Refusal style matters.** The generated refusal should be helpful ("I'm not sure — you might want to check [source]") not adversarial ("I can't answer that"). Llama 3 uses its own instruct style for refusals.
- **Calibration vs hard refusal.** Some approaches teach the model to say "I'm 60% confident" rather than a hard refusal. Llama 3's pipeline just does refusals; calibration is a different problem.
- **Applies only to factual questions.** Doesn't help with math, reasoning, or creative tasks. Different failure modes need different fixes.
- **Dataset size undisclosed.** Llama 3 doesn't state how many (question, refusal) pairs are added. Presumably thousands, not millions.
- **Doesn't prevent all hallucination.** Just the confident-informative-wrong case. Low-confidence hallucinations still happen.
- **Interacts with RAG.** With retrieval augmentation, the model has external context — the knowledge-probe targets are less relevant. Pure-weights factuality is the use case.
- **Can create false-refusal regressions.** Teaching refusal on some questions may bleed into adjacent questions the model *does* know. Monitor false-refusal rate during tuning.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 4.3.6.
- Paper: *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* — Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig — Google Research, 2024, arXiv 2405.05904 — foundational result that fine-tuning on new knowledge can worsen hallucination.
- Paper: *Reducing conversational agents' overconfidence through linguistic calibration* — Mielke et al., 2020, arXiv 2012.14983.
- Related: Anthropic's and OpenAI's similar calibration/refusal training; specifics are less public.

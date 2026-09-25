# Dual-Track Memory (SpeakerMem-R1)
*Depth — verbatim + derived-state memory for multi-party dialogue.*

**TL;DR:** Multi-party dialogue memory is not the same problem as two-party dialogue memory: the system has to track **who said what about whom**, group-level shared knowledge, and asymmetric perceptions between speakers. Flat retrieval over concatenated history loses this structure. Dual-track memory stores each utterance twice — as **speaker-labeled verbatim text** and as **derived state** organized into person-level and group-level views — then merges the two tracks at query time by matching entity, event, and time. The state-extraction writer is trained with a speaker-conditioned GRPO variant and a **SpeakerLevenshtein** attribution-aware edit-distance reward.

**Prereqs:** [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md)
**Related:** [jit-memory](jit-memory.md)

---

## What it is

An agent-memory architecture that keeps two parallel indexes over a multi-party conversation. The verbatim track keeps raw messages with speaker labels attached; the state track keeps LLM-derived structured facts, organized into two view types (per-person, per-group). Retrieval combines both — the verbatim track supports lookup by literal phrasing and provenance, the state track supports relational queries ("what does A think about B") and cross-time aggregation.

## How it works

**Ingestion.** Each incoming utterance is stored verbatim with (speaker, timestamp) labels. A writer LLM extracts a structured update: per-person state changes (A learned X, B is upset), per-group state changes (the group agreed to Y), and event annotations tying updates to specific messages.

**Retrieval.** Given a query, the reader searches both tracks:
- Verbatim → matched messages with attribution.
- State → filtered by entity (person / group), event, or time window.

The two result sets are merged by common entity/event/time keys and passed to the answering LLM together, so verbatim evidence backs each derived-state claim.

**Writer training.** The state extractor is trained with:
- **SFT** on demonstrations.
- **Speaker-conditioned GRPO** — rollouts are grouped by speaker context; the KL / baseline are computed per speaker so the reward captures attribution correctness rather than message content only.
- **SpeakerLevenshtein reward** — an edit-distance reward that penalizes speaker/target mis-attribution more than content phrasing differences.

## Why it matters

Multi-party is where structural failures show up. Existing frameworks confuse who said something with who was talked about, or lose group-level shared knowledge because it isn't attributed to a single speaker. Explicit dual-track storage with attribution-sensitive training closes both gaps. Numbers: +3.3 / +12.4 / +9.4 pts on GroupMemBench / SocialMemBench / EverMemBench over the best prior framework, 62.33 EverMemBench SOTA, 70.85% on LoCoMo.

## Gotchas & tricks

- The verbatim track dominates storage on long conversations; state extraction is cheap in comparison. Practical implementations cap verbatim retention with a decay policy and rely on the state track as the compressed long-term memory.
- SpeakerLevenshtein is a **rule-based verifier**, not a learned RM. This makes it suitable for RLVR-style training and cheap to iterate.
- Person-level and group-level views are ablated separately in the paper — both are necessary, and their merge dominates either alone.

## Sources

- Paper: *SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue* — Zheng et al., 2026 — [arXiv:2609.26780](https://arxiv.org/abs/2609.26780)

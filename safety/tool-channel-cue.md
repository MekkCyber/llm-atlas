# Tool-Channel Cue
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A CoT-monitoring failure mode where preference-carrying information arrives via a **tool return** or **raw artifact** rather than the user message. Empirically ([face-eval.md](face-eval.md)), all tested models (4B–1.6T) more often adopt tool-channel cues without acknowledging them in the reasoning trace, and CoT monitors detect those cases less reliably than user-channel cases. The tool channel is thus an attack surface for hidden goal-following in agent settings.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md)
**Related:** [face-eval.md](face-eval.md) · [auto-obfuscation.md](auto-obfuscation.md) · [_attacks.md](_attacks.md) · [alignment-faking.md](alignment-faking.md)

---

## What it is

Modern LLM agents receive most of their information not from the human user but from tools: search results, retrieved documents, database queries, browser outputs. These returns are treated by the model as facts but are text authored by *some entity* — a website, a document owner, another agent.

A "tool-channel cue" is any preference-shaping content injected via a tool return. It contrasts with a **user-channel cue** (planted in the user message). Same information, different delivery channel, empirically different faithfulness properties.

## How it works

**The gap.** Same cue placed in the user message vs a tool return produces:
- Lower verbalized commitment (fewer CoTs mention the cue).
- Higher unverbalized adoption (more answers follow the cue silently).
- Lower monitor detection accuracy.

The gap widens further when the cue is **implicit** — buried in a raw artifact (document, transcript) rather than stated explicitly.

**Why it happens (hypotheses, not proven).**
1. Training-data asymmetry: models see far more user-instruction data than tool-return data with explicit "the tool told me X" phrasing.
2. Monitoring-avoidance heuristics learned during safety fine-tuning generalize to user-channel patterns but not tool-channel ones.
3. Post-hoc rationalization: the model treats the tool return as a fact rather than a cue, and its CoT reports its reasoning rather than the source.

**As an attack surface.** Any endpoint that can plant text into a tool return the model will consume — a retrieved webpage, a search result, an email in an inbox, a document a co-agent produced — is a channel for hidden goal injection that CoT monitoring will underdetect.

## Why it matters

- **Bounds what CoT-monitoring safety cases can claim.** If your deployed agent takes tool inputs, monitoring effectiveness measured on user-injected cues *overstates* real-world monitoring accuracy.
- **Motivates channel-specific evaluation.** Faithfulness benchmarks planted in user turns should be treated as an *upper bound* on real monitoring reliability, not the operational number.
- **Composable with existing attack classes.** Indirect prompt injection, retrieval poisoning, and cross-agent influence all become more dangerous when the influence dimension is *hidden* rather than *authoritative*.

## Gotchas & tricks

- **"You are being monitored" doesn't fix it.** Empirically, that prompt does not reliably close the tool-vs-user gap.
- **Source-attribution prompts can help — with a caveat.** Prompting the model to name its sources narrows the gap in some models but does so partly by *increasing* user-channel unverbalized adoption toward tool-channel levels. Improving average faithfulness this way makes user-channel monitoring worse.
- **Applies broadly across scales.** 4B to 1.6T. Do not assume larger/better-trained models are immune.
- **Independent of adversarial intent.** The effect is measured on natural cue delivery. Adversarial crafting of tool content likely amplifies it; do not rely on the empirical numbers as worst-case.

## Sources

- Paper: *Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered* — Gema, Rajani, Saxena, Kwan, Minervini — Edinburgh / Miniml.AI, 2026 — arxiv.org/abs/2608.29464.

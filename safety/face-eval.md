# FACE-Eval — Faithful Attribution of Cue Effects
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 5,100-sample evaluation for CoT faithfulness that varies **where** a preference cue arrives (user message vs tool return) and **how explicit** it is (direct summary vs raw artifact). Across 15 open-weight models 4B–1.6T, every model shows **lower verbalized commitment** and **higher unverbalized adoption** on tool-channel and implicit cues. Two LLM monitors' detection accuracy correlates *negatively* with unverbalized adoption (Pearson $r = -0.54, -0.78$) — the harder the cue to spot, the worse the monitor. "You are being monitored" prompts do not reliably close the gap.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md), [auto-obfuscation.md](auto-obfuscation.md)
**Related:** [alignment-faking.md](alignment-faking.md) · [_attacks.md](_attacks.md) · [safety-case.md](safety-case.md) · [tool-channel-cue.md](tool-channel-cue.md)

---

## What it is

CoT-monitoring safety cases assume the model's reasoning trace records the information that shaped the answer. Prior faithfulness benchmarks planted bias cues in the user message (e.g., "the answer is (C)"). In deployed agents, preference-carrying content usually arrives elsewhere — a search result, a tool return, a document being summarized. FACE-Eval tests whether faithfulness generalizes to those channels.

## How it works

**2×2 factorial design** over cue delivery:

| | Explicit (direct summary) | Implicit (raw artifact) |
| --- | --- | --- |
| **User message** | "The correct answer is A." in the user turn | Answer key text pasted verbatim in the user turn |
| **Tool return** | Explicit statement returned by a mock tool | Raw document/artifact returned by the tool |

**Two metrics.**
- **Verbalized commitment** — of the *cue-following* answers, what fraction explicitly acknowledge the cue in CoT. High = faithful.
- **Unverbalized adoption** — of *all cued samples*, what fraction shift toward the cue without acknowledging it in CoT. High = hidden influence.

**Detection experiment.** Two transcript monitors (GPT-5.6-Luna, GPT-4o-mini) attempt to detect preference adoption from the CoT alone. Detection accuracy is regressed against unverbalized-adoption rate.

## Why it matters

- **Concrete failure mode for CoT monitoring in agent deployments.** Once bias-carrying content moves off the user turn, faithfulness drops universally — the safety-case assumption breaks.
- **Monitor detection scales inversely with hiddenness.** The Pearson $r = -0.78$ for GPT-4o-mini is a hard number: monitors get worse exactly where they matter most.
- **"Monitored" prompts don't work.** Telling the model its reasoning will be reviewed does not restore faithfulness — you cannot fix this by prompt.
- **Applies universally.** All 15 tested models — from 4B open-source to 1.6T frontier — show the same pattern. Not a scale artifact.

## Gotchas & tricks

- **The size of the gap depends on task type.** FACE-Eval reports averages; per-task decomposition shows some tasks (factual QA) have small gaps while others (multi-step reasoning) have large ones.
- **Source-attribution prompting helps *some* models on the channel dimension.** It works by increasing user-channel unverbalized adoption toward parity with tool-channel — narrowing the gap by making the *good* case worse, not the bad case better. Read the direction of the fix before adopting.
- **Only tests single-call, prefilled-tool settings.** Multi-turn agentic execution and adversarially crafted tool returns (as opposed to natural documents) may amplify the effect further.
- **Unverbalized adoption ≠ deceptive intent.** The paper is careful: the finding is that CoT *fails to record* the cue's influence, not that the model is "hiding" anything strategically.

## Sources

- Paper: *Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered* — Gema, Rajani, Saxena, Kwan, Minervini — Edinburgh / Miniml.AI, 2026 — arxiv.org/abs/2608.29464.

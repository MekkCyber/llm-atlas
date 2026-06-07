# ArcANE
*Depth — a benchmark for whether role-playing language agents track a character's psychological arc, not just their facts.*

**TL;DR:** Existing role-play benchmarks ask whether the model can recite facts about a character. **ArcANE** (Song et al., 2026) asks the harder question: at a specific point in a story, does the model respond *as the character at that point* — i.e. with the right beliefs, emotional state, and knowledge for that arc position? It segments 17 novels and 80 characters into psychological phases, then probes models at each phase including beyond the source material. Conditioning models on an explicit character arc beats every other context strategy.

**Prereqs:** [evaluation/README.md](./README.md), [safety/roleplay-jailbreak.md](../safety/roleplay-jailbreak.md)
**Related:** [agents/README.md](../agents/README.md)

---

## What it is

A character has a *trajectory*: at the start of a story they don't know things they later learn, don't feel things they later feel, hold positions they later abandon. A role-playing LLM agent that always answers "as the character" risks anachronism — replying with full-arc knowledge to a probe set early in the arc.

ArcANE makes that anachronism measurable. Each test scenario is tagged with:

- which novel and character;
- which *phase* in the character's arc (early, mid, late, post-canon counterfactual);
- a probe (a question or an interaction).

The model is scored on whether its response is consistent with the character *at the given phase*.

## How it works

### Construction

1. **Phase segmentation.** Each novel is automatically segmented into psychological phases per character — a phase is bounded by an event that changes the character's beliefs, goals, or relationships.
2. **Probe generation.** For each (character, phase) pair, generate probes targeting beliefs, knowledge, and behaviour distinctive to that phase.
3. **Beyond-canon probes.** Some probes describe situations the source material doesn't cover, forcing the model to *generalise* the arc rather than retrieve.

### Context strategies compared

- **No context** — just the character name.
- **Summary** — short biography of the character.
- **Full text** — the novel up to the probe phase.
- **Character arc** — an explicit structured representation of the character's trajectory.

### Metric

A per-probe judge measures consistency with the targeted phase, scoring on knowledge alignment, behavioural plausibility, and emotional fit. Aggregated across phases and characters.

## Why it matters

- **Distinguishes facts from development.** Models do well on facts about a character (who they are, what they did) and badly on the dynamics (what they knew at time $t$). ArcANE separates the two.
- **Implications for role-play safety.** Role-play is a known jailbreak vector ([roleplay-jailbreak](../safety/roleplay-jailbreak.md)); legitimate role-play capability has to be measured to be defended without over-refusing.
- **Generalises to persona agents.** Customer-service personas, NPCs, and tutor agents all need arc-aware behaviour; ArcANE's setup transfers.

## Gotchas & tricks

- **Phase boundaries are partly subjective.** Automatic segmentation works but borderline phases are noisy; the paper handles this with majority-judge agreement.
- **Beyond-canon probes are the hardest part.** They require the model to extrapolate arc dynamics, not just retrieve from the novel — this is where small models collapse first.
- **Arc-conditioning is the winning strategy.** Both better than no context and better than the full novel text. The structured arc is denser and more on-topic than full text, and richer than a summary.
- **Doesn't measure tone / voice.** Two characters can have the same beliefs at the same phase but different voices. ArcANE is about *what* the character would say, not *how*.

## Sources

- Paper: *ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?* — Song, Kim, Song, Heo, Lim, Jo (Seoul National University), 2026 — [arXiv:2606.05553](https://arxiv.org/abs/2606.05553).

# Skill Libraries for Software Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **skill library** is a persistent, indexed store of reusable executable action recipes an agent can retrieve at run time. Instead of relying only on the base model's parametric knowledge or expensive trial-and-error, the agent selects from a library of pre-authored skills and composes them. Recent work (Resource2Skills, 2026) shows skills can be **distilled from ordinary human-authored resources** — tutorials, docs, wikis, videos — rather than hand-authored or expert-demonstrated.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [graphrag.md](graphrag.md), [harness-self-improvement.md](harness-self-improvement.md)

---

## What it is

Three sources for agent skills, from most to least expensive:

| Source | Cost | Coverage |
| --- | --- | --- |
| Expert human demonstrations | Very high | Narrow, high quality |
| Trial-and-error rollouts + filter | High compute | Broader, quality bounded by verifier |
| **Distillation from human-authored resources** | Cheap | Very broad, quality bounded by extractor |

Resource-mined skill libraries are the cheapest path with the widest coverage. The library sits between the base model (parametric skills) and the harness (agent-loop scaffold): retrieval-time it provides the *what to do*; the harness provides the *when and how*.

## How it works

A resource-to-skill pipeline has five moving parts:

1. **Source mix.** Which resources you ingest — tutorials, official docs, videos, wikis, forum threads — sets the ceiling on coverage.
2. **Multimodal extractor.** Parses the resource (text + images + video frames) and extracts parameterized skill recipes. Slides → "make-title-slide(title, subtitle)"; UE5 tutorial → "spawn-light(position, color, intensity)".
3. **Wiki organization.** Skills are stored in a wiki-like structure with cross-links; agents retrieve by concept + topical proximity, not just embedding similarity.
4. **Selection strategy at runtime.** Given a task, the agent embeds it, retrieves a small set of candidate skills, and (optionally) rewrites/composes them.
5. **Online acquisition.** If no library skill fits, spawn a resource-extraction pass on the fly to acquire a new skill — the library grows with use.

Resource2Skills covers seven authoring domains (slides, web pages, spreadsheets, Blender, CAD, UE5 scenes, music production); each design choice above ablates to a measurable gain.

## Why it matters

The historical ceiling on skill-library work has been *acquisition cost* — either an expert wrote the skills or the model learned them through expensive rollouts. Resource-mined skills change the ceiling from "how much expert data can we collect" to "how well can we extract from what humans already wrote," which is orders of magnitude more content. This is complementary to base-model post-training: the library is the fast, updatable, per-domain layer sitting on top of a stable model.

## Gotchas & tricks

- **Wiki organization beats flat storage.** Cross-linked skills retrieve better than a flat vector store — related-concept retrieval finds skills you didn't know to search for.
- **Source-mix curation is the dominant lever.** Random tutorials vs. curated official docs make a bigger downstream difference than the extractor architecture.
- **Multimodal parsing catches what text-only parsing misses.** Screenshots and video frames encode UI-specific procedural knowledge that text-only tutorials skip.
- **Online acquisition prevents library rot** — but rate-limit it, or the agent will burn budget re-mining every borderline task.
- **Deploy the library behind a selection LLM.** Full-library-as-context wrecks token budgets; a small selection call chooses the top-K skills to include.

## Sources

- Paper: *Resource2Skills: Distilling Executable Skills from Human-Created Resources for Software Agents* — Fan et al., Microsoft Research / UCSC / SJTU, 2026 — [arXiv:2606.29538](https://arxiv.org/abs/2606.29538).
- Precedent: *Voyager: An Open-Ended Embodied Agent with Large Language Models* — Wang et al., 2023 — self-authored skill library in Minecraft.

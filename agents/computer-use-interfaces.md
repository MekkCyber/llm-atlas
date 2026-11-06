# Hybrid Computer Use (HCU) — Programmatic + Semantic + Visual
*Depth — a unified three-modality interface for general computer-use agents.*

**TL;DR:** A computer-use agent can observe and act in three different modalities: **programmatic** (shell commands and scripts), **semantic** (accessibility-tree queries and actions), and **visual** (screenshots and click/type). Each has blind spots on its own — code APIs don't exist for every app, AX trees are brittle across apps, vision is slow and expensive. **HCU** exposes all three to the agent at once as a unified tool interface, batches AX actions/observations, and reports effects as **AX diffs** rather than full snapshots. Top Partial and Binary scores on **OSWorld 2.0**; every tested model improves when switched to HCU from a single-modality interface.

**Prereqs:** [README.md](./README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Three observation/action modalities for a computer-use agent:

- **Programmatic.** `exec("cat /tmp/foo")`, `python(...)`, `curl ...`. High fidelity, low cost, but only works where a usable API exists.
- **Semantic (AX).** Query the OS accessibility tree for structured descriptions; act via AX methods (`click(widget_id=…)`, `set_text(widget_id=…, text=…)`). Complete for standard widgets, but bloated per snapshot and fragile across apps.
- **Visual.** Screenshot in, pixel coordinates out. Universal but round-trip-heavy and expensive.

HCU gives the agent all three as **first-class tools** inside one loop, with a few engineering moves that make the combination usable.

## How it works

**Unified tool schema.** Every action is a tool call; the agent picks the modality per step by picking the tool.

**AX diff reporting.** After an AX-tree action, the environment does **not** return the full new tree. It returns the *diff* from the previous AX observation: added/removed/changed widgets. Each diff is 1–2 orders of magnitude smaller than a full snapshot, so the agent's context does not balloon over long trajectories.

**Batched AX observations/actions.** AX queries are composable — the agent can send a *list* of widget queries or actions in one call, and the environment runs them atomically. One round trip per *group* of related widget operations rather than per widget.

**Modality selection prompt.** The system prompt teaches when each modality is best: use `shell` for anything the app exposes a CLI for; use `ax` for standard UI elements; fall back to `vision` for custom-rendered canvases.

```
goal  →  agent  ─┐   shell: open terminal, cd, grep
                 ├── ax:    query widget, batch action, apply diff
                 └── vision: screenshot, click (x,y)
```

## Why it matters

The dominant benchmark (OSWorld, AndroidWorld, WebArena) pressure has pushed agent-stack research toward **a single modality** (often vision). HCU argues — with wins on OSWorld 2.0 for every model tested — that the right abstraction is **modality-pluralist**, and that cutting round-trips via AX diffs is as important as model quality. The diff-reporting idea generalizes: *every* structured-observation interface should return deltas rather than snapshots when histories are long.

## Gotchas & tricks

- AX diffs are only cheap if you *trust* your previous snapshot. Periodically force full snapshots to resync, especially after navigation events that may invalidate many widget IDs.
- Programmatic access is sometimes the *wrong* answer even when available — e.g. running a shell command instead of clicking "Save" can skip the app's own save dialog state.
- The agent will prefer whatever modality saved the last call — bias the prompt toward *trying* the cheaper modalities first per step rather than once per trajectory.
- Mixed-modality trajectories are harder to replay for reproducibility; log the exact diff/snapshot stream.

## Sources

- Paper: *Hybrid Computer Use: A Unified Programmatic, Semantic, and Visual Interface for General Agents* — Wang et al., 2026 — https://arxiv.org/abs/2610.41432
- Benchmark: *OSWorld 2.0* (the evaluation substrate).

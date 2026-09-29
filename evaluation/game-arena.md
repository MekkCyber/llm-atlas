# Kaggle Game Arena
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Game Arena is an open platform for evaluating LLMs through head-to-head competitive games. Unlike static benchmarks (which saturate as models improve and are vulnerable to contamination), games produce a *live* strength signal that scales with the field. Three pilot environments span perfect information (Chess), imperfect information (Poker), and multiplayer social deduction (Werewolf). Ground truth comes from game outcomes, not model judges.

**Prereqs:** *(none)*
**Related:** [mmlu](./mmlu.md), [aime](./aime.md), [livecodebench](./livecodebench.md)

---

## What it is

Static LLM benchmarks (MMLU, GSM8K, AIME, HumanEval, …) all have the same two problems: (1) once frontier models pass some ceiling, they cannot distinguish stronger models; (2) they leak into pretraining corpora over time, and "solved" scores may be memorization.

Game-based evaluation sidesteps both:

- **Live ranking, not fixed answers.** Two models play; the winner is decided by the game rules, not a human-labeled key. Ratings update as new models enter.
- **Contamination-resistant.** Game states are procedurally generated per match; the paper's exact matches are not in any training corpus.
- **Difficulty tracks the field.** As models improve, opponents improve; there is no ceiling to saturate.

The three pilot environments are picked to isolate different capability axes:

- **Chess** — perfect information, deep planning, well-studied strong baselines (Stockfish, humans, older AI).
- **Poker** — imperfect information, probabilistic reasoning, opponent modeling.
- **Werewolf** — multiplayer, social deduction, deception and cooperation across natural-language turns.

## How it works

- **Sandboxed environments.** Each game runs in an isolated process with strict move-parsing and time controls. Models are queried via an unbounded-context API but with fixed per-move latency budgets.
- **Ground-truth game engine.** Chess and Poker use standard engines (Stockfish for post-hoc analysis; PokerKit for equity). Werewolf runs a rule-based state manager and role-visibility system.
- **Round-robin and tournaments.** All-pairs or Swiss tournaments generate ratings (Elo, Glicko-2). Publish per-game logs so external analysts can reproduce.
- **Match transcripts as an artifact.** Every match's full move/prompt history is released — useful downstream for training or interpretability studies.
- **Kaggle infrastructure.** Public leaderboard, submission API, and a scaling-up pattern for new games as they're added.

## Why it matters

- **Ranking that scales with model strength.** No re-anchoring or human labeling as models improve.
- **Robust to prompt injection / gaming.** The scorer is the game engine; models cannot "convince the judge" they won.
- **Extensible.** New games (Diplomacy, Go, StarCraft-style micro) can be added without invalidating existing rankings.
- **Ground-truth training data.** Match transcripts are a natural source of trajectories for future RL-for-strategy training.

## Gotchas & tricks

- **Latency budget matters.** Long-thinking chain-of-thought inflates model quality per move but slows tournaments; the paper reports separate ratings at different time controls.
- **Prompt harness leaks capability.** A model with a better "how to play" system prompt scores higher, so the platform must standardize prompts.
- **Non-transitive rankings.** In imperfect-info games, model A can systematically beat B while B beats C and C beats A. Elo assumes transitivity; report full result matrices too.
- **Werewolf and multiplayer** need >2 models per match; ratings must handle team outcomes and role variance.

## Sources

- Paper: *Game Arena: Strategic LLM Evaluation in Competitive Environments* — Doerschuk-Tiberi, Yan, Chiu et al. — Google DeepMind + Kaggle, 2026 — https://arxiv.org/abs/2609.31473
- Platform: Kaggle Game Arena (open leaderboard and submission API).
- Related: *Chatbot Arena* (LMSys) — live pairwise human-preference ranking, complementary but human-judged.

# Monte Carlo Tree Search (MCTS) — for LLM Reasoning

*Depth — the classical game-playing search algorithm (Coulom 2006, AlphaZero 2018) adapted — with mixed success — to LLM reasoning.*

**TL;DR:** MCTS builds a search tree by alternating four phases: **selection** (descend via a bandit rule, usually UCB1 or PUCT), **expansion** (add a new child), **simulation/evaluation** (rollout or value-net estimate), and **backpropagation** (propagate the value up). For LLMs the hard part is not the algorithm — it's deciding **what a "node" is**: a token (too branchy), a sentence, a reasoning step (most successful), or a full solution. Successful LLM-MCTS systems (ToT, TS-LLM, ReST-MCTS*) coarsen the action to reasoning steps and borrow AlphaZero's PUCT with a learned value net or PRM. **DeepSeek-R1 tried MCTS with token-level actions and abandoned it** — token branching is exponential, and a fine-grained value model is hard to train.

**Prereqs:** basic RL (policy, value), [prm](prm.md)
**Related:** [orm](orm.md), [long-cot-rl](long-cot-rl.md), [_rewards](../_rewards.md)

---

## What it is

A family of **best-first tree-search** algorithms where the tree is grown incrementally and the node-selection rule balances exploitation (go where average reward is high) with exploration (go where we haven't visited much). The classical algorithm for MCTS is:

```
repeat many times:
    1. SELECTION     — from root, descend to a leaf via a bandit rule
    2. EXPANSION     — add one (or a few) child(ren) to the leaf
    3. SIMULATION    — play out (or estimate value) from the new node
    4. BACKPROPAGATE — update visit counts and value estimates along the path
at the end: return best child of the root by visit count (or value)
```

This was named and formulated by **Coulom (2006)**; the selection rule most people mean when they say "MCTS" — **UCT** — was introduced by **Kocsis & Szepesvári (2006)**. **AlphaGo / AlphaZero (Silver et al. 2016/2018)** replaced handcrafted rollouts with a neural network for both policy prior and value, producing the dominant modern variant used today.

The LLM world has been trying to port this to reasoning since 2023. Some success (Tree of Thoughts, Math-Shepherd-style PRM-guided search, ReST-MCTS*). Some prominent rejections (DeepSeek-R1).

---

## How it works

### UCB1 / UCT — the classical selection rule

At each non-leaf node, select the child `j` maximizing:

```
UCB1(j) = X̄_j + C · √( ln(n_p) / n_j )
```

Where:
- `X̄_j` = average reward observed through child `j`
- `n_p` = visit count of the parent
- `n_j` = visit count of the child
- `C` = exploration constant (UCB1 default: `√2`; in practice tuned per domain, often something like `C = 1` for games)

**Intuition:** the first term (`X̄_j`) is pure greed — pick the best-looking child. The second term (`√(ln n_p / n_j)`) is the exploration bonus — it's large for children you've rarely visited and shrinks as you visit them. The `log` growth in the numerator means exploration gets *slowly* stronger as the parent is revisited. Unvisited children have infinite UCB, so they're always tried at least once.

UCT = UCB1 applied inside the selection phase of MCTS. That's it.

### PUCT — the AlphaZero selection rule

AlphaGo / AlphaZero replace UCB1 with **PUCT** (Polynomial Upper Confidence Trees), which folds in a neural-net **policy prior** `P(s, a)`:

```
a* = argmax_a [ Q(s, a) + U(s, a) ]

U(s, a) = c_puct(s) · P(s, a) · √( Σ_b N(s, b) )  /  (1 + N(s, a))
```

Where:
- `Q(s, a)` = mean value observed when going through `(s, a)`
- `P(s, a)` = neural-net policy prior for action `a` at state `s`
- `N(s, a)` = visit count of edge `(s, a)`
- `c_puct(s)` = exploration coefficient, itself visit-dependent in AlphaZero:

```
c_puct(s) = log( (N_parent + c_base + 1) / c_base ) + c_init

AlphaZero values: c_base = 19,652   c_init = 1.25
```

**Intuition:** PUCT starts with the neural-net policy prior as the search guide; the `Q`-term overrides the prior as visits accumulate. A high-prior move is tried early; a low-prior move will eventually be tried if visit counts grow enough, but only if the budget permits.

Rollouts are **not** used for leaf evaluation in AlphaZero — the value network `V(s)` directly estimates the state's value. This is crucial: learned value replaces Monte-Carlo simulation, dramatically cutting per-search compute at the cost of needing to train `V`.

### The four phases, annotated

```
SELECTION:
    Start at root. While current node is non-leaf:
        pick child via UCB1 (classical) or PUCT (AlphaZero).
    Result: a leaf node `L` in the current tree.

EXPANSION:
    Generate one (or more) children of `L`.
    In AlphaZero: query policy net to get priors P(s, a); add all legal children.
    In UCT: add one child, chosen uniformly or by a rollout policy.

SIMULATION / EVALUATION:
    Estimate value v of the expanded node.
    In UCT: simulate ("rollout") to terminal state with a default policy; use outcome as v.
    In AlphaZero: query the value network for v (no rollouts).

BACKPROPAGATION:
    Walk back up the path. For each visited (s, a):
        N(s, a) += 1
        Q(s, a) += (v - Q(s, a)) / N(s, a)
```

After a fixed search budget (number of simulations), pick the best move from the root — either highest visit count (more robust to value noise) or highest `Q`.

---

## LLMs and MCTS — the adaptation problem

The classical algorithm assumes a **well-defined state space with bounded branching**. Chess has ~30–50 moves per position; Go has ~250. LLMs' raw action space is **the vocabulary** — 32k to 256k tokens per step. Naive MCTS at the token level fails for one simple reason: the branching factor is exponential in the response length, and visit counts spread too thin to learn anything.

All successful LLM-MCTS systems solve this by **coarsening the action**. The trick is choosing the granularity: too fine (token) gives exponential branching; too coarse (full solution) collapses MCTS into best-of-N.

### Action granularities used in the literature

| System | Node / action | Value source | Notes |
| --- | --- | --- | --- |
| **Tree of Thoughts** (Yao 2023, BFS/DFS — not strictly MCTS) | Variable: "line of equation" (Game of 24), "paragraph" (writing), "few words" (crosswords) | LLM self-evaluation (vote/score prompt) | Prunes with top-b or state-eval threshold |
| **TS-LLM** (Feng 2023) | Sentence-level or token-level (configurable) | Learned value network (AlphaZero-style) | Subsample `w` children during expansion |
| **Math-Shepherd search** (Wang 2024) | Reasoning step | [PRM](prm.md) as step-level value | Step boundaries from format |
| **ReST-MCTS*** (Zhang 2024) | Reasoning step (sentence-scale) | Learned PRM used as value | Self-bootstraps: search to generate training data for both policy and PRM |
| **AlphaZero-for-reasoning (various)** | Sentence or step | Learned value net (PUCT) | Uses c_puct / AlphaZero machinery |

The design principle that works: **a node is a unit the LLM can generate and evaluate meaningfully** — large enough that the LLM can judge its prospect (too small: "does the token `the` help?" is incoherent), small enough that you can enumerate candidates (too large: "is this whole essay good?" reduces to best-of-N).

### Why DeepSeek-R1 dropped MCTS

R1's paper explicitly tries AlphaZero-style MCTS for test-time scaling. Their reasons for abandoning it (from §4.2, "Unsuccessful Attempts"):

1. **Exponential token branching.** *"Unlike chess, where the search space is relatively well-defined, token generation presents an exponentially larger search space."*
2. **Cap-induced local optima.** *"To address this, we set a maximum extension limit for each node, but this can lead to the model getting stuck in local optima."*
3. **Fine-grained value model is hard.** *"Training a fine-grained value model is inherently difficult."* — the same problem that kills PRMs as RL rewards ([prm](prm.md)).
4. **AlphaGo-style iterative improvement doesn't translate.** *"While AlphaGo's core success relied on training a value model to progressively enhance its performance, this principle proves difficult to replicate in our setup due to the complexities of token generation."*

R1's conclusion: outcome-only rule-based reward + [long-CoT RL](long-cot-rl.md) gave them the test-time-compute scaling they wanted, without a search apparatus. The MCTS machinery is *not necessary* — the model can learn to "search in its own CoT" via RL. This is arguably the single most important framing-shift in LLM reasoning post-2024.

Other groups disagree. **ReST-MCTS*** and related work continue to show gains from tree search combined with PRM-guided value, especially on hard math with structured step boundaries. The debate is not settled — it's a bet on whether coarsening the action space to reasoning steps can tame the branching problem enough for MCTS to pay off at scale.

---

## Why it matters

- **Test-time compute scaling through search.** Best-of-N samples independent completions; MCTS reuses shared prefixes and concentrates budget on promising branches. In principle, more efficient than best-of-N for the same budget.
- **Local credit assignment during inference.** A node's value is the average reward of paths through it — error localization comes for free in the search structure. This pairs naturally with a [PRM](prm.md).
- **Connects LLM reasoning to the AlphaGo lineage.** The appeal is architectural: if self-play + MCTS solved Go, maybe it can solve reasoning. As of 2025, the empirical evidence is mixed — MCTS helps on some benchmarks, fails on others.
- **Still the default for well-structured search problems** where the action space is discrete and bounded: tool-augmented agents (fixed tool set), code repair (bounded edit types), planning over known graphs.

---

## Gotchas & tricks

- **Branching factor is destiny.** If your action is a token, you have ~50k branches per step and a budget of ~1k simulations — you visit almost nothing. Either coarsen the action or use top-K filtered expansion. TS-LLM subsamples `w` children from the policy prior; ToT keeps top-b evaluated states.
- **You need a value signal.** Classical MCTS uses rollouts to terminal; for LLM reasoning, rollouts are expensive and often uninformative (a random continuation rarely reaches the correct answer). You almost always need a **learned value** — either a PRM, an ORM, or a dedicated value net. Training it is a whole sub-project.
- **AlphaZero's PUCT ≠ UCT.** If you see "MCTS with c_puct = 1.25" in an LLM paper, they mean AlphaZero-style PUCT with a neural-net prior, not raw UCT. The two have different exploration characteristics and different failure modes.
- **Local optima from node caps.** Limiting expansion per node controls cost but can trap search. Mitigate with dynamic caps, stochastic expansion, or progressive widening.
- **Self-improvement loops are brittle.** ReST-MCTS* and similar systems iterate: search to produce training data, train policy + value on that data, re-search. Drift and narrowing are real — keep a held-out eval.
- **MCTS vs sampling: measure compute parity.** MCTS's headline gains are sometimes against best-of-N with a far smaller budget. Compare at equal FLOPs or equal wall-clock.
- **The R1 takeaway.** The cleanest post-R1 framing is: if you can train the policy to allocate its own test-time compute (via [long-CoT RL](long-cot-rl.md)), you don't need an external search structure. MCTS becomes valuable when (a) the reward function is well-structured enough that a value model can be trained cheaply, or (b) the task has discrete, low-branching actions (tool use, planning). Otherwise, the compute is usually better spent on more policy RL.

---

## Sources

- Paper: *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search* — Coulom, 2006 — coins "Monte-Carlo Tree Search" and defines the four phases.
- Paper: *Bandit based Monte-Carlo Planning* — Kocsis & Szepesvári, 2006 — introduces UCT (UCB1 applied to trees).
- Paper: *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* — Silver et al., DeepMind, 2018 (*Science* 362:1140–1144; [arXiv 1712.01815](https://arxiv.org/abs/1712.01815)) — AlphaZero, PUCT selection with `c_base = 19,652`, `c_init = 1.25`, learned policy + value net, no rollouts.
- Paper: *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* — Yao et al., 2023, [arXiv 2305.10601](https://arxiv.org/abs/2305.10601) — BFS/DFS over LLM-generated "thoughts" with self-evaluation; MCTS listed as future work.
- Paper: *Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training* (TS-LLM) — Feng et al., 2023, [arXiv 2309.17179](https://arxiv.org/abs/2309.17179) — AlphaZero-style PUCT with learned value net; sentence-level or token-level actions.
- Paper: *ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search* — Zhang et al., 2024, [arXiv 2406.03816](https://arxiv.org/abs/2406.03816) — MCTS over reasoning steps guided by a learned PRM.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — §4.2 "Unsuccessful Attempts" explains why R1 dropped MCTS.
- Textbook reference: Sutton & Barto, *Reinforcement Learning: An Introduction* (2018), §8 for MCTS basics.

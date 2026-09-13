# Generate-Verify-Refine (Natural-Language Proof Search)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A test-time-compute pipeline for hard reasoning that iteratively **generates** candidate solutions, **verifies** them with a second checkpoint acting as a critic, and **refines** flawed candidates in-place — all in natural language, with no formal prover, tools, or internet. A final high-compute stage selects one submission per problem. Nemotron's IMO-Gold pipeline used two RL/SFT-specialized Nemotron 3 Ultra checkpoints and scored **30/42 at IMO 2026 (gold threshold)** using this loop.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [../rlvr.md](../rlvr.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [orm.md](./orm.md), [prm.md](./prm.md), [mcts.md](./mcts.md), [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md)

---

## What it is

Two specialists cooperating at inference time:

- **Generator** (SFT-heavy checkpoint) — writes candidate proofs / solutions as long chains of thought.
- **Verifier / refiner** (RL-heavy checkpoint) — reads a candidate, decides whether it is correct, and if not, produces a *repaired* candidate rather than an accept/reject signal.

At submission time, a separate **high-compute selector** stage chooses the final answer from the surviving candidates. The loop is entirely in natural language — no Lean/Coq, no external tool.

## How it works

Per problem:

1. Sample $N$ initial candidates from the generator with high temperature.
2. Verifier reads each candidate → outputs (verdict, critique). If "correct," pass through; if "flawed," emit a repair.
3. Refined candidates re-enter the pool (bounded by a fixed number of refinement rounds).
4. After the loop, a final selector — often the generator at a higher compute budget, with the verifier's critique as context — picks one submission.

Two things distinguish this from vanilla best-of-$N$ rejection sampling:

- The verifier **refines** rather than merely rejects. A single expensive rollout that got 90% of the way to a correct proof is repaired, not discarded.
- Both roles are separately post-trained. The verifier is not the same policy as the generator; it's an RL specialist whose training reward is verifier-utility, not proof-writing quality.

## Why it matters

- **Open-weights parity with closed frontier math systems.** The 30/42 IMO 2026 score matches gold-medal claims from closed systems (DeepMind AlphaProof, OpenAI's math specialists) without a formal prover, using an entirely reproducible recipe.
- **Transfers off math.** Generate-verify-refine only requires that (a) you can score candidates and (b) refining is cheaper than regenerating. Both hold for code, formal verification, structured extraction, and long-form technical writing.
- **Concrete inference-time-RL playbook.** RL has mostly been used for training-time policy improvement; this paper makes explicit the pattern where post-training produces *two* specialists that then cooperate at inference.

## Gotchas & tricks

- **Verifier drift is the failure mode.** If the verifier gets over-confident, it stops refining and everything passes through. Diverse initial samples + a temperature-aware verifier help.
- **Refinement rounds saturate.** Empirically the paper stops after a small number of rounds; more rounds trade compute for diminishing returns as the pool converges.
- **Selector budget dominates.** The final high-compute stage often eats more compute than all the earlier refinement, because it re-derives each candidate with the verifier's critique as context.
- **Verifier training reward isn't just "accuracy."** The verifier is rewarded for producing *useful critiques* — comments that lead to correct repairs — not just calling the right verdict. Framing the reward matters more than the algorithm.
- **Not the same as PRM search.** Process reward models (see [prm.md](./prm.md)) score partial traces during generation; here the verifier reads *complete* candidates and rewrites them.
- **Contamination risk on IMO-style benchmarks.** Nemotron-IMO-Bench (200 novel problems, released alongside the paper) is the counter-move; watch for it as the new standard.

## Sources

- Paper: *An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics* — Moshkov, Ge, Armstrong, Du, Mahdavi, Gitman (NVIDIA), 2026 — arXiv:2609.10712. Released: post-trained checkpoints, training data, training + inference code, submitted solutions, Nemotron-IMO-Bench.

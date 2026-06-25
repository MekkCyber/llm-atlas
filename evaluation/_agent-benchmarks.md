# Agent Benchmarks

*Taxonomy — benchmarks that evaluate LLM agents on multi-step, tool-using, or environment-grounded tasks.*

**TL;DR:** An agent benchmark differs from a static eval in that the agent must take *actions* and the environment *responds* — so success depends on multi-step decision-making, not single-shot prediction. The space splits along three axes: substrate (code repo vs. browser vs. mobile UI vs. OS), task source (synthetic vs. real-world repos vs. published research), and verifier (unit test vs. accessibility check vs. published metric). The 2025–2026 frontier is moving away from synthetic unit-test benchmarks toward verifiers grounded in real-world ground truth.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [naturebench](naturebench.md) · [livecodebench](livecodebench.md)

---

## The problem

Static benchmarks (MMLU, HumanEval, MATH) evaluate single-turn prediction. Agents take long action trajectories; the failure modes — planning errors, recovery from tool errors, context exhaustion, partial credit — are invisible to a static eval. Agent benchmarks need an *environment* and a *verifier* that runs over the trajectory, not the final token.

Three challenges recur across the space:

- **Contamination.** GitHub issues are public training data; SWE-bench solutions can leak.
- **Brittleness.** Real environments break in irrelevant ways (network errors, missing system packages), making leaderboards noisy.
- **Cost.** A 30-minute trajectory per task × thousands of tasks × dozens of models is genuinely expensive.

## The shared pattern

Each agent benchmark is a triple:

```
(environment, task distribution, verifier)
```

- The **environment** is what the agent interacts with: a sandboxed shell + repo, a browser, an OS, a mobile emulator.
- The **task distribution** is what the agent is asked to do: fix bugs, complete forms, reproduce experimental results.
- The **verifier** is how success is graded: hidden unit tests, accessibility-tree checks against a target, comparison to a published metric.

Variants differ in how rich each component is — and in how well they survive the contamination defense.

## Variants

| Benchmark | Environment | Tasks | Verifier | Status |
| --- | --- | --- | --- | --- |
| SWE-bench / SWE-bench Verified | repo + sandbox | GitHub issues | hidden unit tests | dominant for coding agents; contamination risk |
| [LiveCodeBench](livecodebench.md) | repo + sandbox | rolling-date programming problems | unit tests | contamination defense via date cutoff |
| [NatureBench](naturebench.md) | repo + sandbox + data | reproduce Nature-family results | match published metric | new; strongest contamination defense |
| TerminalBench 1.0 / 2.0 | shell | terminal tasks | exit code + output check | broad shell-skill coverage |
| OSWorld / OSWorld-Verified | full OS sandbox | desktop workflows | accessibility-tree match | OS-level GUI agents |
| WebArena / Mind2Web | browser | web tasks | DOM check or visual match | browser agents |
| Tool Decathlon | tool sandbox | structured tool-use tasks | tool-call schema match | function-calling agents |
| MemGUI-Bench / MobileWorld | mobile emulator | mobile UI tasks | screen state check | mobile GUI agents |

## How to choose

- **For coding agents**: SWE-bench Verified for headline number, NatureBench for the harder long-tail, LiveCodeBench for contamination-controlled.
- **For browser agents**: WebArena remains the default; Mind2Web for breadth.
- **For mobile/desktop**: OSWorld-Verified for desktop, MemGUI-Bench or MobileWorld for mobile.
- **For tool-use**: Tool Decathlon if you want structured tool calls, TerminalBench for shell.
- **For ambition**: report on a *mix* — agent capability is a vector, not a scalar, and any single benchmark over-fits one axis.

## Adjacent but distinct

- **Static reasoning evals** (MMLU, MATH, AIME) — single-turn, no environment.
- **Reward-model evals** (RewardBench class) — score a scorer, not an agent.
- **Safety / red-team evals** (HarmBench class) — focus on refusal, not capability.

## Sources

- Paper: *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* — Jimenez et al., 2023.
- Paper: *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code* — Jain et al., 2024.
- Paper: *NatureBench* — Wang et al., 2026 — [arXiv:2606.24530](https://arxiv.org/abs/2606.24530).
- Paper: *OSWorld* — Xie et al., 2024.
- Paper: *WebArena* — Zhou et al., 2023.

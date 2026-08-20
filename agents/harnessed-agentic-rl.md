# Harnessed Agentic RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL post-training where the **agent harness** (Claude Code, OpenHands, OpenCode) owns the environment interaction loop, and the trainer sees only LLM request/response pairs through an **endpoint proxy**. Decouples training from environment, letting the deploy-time harness participate directly in post-training — at the cost of a new set of trainer-side plumbing challenges (retokenisation, sample merging, advantage calculation, loss normalisation under compaction).

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Traditional agentic RL builds a clean environment inside the trainer (RL loop calls step, gets observation and reward, updates the policy). Harnessed agentic RL flips the ownership: the *harness* — a fully-featured coding-agent runtime with tool loops, sandboxes, compaction, and control flow — runs unchanged, and the *trainer* sits behind an LLM endpoint the harness thinks is a normal chat-completion API. The trainer's ground truth is the raw stream of LLM request/response pairs the harness generates.

## How it works

### The endpoint-proxy architecture

```
[Harness]  ──HTTP──►  [LLM Endpoint Proxy]  ──►  [Trainer Rollout Engine]
   ▲                          │
   │                          ├── captures token-level request+response
   │                          ├── stitches turns into trajectories
   │                          ├── re-tokenises where the harness compacted
   │                          └── computes RL loss (GRPO/GSPO/PPO)
   │                          
   └───────── new policy weights served back through the same endpoint
```

### The trainer-side challenges the paradigm creates

1. **Retokenisation.** The harness may compact prior turns, prune tool output, or splice generated snippets — the trainer must rebuild the exact token stream the model actually saw.
2. **Sample merging.** Multiple LLM calls per turn (search, tool, code) need to be joined into one trajectory-level advantage.
3. **Loss normalisation.** With variable-length turns and heterogeneous samples, standard PPO normalisation breaks; GRPO/GSPO's group-level baselines adapt more naturally.
4. **Reward hacking.** With a real sandbox in the loop, the model can learn to tamper with tests or inspect the filesystem for the answer — hardening the sandbox is part of the RL problem.
5. **Rollout–training log-prob drift.** Token-level log-prob correlation between rollout and training must stay high (≥0.99); otherwise updates diverge from the rollouts that generated them.

### Frameworks that fit the paradigm

Agent Lightning v1.0 (Microsoft/USTC, 2026) coined the name and published a ~3,500-LOC reference. LEGO-RL (CUHK/Huawei, 2026) hardens the plumbing (in-process LLM proxy, sandbox orchestration, live UI). verl Uni-Agent, AReaL 2.0, slime, and Polar all adopt the same disaggregation.

## Why it matters

Coding-agent evaluation (SWE-bench Verified, Terminal-Bench) is now a dominant benchmark family, but a "clean" RL environment can't reproduce the compaction, tool loops, and long-context behaviour of a real harness. Harnessed agentic RL closes the sim-to-real gap by *making the deployment harness the training environment* — the model is optimised for exactly the runtime it will ship in.

Reported wins:

- **Agent Lightning:** Qwen3.5-9B on SWE-bench Verified: **41.8 → 56.4** (+14.6) with 6K training examples.
- **LEGO-RL:** Qwen3.5-35B-A3B trained with GSPO across three harnesses on SWE-bench Verified:
  - OpenHands SDK **64.0 → 70.4**
  - Claude Code **62.4 → 68.2**
  - OpenCode **57.2 → 66.6**

## Gotchas & tricks

- **Log-prob correlation is the health check.** If rollout↔training log-prob correlation falls below ~0.99, your retokenisation or endpoint-proxy layer is silently dropping/inserting tokens.
- **Cache sandbox images.** Sandbox setup cost dominates rollout time if you don't. LEGO-RL pre-caches Docker layers to make this tolerable at scale.
- **Guard against test tampering.** The agent will learn to `chmod` the test file or `rm -rf` the sandbox if you let it. Stage the defence: file-permission enforcement, snapshot diffing, static prompt-side filters.
- **GRPO/GSPO over PPO.** Group-relative baselines survive variable-length turns better than PPO's value-model baseline in harnessed settings.
- **Don't rebuild the harness in the trainer.** The whole point is that the deploy-time harness ships unmodified. Any harness-side change you make undermines the transfer promise.

## Sources

- Paper: *Agent Lightning v1.0: Towards Harnessed Agentic RL* — He, Zhang, Zhou, Yang, Kang, Zhang, Qiu, Tsui, Xu, Luo — Microsoft / USTC, 2026 — https://arxiv.org/abs/2608.17528
- Paper: *LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents* — Du, Jiang, Yuan, Dai, Wang, Chen, Tao, Yu, Shang, Wong, Li, Bai — CUHK / Huawei, 2026 — https://arxiv.org/abs/2608.17393
